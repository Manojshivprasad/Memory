"""Memory evaluation harness.

    python -m eval.harness --model openrouter/openai/gpt-4.1-nano [--only work-software] [--sleep] [--tag baseline]

For every scenario (see eval/scenarios.py), against the real API + Postgres + Redis (in-process):
  1. each chat session: per turn  retrieve(query) -> commit(turn) -> worker forms memory   (same path as proxy mode)
  2. close + consolidate each chat (episode + brief)
  3. score what was STORED:   expected facts present & active, stale facts not active, noise absent, bloat
  4. score what is RECALLED:  probes asked in a NEW chat of the project (must / must_not in injected memory)
  5. cross-project probes:     project facts must not leak into another project; user-level prefs should
Writes eval/reports/<tag>-<time>.json and .md, and prints a summary. Stop any running worker first
(it would consume the harness's turns).
"""

import argparse
import asyncio
import json
import os
import time
import uuid
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path


def _configure_env(args: argparse.Namespace) -> None:
    os.environ["MEMORY_LLM_MODEL"] = args.model
    os.environ["MEMORY_AGENT_LLM_MODEL"] = args.agent_model or args.model
    os.environ.setdefault("MEMORY_EMBEDDER", "fastembed")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")


# ----------------------------------------------------------------------------- LLM usage accounting
USAGE = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0, "failures": 0}


def _install_usage_meter() -> None:
    import litellm

    from app import llm as llm_mod
    from app.runtime.agent import memory_agent

    orig = llm_mod.acompletion

    async def metered(messages, **kw):  # noqa: ANN001, ANN202
        try:
            resp = await orig(messages, **kw)
        except Exception:
            USAGE["failures"] += 1
            raise
        USAGE["calls"] += 1
        u = getattr(resp, "usage", None)
        if u:
            USAGE["prompt_tokens"] += u.prompt_tokens or 0
            USAGE["completion_tokens"] += u.completion_tokens or 0
        try:
            USAGE["cost_usd"] += float(litellm.completion_cost(completion_response=resp) or 0)
        except Exception:  # noqa: BLE001 - price unknown to LiteLLM
            pass
        return resp

    llm_mod.acompletion = metered
    memory_agent.acompletion = metered


# ----------------------------------------------------------------------------- matching helpers
def group_hit(text: str, group: list[str]) -> bool:
    t = text.lower()
    return any(alt.lower() in t for alt in group)


def fact_hit(text: str, fact: list[list[str]]) -> bool:
    return all(group_hit(text, g) for g in fact)


# ----------------------------------------------------------------------------- runner
class Runner:
    def __init__(self, client, drain, svc: dict, user: str, agent: str) -> None:  # noqa: ANN001
        self.c = client
        self.drain = drain
        self.svc = svc
        self.user = user
        self.agent = agent

    async def post(self, path: str, body: dict | None = None, **params):  # noqa: ANN003, ANN201
        r = await self.c.post(path, json=body, params={"user_id": self.user, **params}, headers=self.svc)
        r.raise_for_status()
        return r.json()

    async def get(self, path: str, **params):  # noqa: ANN003, ANN201
        r = await self.c.get(path, params={"user_id": self.user, **params}, headers=self.svc)
        r.raise_for_status()
        return r.json()

    async def run_session(self, pid: str, turns: list[tuple[str, str]]) -> str:
        chat = await self.post(f"/v1/projects/{pid}/chats", {"title": ""})
        for user_msg, assistant_msg in turns:
            # 1) recall exactly like the proxy does (this also records what was retrieved for reconsolidation)
            await self.post("/v1/memory/retrieve", {"project_id": pid, "chat_id": chat["id"], "user_id": self.user,
                                                     "query": user_msg, "agent": "off"})
            # 2) commit the turn; 3) let the worker form memory before the next turn (keeps order realistic)
            await self.post("/v1/memory/commit", {"project_id": pid, "chat_id": chat["id"], "user_id": self.user,
                                                   "user_msg": user_msg, "assistant_msg": assistant_msg})
            await self.drain()
        await self.post(f"/v1/chats/{chat['id']}/close", wait="true")
        return chat["id"]

    async def probe(self, pid: str, chat_id: str, query: str) -> dict:
        return await self.post("/v1/memory/retrieve", {"project_id": pid, "chat_id": chat_id, "user_id": self.user,
                                                        "query": query, "agent": self.agent})


async def evaluate_scenario(rn: Runner, sc, do_sleep: bool) -> dict:  # noqa: ANN001
    t0 = time.perf_counter()
    proj = await rn.post("/v1/projects", {"name": sc.project, "description": ""})
    pid = proj["id"]
    for turns in sc.sessions:
        await rn.run_session(pid, turns)
    if do_sleep:
        await rn.post(f"/v1/projects/{pid}/sleep")

    mems_all = await rn.get("/v1/memory", project_id=pid, include_invalid="true", include_archived="true", limit=500)
    active = [m for m in mems_all if m["valid_until"] is None and m["status"] != "archived"]
    project_active = [m for m in active if m["scope"] != "user"]
    brief = (await rn.get(f"/v1/projects/{pid}/brief"))["brief"]

    stored = []
    for label, fact in sc.expect_stored:
        # facts must exist as semantic memories, not only inside a past-chat summary
        hits = [m["content"] for m in active if m["kind"] != "episode" and fact_hit(m["content"], fact)]
        stored.append({"label": label, "ok": bool(hits), "matched": hits[:3]})
    superseded = []
    for label, fact in sc.expect_superseded:
        # Past-chat episodes are history ("back then we used Postgres"): judged by probes (leakage), not here.
        still_active = [m["content"] for m in active if m["kind"] != "episode" and fact_hit(m["content"], fact)]
        in_episodes = [m["content"] for m in active if m["kind"] == "episode" and fact_hit(m["content"], fact)]
        closed = [m["content"] for m in mems_all if m["valid_until"] and fact_hit(m["content"], fact)]
        superseded.append({"label": label, "ok": not still_active, "still_active": still_active,
                           "history_kept": bool(closed), "in_episodes": len(in_episodes)})
    absent = []
    for label, fact in sc.expect_absent:
        noise = [m["content"] for m in active if fact_hit(m["content"], fact)]
        absent.append({"label": label, "ok": not noise, "found": noise})

    probe_chat = await rn.post(f"/v1/projects/{pid}/chats", {"title": "probe"})
    probes = []
    for p in sc.probes:
        res = await rn.probe(pid, probe_chat["id"], p.query)
        mem_text = res["context"]
        full = mem_text + "\n" + res["brief"]
        missing = [g for g in p.must if not group_hit(full, g)]
        leaked = [g for g in p.must_not if group_hit(full, g)]
        probes.append({
            "query": p.query, "note": p.note, "ok": not missing and not leaked,
            "recall_ok": not missing, "precision_ok": not leaked,
            "missing": missing, "leaked": leaked,
            "from_memories_only": not [g for g in p.must if not group_hit(mem_text, g)],
            "tokens": res["tokens"], "gate": res["gate"], "latency_ms": res["latency_ms"].get("total"),
            "injected": [m["content"] for m in res["memories"]], "brief_used": bool(res["brief"]),
        })

    return {
        "id": sc.id, "domain": sc.domain, "seconds": round(time.perf_counter() - t0, 1),
        "stored": stored, "superseded": superseded, "absent": absent, "probes": probes,
        "memory_count": len(project_active), "max_memories": sc.max_memories,
        "bloat": len(project_active) > sc.max_memories,
        "memories": [{"kind": m["kind"], "content": m["content"], "scope": m["scope"], "belief": m["is_belief"],
                      "closed": bool(m["valid_until"]), "status": m["status"]} for m in mems_all],
        "brief": brief, "project_id": pid,
    }


def summarize(results: list[dict], cross: list[dict]) -> dict:
    def rate(items: list[bool]) -> float | None:
        return round(sum(items) / len(items), 3) if items else None

    stored = [s["ok"] for r in results for s in r["stored"]]
    sup = [s["ok"] for r in results for s in r["superseded"]]
    hist = [s["history_kept"] for r in results for s in r["superseded"]]
    absent = [s["ok"] for r in results for s in r["absent"]]
    p_all = [p for r in results for p in r["probes"]]
    by_domain: dict[str, dict] = defaultdict(lambda: {"stored": [], "probes": []})
    for r in results:
        by_domain[r["domain"]]["stored"] += [s["ok"] for s in r["stored"]]
        by_domain[r["domain"]]["probes"] += [p["ok"] for p in r["probes"]]
    tokens = [p["tokens"] for p in p_all]
    return {
        "capture_recall": rate(stored),
        "stale_facts_retired": rate(sup),
        "stale_history_kept": rate(hist),
        "noise_rejected": rate(absent),
        "probe_pass": rate([p["ok"] for p in p_all]),
        "probe_recall": rate([p["recall_ok"] for p in p_all]),
        "probe_precision": rate([p["precision_ok"] for p in p_all]),
        "probe_recall_from_memories": rate([p["from_memories_only"] for p in p_all if p["recall_ok"]]),
        "cross_project_pass": rate([c["ok"] for c in cross]),
        "bloated_scenarios": [r["id"] for r in results if r["bloat"]],
        "avg_memories_per_scenario": round(sum(r["memory_count"] for r in results) / max(1, len(results)), 1),
        "avg_context_tokens": round(sum(tokens) / max(1, len(tokens)), 1),
        "by_domain": {d: {"capture": rate(v["stored"]), "probes": rate(v["probes"])} for d, v in by_domain.items()},
        "llm": dict(USAGE, cost_usd=round(USAGE["cost_usd"], 4)),
    }


def to_markdown(summary: dict, results: list[dict], cross: list[dict], meta: dict) -> str:
    L = [f"# Memory harness report: {meta['tag']}", "",
         f"model `{meta['model']}` · agent `{meta['agent']}` · {meta['when']} · {meta['seconds']}s", "", "## Summary", ""]
    for k in ("capture_recall", "stale_facts_retired", "stale_history_kept", "noise_rejected", "probe_pass",
              "probe_recall", "probe_precision", "probe_recall_from_memories", "cross_project_pass",
              "avg_memories_per_scenario", "avg_context_tokens", "bloated_scenarios"):
        L.append(f"- **{k}**: {summary[k]}")
    L += ["", f"- **LLM**: {summary['llm']}", "", "| domain | capture | probes |", "|---|---|---|"]
    for d, v in summary["by_domain"].items():
        L.append(f"| {d} | {v['capture']} | {v['probes']} |")
    L += ["", "## Failures", ""]
    for r in results:
        fails = ([f"stored ✗ {s['label']}" for s in r["stored"] if not s["ok"]]
                 + [f"stale still active ✗ {s['label']}: {s['still_active']}" for s in r["superseded"] if not s["ok"]]
                 + [f"noise stored ✗ {s['label']}: {s['found']}" for s in r["absent"] if not s["ok"]]
                 + [f"probe ✗ “{p['query']}” missing={p['missing']} leaked={p['leaked']} injected={p['injected']}"
                    for p in r["probes"] if not p["ok"]]
                 + ([f"bloat: {r['memory_count']} memories > {r['max_memories']}"] if r["bloat"] else []))
        if fails:
            L.append(f"### {r['id']} ({r['domain']})")
            L += [f"- {f}" for f in fails]
            L.append("<details><summary>memories</summary>\n")
            L += [f"  - [{m['kind']}{', user' if m['scope'] == 'user' else ''}{', belief' if m['belief'] else ''}"
                  f"{', closed' if m['closed'] else ''}] {m['content']}" for m in r["memories"]]
            L.append("\n</details>\n")
    for c in cross:
        if not c["ok"]:
            L.append(f"- cross-project ✗ “{c['query']}” missing={c['missing']} leaked={c['leaked']} injected={c['injected']}")
    return "\n".join(L)


async def main(args: argparse.Namespace) -> None:
    _configure_env(args)
    import httpx

    from app.embeddings import warmup
    from app.gateway.providers import warm
    from app.main import app
    from app.workers.worker import TurnConsumer, ensure_group
    from eval.scenarios import CROSS_PROJECT_PROBES, SCENARIOS

    warm()
    _install_usage_meter()
    warmup()
    await ensure_group()
    consumer = TurnConsumer(f"eval-{uuid.uuid4().hex[:6]}")

    async def drain() -> None:
        while await consumer.read_once(block_ms=300):
            pass

    t0 = time.perf_counter()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://eval", timeout=120) as c:
        reg = (await c.post("/v1/auth/register", json={"tenant_name": "eval", "password": "pw-12345678",
                                                        "email": f"eval-{uuid.uuid4().hex[:8]}@example.com"})).json()
        admin = {"Authorization": f"Bearer {reg['access_token']}"}
        key = (await c.post("/v1/api-keys", json={"name": "eval"}, headers=admin)).json()["api_key"]
        tok = (await c.post("/v1/auth/token", json={"api_key": key})).json()["access_token"]
        rn = Runner(c, drain, {"Authorization": f"Bearer {tok}"}, "eval-user", args.agent)

        scenarios = [s for s in SCENARIOS if not args.only or s.id in args.only.split(",")]
        results = []
        for sc in scenarios:
            r = await evaluate_scenario(rn, sc, args.sleep)
            ok_s = sum(s["ok"] for s in r["stored"])
            ok_p = sum(p["ok"] for p in r["probes"])
            print(f"{sc.id:20} stored {ok_s}/{len(r['stored'])}  probes {ok_p}/{len(r['probes'])}  "
                  f"memories {r['memory_count']}  {r['seconds']}s", flush=True)
            results.append(r)

        cross = []
        if not args.only:
            other = await rn.post("/v1/projects", {"name": "Unrelated project", "description": ""})
            chat = await rn.post(f"/v1/projects/{other['id']}/chats", {"title": "probe"})
            for p in CROSS_PROJECT_PROBES:
                res = await rn.probe(other["id"], chat["id"], p.query)
                full = res["context"] + "\n" + res["brief"]
                missing = [g for g in p.must if not group_hit(full, g)]
                leaked = [g for g in p.must_not if group_hit(full, g)]
                cross.append({"query": p.query, "note": p.note, "ok": not missing and not leaked, "missing": missing,
                              "leaked": leaked, "injected": [m["content"] for m in res["memories"]]})

    summary = summarize(results, cross)
    meta = {"tag": args.tag, "model": args.model, "agent": args.agent, "when": datetime.now().isoformat(timespec="seconds"),
            "seconds": round(time.perf_counter() - t0, 1)}
    out = Path(__file__).parent / "reports"
    out.mkdir(exist_ok=True)
    stem = f"{args.tag}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    (out / f"{stem}.json").write_text(json.dumps({"meta": meta, "summary": summary, "results": results, "cross": cross},
                                                 indent=2, default=str), encoding="utf-8")
    (out / f"{stem}.md").write_text(to_markdown(summary, results, cross, meta), encoding="utf-8")
    print("\n" + json.dumps({k: v for k, v in summary.items() if k != "by_domain"}, indent=2))
    print("by domain:", json.dumps(summary["by_domain"]))
    print(f"report: eval/reports/{stem}.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openrouter/openai/gpt-4o-mini")
    ap.add_argument("--agent-model", default=None)
    ap.add_argument("--agent", default="off", choices=["off", "auto", "always"])
    ap.add_argument("--only", default="")
    ap.add_argument("--sleep", action="store_true", help="run nightly consolidation before probing")
    ap.add_argument("--tag", default="run")
    asyncio.run(main(ap.parse_args()))
