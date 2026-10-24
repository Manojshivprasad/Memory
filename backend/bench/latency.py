"""Hot-path latency benchmark (runtime overhead only, no LLM).

    python -m bench.latency --memories 1000 --queries 200

Seeds one project with N memories via the explicit write path, then measures:
  prefetch tick, submit with a warm prefetch pool, cold submit.
"""

import argparse
import asyncio
import random
import statistics
import time
import uuid

import httpx

from app.embeddings import warmup
from app.main import app

TOPICS = ["database", "auth", "billing", "deploy", "cache", "search", "email", "logging", "queue", "frontend"]
TECH = ["postgres", "redis", "kafka", "fastapi", "next.js", "stripe", "sendgrid", "sentry", "docker", "aws"]


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(len(xs) * p))], 2)


async def main(n_mem: int, n_q: int) -> None:
    warmup()
    c = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://bench", timeout=60)
    email = f"bench-{uuid.uuid4().hex[:8]}@example.com"
    r = await c.post("/v1/auth/register", json={"tenant_name": "bench", "email": email, "password": "pw-12345678"})
    admin = {"Authorization": f"Bearer {r.json()['access_token']}"}
    key = (await c.post("/v1/api-keys", json={"name": "b"}, headers=admin)).json()["api_key"]
    svc = {"Authorization": f"Bearer {(await c.post('/v1/auth/token', json={'api_key': key})).json()['access_token']}"}
    q = {"user_id": "bench-user"}
    pid = (await c.post("/v1/projects", json={"name": "bench"}, params=q, headers=svc)).json()["id"]
    cid = (await c.post(f"/v1/projects/{pid}/chats", json={}, params=q, headers=svc)).json()["id"]

    print(f"seeding {n_mem} memories ...")
    rnd = random.Random(7)
    for i in range(n_mem):
        t, x = rnd.choice(TOPICS), rnd.choice(TECH)
        await c.post("/v1/memory/remember", headers=svc, json={
            "project_id": pid, "user_id": "bench-user", "importance": rnd.random(),
            "content": f"Note {i}: the {t} component uses {x} with setting {rnd.randint(1, 10_000)}"})

    queries = [f"how is the {rnd.choice(TOPICS)} set up with {rnd.choice(TECH)}" for _ in range(n_q)]
    base = {"project_id": pid, "chat_id": cid, "user_id": "bench-user", "agent": "off"}
    # warm-up (a running server has warm pools / prepared statements); not measured
    for qq in queries[:30]:
        await c.post("/v1/memory/prefetch", headers=svc, json={"project_id": pid, "chat_id": cid,
                                                                "user_id": "bench-user", "partial": qq})
        await c.post("/v1/memory/retrieve", headers=svc, json={**base, "query": qq})
        await c.post("/v1/memory/retrieve", headers=svc, json={**base, "chat_id": None, "query": qq + " ?"})
    res: dict[str, list[float]] = {"prefetch": [], "submit_warm": [], "submit_cold": []}
    inner: dict[str, list[float]] = {"prefetch": [], "submit_warm": [], "submit_cold": []}
    marks: dict[str, list[float]] = {}
    hits = 0
    for qq in queries:
        # A debounced typist: prefetch at word boundaries; the last one (pause before Enter) has the full text.
        ws = qq.split()
        for k in range(3, len(ws) + 1, 2) if len(ws) % 2 else list(range(3, len(ws), 2)) + [len(ws)]:
            t0 = time.perf_counter()
            pr = (await c.post("/v1/memory/prefetch", headers=svc, json={
                **{k2: base[k2] for k2 in ("project_id", "chat_id", "user_id")}, "partial": " ".join(ws[:k])})).json()
            res["prefetch"].append((time.perf_counter() - t0) * 1000)
            inner["prefetch"].append(pr["latency_ms"])
            await asyncio.sleep(0.15)  # debounce gap
        t0 = time.perf_counter()
        r = (await c.post("/v1/memory/retrieve", headers=svc, json={**base, "query": qq})).json()
        dt = (time.perf_counter() - t0) * 1000
        hits += r["cache_hit"]
        path = "submit_warm" if r["cache_hit"] else "submit_cold"
        res[path].append(dt)
        inner[path].append(r["latency_ms"]["total"])
        t0 = time.perf_counter()
        r = (await c.post("/v1/memory/retrieve", headers=svc,
                          json={**base, "chat_id": None, "query": qq + " ?"})).json()
        res["submit_cold"].append((time.perf_counter() - t0) * 1000)
        inner["submit_cold"].append(r["latency_ms"]["total"])
        for k, v in r["latency_ms"].items():
            marks.setdefault(k, []).append(v)

    print(f"\n{n_mem} memories, {n_q} queries, prefetch reuse on submit: {hits}/{n_q}")
    print(f"{'path':<14}{'HTTP p50':>10}{'HTTP p95':>10}{'runtime p50':>13}{'runtime p95':>13}{'n':>6}")
    for k, xs in res.items():
        if xs:
            ys = inner[k]
            print(f"{k:<14}{statistics.median(xs):>10.2f}{pct(xs, 0.95):>10}"
                  f"{statistics.median(ys):>13.2f}{pct(ys, 0.95):>13}{len(xs):>6}")
    print("cold breakdown p50:", {k: round(statistics.median(v), 2) for k, v in marks.items()})
    print("cold breakdown p95:", {k: pct(v, 0.95) for k, v in marks.items()})
    await c.aclose()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--memories", type=int, default=1000)
    ap.add_argument("--queries", type=int, default=200)
    a = ap.parse_args()
    asyncio.run(main(a.memories, a.queries))
