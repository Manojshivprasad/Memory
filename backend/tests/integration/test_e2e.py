"""End-to-end: real Postgres + Redis (docker compose up), API in-process, worker drained inline.

    MEMORY_IT=1 pytest tests/integration -q
"""

import os
import uuid

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import text

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio(loop_scope="module"),
    pytest.mark.skipif(os.environ.get("MEMORY_IT") != "1", reason="set MEMORY_IT=1 (needs docker compose up)"),
]


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def env():
    from app.embeddings import warmup
    from app.main import app
    from app.workers.worker import TurnConsumer, ensure_group

    warmup()
    await ensure_group()
    consumer = TurnConsumer(f"it-{uuid.uuid4().hex[:6]}")
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")

    async def drain() -> None:
        while await consumer.read_once(block_ms=200):
            pass

    async def tenant() -> tuple[dict, str]:
        email = f"it-{uuid.uuid4().hex[:8]}@example.com"
        r = await client.post("/v1/auth/register", json={"tenant_name": "IT", "email": email, "password": "pw-12345678"})
        admin = {"Authorization": f"Bearer {r.json()['access_token']}"}
        key = (await client.post("/v1/api-keys", json={"name": "it"}, headers=admin)).json()["api_key"]
        tok = (await client.post("/v1/auth/token", json={"api_key": key})).json()["access_token"]
        return {"Authorization": f"Bearer {tok}"}, key

    yield {"client": client, "drain": drain, "tenant": tenant}
    await client.aclose()


async def _setup(env, user="u1"):
    c = env["client"]
    svc, _ = await env["tenant"]()
    q = {"user_id": user}
    p = (await c.post("/v1/projects", json={"name": "Acme API"}, params=q, headers=svc)).json()
    chat = (await c.post(f"/v1/projects/{p['id']}/chats", json={}, params=q, headers=svc)).json()
    return svc, q, p, chat


async def _commit(env, svc, pid, cid, u, a, **kw):
    r = await env["client"].post("/v1/memory/commit", headers=svc, json={
        "project_id": pid, "chat_id": cid, "user_id": "u1", "user_msg": u, "assistant_msg": a, **kw})
    assert r.status_code == 202, r.text
    await env["drain"]()
    return r.json()


async def _retrieve(env, svc, pid, cid, query, **kw):
    r = await env["client"].post("/v1/memory/retrieve", headers=svc, json={
        "project_id": pid, "chat_id": cid, "user_id": "u1", "query": query, "agent": "off", **kw})
    assert r.status_code == 200, r.text
    return r.json()


async def test_cross_chat_recall_contradiction_and_isolation(env):
    c = env["client"]
    svc, q, p, chat1 = await _setup(env)
    pid = p["id"]

    await _commit(env, svc, pid, chat1["id"],
                  "We're building the API in FastAPI with Postgres; deadline is Oct 15.",
                  "Great, FastAPI with Postgres is a solid combination.")
    await _commit(env, svc, pid, chat1["id"], "I prefer concise answers with code examples.", "Noted.")
    closed = (await c.post(f"/v1/chats/{chat1['id']}/close", params={**q, "wait": "true"}, headers=svc)).json()
    assert closed["episode"]["episode_memory_id"]

    # New chat, same project: memory from chat 1 is recalled, brief is included (new chat).
    chat2 = (await c.post(f"/v1/projects/{pid}/chats", json={}, params=q, headers=svc)).json()
    res = await _retrieve(env, svc, pid, chat2["id"], "which database are we using?")
    assert "postgres" in res["context"].lower(), res
    assert res["brief"], "project brief should be injected at the start of a new chat"

    # Another project of the same user never sees it.
    p2 = (await c.post("/v1/projects", json={"name": "Other"}, params=q, headers=svc)).json()
    other = await _retrieve(env, svc, p2["id"], None, "which database are we using?")
    assert "postgres" not in (other["context"] + other["brief"]).lower()

    # Contradiction: the old fact is closed (kept for history), the new one is active.
    await _commit(env, svc, pid, chat2["id"], "Actually we switched to MySQL.", "Got it, MySQL it is.")
    res = await _retrieve(env, svc, pid, chat2["id"], "which database are we using?")
    # Current facts: MySQL only. (A past-chat episode may still say what was true back then.)
    facts = "\n".join(ln for ln in res["context"].lower().splitlines() if not ln.startswith("- [past chat"))
    assert "mysql" in facts and "postgres" not in facts, res
    mems = (await c.get("/v1/memory", params={"project_id": pid, **q, "include_invalid": "true"},
                        headers=svc)).json()
    old = next(m for m in mems if m["pred"] == "uses_database" and (m["obj"] or "").lower() == "postgres")
    assert old["valid_until"] is not None and old["superseded_by"]
    explain = (await c.get(f"/v1/memory/{old['id']}/explain", params=q, headers=svc)).json()
    assert [v["op"] for v in explain["versions"]] == ["ADD", "INVALIDATE"]
    assert explain["provenance"][0]["chat_id"] == chat1["id"]

    # Historical question still sees the superseded fact.
    hist = await _retrieve(env, svc, pid, chat2["id"], "which database did we use before we switched?")
    assert "no longer true" in hist["context"]


async def test_beliefs_adoption_gate_budget_idempotency(env):
    c = env["client"]
    svc, q, p, chat = await _setup(env)
    pid, cid = p["id"], chat["id"]

    await _commit(env, svc, pid, cid, "How should I cache user sessions?",
                  "I recommend using Redis with a 30 minute TTL for sessions.")
    mems = (await c.get("/v1/memory", params={"project_id": pid, **q}, headers=svc)).json()
    belief = next(m for m in mems if "redis" in m["content"].lower())
    assert belief["is_belief"] is True, "assistant suggestions are unconfirmed beliefs"

    await _commit(env, svc, pid, cid, "Sounds good, let's do that.", "Great.")
    mems = (await c.get("/v1/memory", params={"project_id": pid, **q}, headers=svc)).json()
    assert next(m for m in mems if m["id"] == belief["id"])["is_belief"] is False, "adoption confirms it"

    hi = await _retrieve(env, svc, pid, cid, "hi")
    assert hi["gate"] == "phatic" and hi["context"] == "" and hi["brief"] == ""

    small = await _retrieve(env, svc, pid, cid, "redis session cache ttl", budget_tokens=20, brief="never")
    assert small["tokens"] <= 20

    tid = str(uuid.uuid4())
    for _ in range(2):
        await _commit(env, svc, pid, cid, "Remember that staging is at staging.acme.dev", "Noted.", turn_id=tid)
    turns = (await c.get(f"/v1/chats/{cid}/turns", params=q, headers=svc)).json()
    assert sum(1 for t in turns if t["id"] == tid) == 1
    mems = (await c.get("/v1/memory", params={"project_id": pid, **q}, headers=svc)).json()
    assert sum(1 for m in mems if "staging.acme.dev" in m["content"]) == 1


async def test_prefetch_warm_path(env):
    svc, q, p, chat = await _setup(env)
    pid, cid = p["id"], chat["id"]
    await _commit(env, svc, pid, cid, "We use Postgres as the database.", "OK.")
    r = await env["client"].post("/v1/memory/prefetch", headers=svc, json={
        "project_id": pid, "chat_id": cid, "user_id": "u1", "partial": "which database are we using"})
    assert r.status_code == 200 and r.json()["candidates"] >= 1
    res = await _retrieve(env, svc, pid, cid, "which database are we using?")
    assert res["cache_hit"] is True, res["latency_ms"]
    assert "postgres" in res["context"].lower()


async def test_proxy_streaming_writes_back(env):
    c = env["client"]
    svc, q, p, chat = await _setup(env)
    body = {"model": "openai/gpt-4o-mini", "stream": True, "user": "u1",
            "mock_response": "Use FastAPI dependency injection for auth.",
            "messages": [{"role": "user", "content": "We are using FastAPI. How should I do auth?"}],
            "memory": {"project_id": p["id"], "chat_id": chat["id"]}}
    async with c.stream("POST", "/v1/chat/completions", json=body, headers=svc) as r:
        assert r.status_code == 200
        assert r.headers["x-memory-chat-id"] == chat["id"]
        raw = "".join([chunk async for chunk in r.aiter_text()])
    assert raw.rstrip().endswith("data: [DONE]")
    await env["drain"]()
    turns = (await c.get(f"/v1/chats/{chat['id']}/turns", params=q, headers=svc)).json()
    assert len(turns) == 1 and "dependency injection" in turns[0]["assistant_msg"]


@pytest.mark.skipif(os.environ.get("MEMORY_EMBEDDER") != "fastembed", reason="needs real embeddings")
async def test_semantic_recall_and_strictness(env):
    svc, q, p, chat = await _setup(env)
    pid, cid = p["id"], chat["id"]
    await _commit(env, svc, pid, cid, "We're building the API in FastAPI with Postgres; deadline is Oct 15.", "Great.")
    await _commit(env, svc, pid, cid, "I prefer concise answers with code examples.", "Noted.")
    chat2 = (await env["client"].post(f"/v1/projects/{pid}/chats", json={}, params=q, headers=svc)).json()

    # No shared words with the stored memory: only meaning can find it.
    res = await _retrieve(env, svc, pid, chat2["id"], "what DB did we pick?", brief="never")
    assert "postgres" in res["context"].lower(), res
    assert any(m["evidence"] == "semantic" for m in res["memories"])

    # Unrelated questions get no project facts (strict = cheap). Only standing response-style preferences
    # ("concise answers with code examples") may apply, as they govern every answer.
    for unrelated in ["write a regex to validate emails", "explain how quicksort works",
                      "what is the capital of peru"]:
        res = await _retrieve(env, svc, pid, chat2["id"], unrelated, brief="never")
        others = [m for m in res["memories"] if not (m["kind"] == "preference" and m["evidence"] == "constraint")]
        assert others == [], (unrelated, res["memories"])


@pytest.mark.skipif(not os.environ.get("MEMORY_LLM_MODEL"), reason="needs MEMORY_LLM_MODEL + provider key")
async def test_llm_extraction_agent_and_consolidation(env):
    c = env["client"]
    svc, q, p, chat = await _setup(env)
    pid, cid = p["id"], chat["id"]
    await _commit(env, svc, pid, cid, "We're building the shop backend in Django with Postgres. Launch is Nov 20.",
                  "Great stack for an e-commerce backend.")
    await _commit(env, svc, pid, cid, "Alice owns the payments service and she wants Stripe webhooks to be retried.",
                  "Noted, I'll keep that in mind for the payments work.")
    mems = (await c.get("/v1/memory", params={"project_id": pid, **q}, headers=svc)).json()
    assert any("django" in m["content"].lower() for m in mems), mems
    explain = (await c.get(f"/v1/memory/{mems[0]['id']}/explain", params=q, headers=svc)).json()
    assert explain["provenance"][0]["method"] == "llm", explain["provenance"]

    # Memory agent: forced, and auto-triggered by a temporal cue. Must stay within its timeout budget.
    forced = await _retrieve(env, svc, pid, cid, "who should I talk to about payment retries?", agent="always")
    assert forced["agent_used"] is True, forced
    assert "alice" in forced["context"].lower(), forced
    auto = await _retrieve(env, svc, pid, cid, "what did we decide today about the stack?", agent="auto")
    assert auto["agent_reason"] and auto["agent_reason"].startswith("temporal"), auto
    # The agent only adds to automatic recall: the stack facts must be there whether or not it finished in time.
    assert "django" in auto["context"].lower(), auto
    assert auto["latency_ms"]["total"] < 1500, auto["latency_ms"]  # hard budget holds on the submit path
    print("\nagent latency ms:", forced["latency_ms"].get("agent"), auto["latency_ms"].get("agent"),
          "used:", forced["agent_used"], auto["agent_used"])

    # LLM consolidation: episode + brief.
    closed = (await c.post(f"/v1/chats/{cid}/close", params={**q, "wait": "true"}, headers=svc)).json()
    assert closed["episode"]["title"]
    brief = (await c.get(f"/v1/projects/{pid}/brief", params=q, headers=svc)).json()["brief"]
    assert "django" in brief.lower(), brief
    print("episode:", closed["episode"]["title"], "| brief:", brief.replace("\n", " | "))


async def test_tenant_isolation(env):
    from app.db.session import tenant_session

    c = env["client"]
    svc_a, q, p, _ = await _setup(env)
    await _commit(env, svc_a, p["id"], _["id"], "Remember that the launch code is 42.", "Noted.")
    svc_b, _k = await env["tenant"]()
    r = await c.get(f"/v1/projects/{p['id']}", params=q, headers=svc_b)
    assert r.status_code == 404
    # RLS at the database level: tenant B's session sees none of tenant A's rows.
    from app.auth.deps import principal_from_token
    tb = principal_from_token(svc_b["Authorization"].split()[1]).tenant_id
    async with tenant_session(tb) as conn:
        n = (await conn.execute(text("SELECT count(*) FROM memories WHERE project_id = :p"),
                                {"p": uuid.UUID(p["id"])})).scalar_one()
    assert n == 0
