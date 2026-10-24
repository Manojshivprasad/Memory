"""Smoke test against a running API (:8080) + worker (python -m app.workers.worker).

    python scripts/smoke.py

Covers: API key -> service token -> browser subject token, WebSocket type-ahead prefetch,
proxy streaming (LiteLLM mock_response, no provider key needed), async write-back by the worker
process, idle-chat consolidation into an episode + project brief, recall in a NEW chat.
"""

import asyncio
import time
import uuid

import httpx
import orjson
import websockets

BASE = "http://127.0.0.1:8080"


async def wait_for(fn, timeout: float = 30, every: float = 0.5):  # noqa: ANN001, ANN201
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        v = await fn()
        if v:
            return v
        await asyncio.sleep(every)
    raise TimeoutError


async def main() -> None:
    async with httpx.AsyncClient(base_url=BASE, timeout=30) as c:
        await wait_for(lambda: _ok(c))
        r = await c.post("/v1/auth/register", json={"tenant_name": "smoke", "password": "pw-12345678",
                                                    "email": f"smoke-{uuid.uuid4().hex[:6]}@example.com"})
        admin = {"Authorization": f"Bearer {r.json()['access_token']}"}
        key = (await c.post("/v1/api-keys", json={"name": "smoke"}, headers=admin)).json()["api_key"]
        svc = {"Authorization": f"Bearer {(await c.post('/v1/auth/token', json={'api_key': key})).json()['access_token']}"}
        q = {"user_id": "alice"}
        pid = (await c.post("/v1/projects", json={"name": "Shop"}, params=q, headers=svc)).json()["id"]
        chat1 = (await c.post(f"/v1/projects/{pid}/chats", json={}, params=q, headers=svc)).json()["id"]
        sub = (await c.post("/v1/auth/subject-token", headers=svc,
                            json={"user_id": "alice", "project_ids": [pid]})).json()["access_token"]
        print("auth + project + chat + subject token: ok")

        # 1) proxy streaming with the browser-safe subject token
        body = {"model": "openai/gpt-4o-mini", "stream": True,
                "mock_response": "Good choice. I recommend Stripe Checkout for payments.",
                "messages": [{"role": "user", "content": "We're building the shop backend with Django and Postgres."}],
                "memory": {"project_id": pid, "chat_id": chat1}}
        t = time.perf_counter()
        async with c.stream("POST", "/v1/chat/completions", json=body,
                            headers={"Authorization": f"Bearer {sub}"}) as r:
            assert r.status_code == 200, await r.aread()
            timing = r.headers.get("server-timing")
            text = "".join([x async for x in r.aiter_text()])
        assert text.rstrip().endswith("[DONE]")
        print(f"proxy stream ok in {(time.perf_counter() - t) * 1000:.0f} ms; Server-Timing: {timing}")

        # 2) worker process forms memories asynchronously
        async def mems():  # noqa: ANN202
            m = (await c.get("/v1/memory", params={"project_id": pid, **q}, headers=svc)).json()
            return m if any("postgres" in x["content"].lower() for x in m) else None
        m = await wait_for(mems)
        print("worker wrote memories:", sorted(x["content"] for x in m))

        # 3) chat goes idle -> worker consolidates it into an episode + brief (MEMORY_CHAT_IDLE_S=5)
        async def brief():  # noqa: ANN202
            b = (await c.get(f"/v1/projects/{pid}/brief", params=q, headers=svc)).json()["brief"]
            return b or None
        print("brief:", (await wait_for(brief, timeout=40)).replace("\n", " | "))

        # 4) new chat: type-ahead over WebSocket with the subject token, then submit
        chat2 = (await c.post(f"/v1/projects/{pid}/chats", json={}, params=q, headers=svc)).json()["id"]
        url = f"ws://127.0.0.1:8080/v1/prefetch?token={sub}&project_id={pid}&chat_id={chat2}&agent=off"
        async with websockets.connect(url) as ws:
            for partial in ["which data", "which database are", "which database are we using?"]:
                await ws.send(orjson.dumps({"partial": partial}).decode())
                msg = orjson.loads(await ws.recv())
                print(f"  prefetch {partial!r}: {msg['candidates']} candidates in {msg['latency_ms']} ms")
        res = (await c.post("/v1/memory/retrieve", headers={"Authorization": f"Bearer {sub}"},
                            json={"project_id": pid, "chat_id": chat2, "query": "which database are we using?",
                                  "agent": "off"})).json()
        print(f"retrieve in new chat: cache_hit={res['cache_hit']} latency={res['latency_ms']}")
        print(res["context"])
        assert res["cache_hit"] and "postgres" in res["context"].lower()
        print("\nSMOKE OK")


async def _ok(c: httpx.AsyncClient) -> bool:
    try:
        return (await c.get("/healthz")).status_code == 200
    except httpx.HTTPError:
        return False


if __name__ == "__main__":
    asyncio.run(main())
