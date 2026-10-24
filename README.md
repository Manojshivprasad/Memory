# Memory Runtime

A real-time memory layer that sits between your users and your LLM. It recalls the right
project memory **while the user types**, injects only what is relevant within a token budget, and
**learns from every turn in the background**. Nothing on the response path waits for memory writes.

```
Tenant → User → Project → Chat → Turn
                  │
                  ├─ project memory (facts, decisions, goals, preferences, open threads, episodes)
                  ├─ project brief (~150-token living summary; stable, cacheable prompt prefix)
                  └─ chat working memory (Redis)
User-level memory (identity, cross-project preferences) is shared across a user's projects unless `isolated`.
```

## Quick start

```bash
docker compose up -d                          # Postgres 16 + pgvector (5433), Redis 7 (6380)
cd backend
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"   # (bin/ on Linux/macOS)
cp .env.example .env                          # set MEMORY_JWT_SECRET, optional MEMORY_LLM_MODEL + provider key
alembic upgrade head
uvicorn app.main:app --port 8080              # API
python -m app.workers.worker                  # memory worker (Stage B/C/D)
```

Tests: `pytest` (unit). With the containers up, `MEMORY_IT=1 pytest tests/integration` runs the end-to-end tests,
and `MEMORY_EMBEDDER=fastembed` adds the semantic-recall test. `python scripts/smoke.py` checks a running API + worker.
Benchmark: `python -m bench.latency --memories 1000`.

Measured on a laptop (20 cores, Windows) with bge-small embeddings and 1,000 memories in the project, over HTTP.
"Runtime" is the memory work inside the request; "HTTP" includes the request itself.

| Path | HTTP p50 / p95 | Runtime p50 / p95 |
|---|---|---|
| Submit after type-ahead prefetch | 15 / 23 ms | 12 / 19 ms |
| Submit, cold (no prefetch) | 26 / 52 ms | 23 / 47 ms |
| Prefetch tick (while typing) | 26 / 56 ms | 24 / 53 ms |

## Memory evaluation harness (backend/eval/)

```bash
# stop the worker first (it would consume the harness's turns)
python -m eval.harness --model openrouter/openai/gpt-4o-mini --tag nightly   # ~2 cents
python -m eval.extract_bench openrouter/openai/gpt-4o-mini openrouter/openai/gpt-4.1-nano   # model comparison
```

There are 12 multi-session scenarios. Domains:
- personal life
- education (both a self-learner and a parent tutoring a child)
- work/software
- health
- travel
- finance
- cooking/allergies
- creative writing
- career
- pure chit-chat
- a global preference

Assistant replies are scripted, so the only spend is memory extraction. For every turn the harness runs the real path: recall → commit → worker. It then closes each chat and asks probe questions in a **new** chat. It scores:

| Metric | Meaning |
|---|---|
| capture_recall | expected facts exist as memories (past-chat summaries don't count) |
| stale_facts_retired | changed facts (moved, renamed, increased…) are no longer active |
| noise_rejected | chit-chat and general knowledge were not stored |
| probe_pass / recall / precision | the right memory is injected, and stale or unrelated memory is not |
| cross_project_pass | project facts don't leak into another project; user-level preferences do follow the user |

Reports go to `eval/reports/<tag>-<time>.{md,json}`, with every failure and the memories involved.

## Web app (frontend/)

Next.js 16 (App Router), TypeScript, Tailwind v4 and lucide icons. No UI kit and no chart library.

```bash
cd frontend
npm install
echo NEXT_PUBLIC_API_URL=http://localhost:8080 > .env.local
npm run dev -- --port 3100      # add the origin to MEMORY_CORS_ORIGINS in backend/.env
```

| Screen | What you can do there |
|---|---|
| **Home** | Animated squid, greeting and composer. Chips pick the project, memory agent (off/auto/always) and token budget, plus a model picker. While you type, memory is prefetched over WebSocket and the result is shown (e.g. "3 memories ready · 4 ms"). |
| **Chat** | Streaming answers with a "N memories used" chip on each. The **Memory debugger** panel shows, per answer: the recall gate, whether the prefetch was reused or the agent helped, and a latency breakdown. It lists each candidate with its score, evidence and "why this score", plus what was not injected and why, and what memory formed from the turn (ADD / REINFORCE / INVALIDATE …). Chats can be renamed, closed and saved to project memory, deleted, and failed answers retried. |
| **Memory** | Project brief (rebuild on demand), "Consolidate now" (nightly sleep on demand), and tabs for Current / Unconfirmed / Superseded / Faded. Filter by kind, search, **Remember something** (with scope and importance), and open any memory for provenance, versions, strength and recent retrievals. Memories can also be forgotten (hard delete). |
| **Projects** | Create, edit (including the *isolated* toggle) and delete projects. Each project page shows its brief, chats, and **knowledge graph**. |
| **Analytics** | Memory health tiles, new memories per day, composition by kind, lifecycle, and retrieval latency / prefetch hit rate / agent rate. |
| **Settings** | Default model, agent mode, budget, brief mode and debugger toggle. API keys can be created (shown once), listed and revoked, with a backend quick-start snippet. |

The UI signs in as a dashboard user, then exchanges that login for a short-lived *subject* token (`POST /v1/auth/session`). The browser never holds an API key.

## Using it

**Auth.** Register a tenant (`POST /v1/auth/register`) and create an API key (`POST /v1/api-keys`).
Your backend exchanges the key for a service JWT (`POST /v1/auth/token`). For browsers, mint a short-lived
**subject token** bound to one end user and, optionally, specific projects (`POST /v1/auth/subject-token`),
so the API key never reaches the frontend.

**Projects and chats.** `POST /v1/projects`, then `POST /v1/projects/{id}/chats`. Every chat in a project shares
that project's memory. `POST /v1/chats/{id}/close` consolidates the chat into an episode.

**Proxy mode** is OpenAI-compatible and works with any provider through LiteLLM:
```json
POST /v1/chat/completions
{"model": "anthropic/claude-sonnet-5", "stream": true, "user": "end-user-42",
 "messages": [{"role": "user", "content": "what DB did we pick?"}],
 "memory": {"project_id": "…", "chat_id": "…", "budget_tokens": 300, "agent": "auto"}}
```

**Sidecar mode** is for apps that call the LLM themselves:
`POST /v1/memory/retrieve` → call your LLM with `brief` + `context` → `POST /v1/memory/commit`.

**Type-ahead.** Open `ws://host/v1/prefetch?token=…&project_id=…&chat_id=…` and send `{"partial": "..."}`
on a ~150 ms debounce. On submit, the prefetched candidate pool is rescored instead of queried again.
When the user pauses, the memory agent (if needed) runs speculatively.

**Inspect.** `GET /v1/memory?project_id=`, `GET /v1/memory/{id}/explain` (provenance, versions, strength,
recent retrievals), `DELETE /v1/memory/{id}` (hard delete). Every retrieval returns a `trace_id` and `Server-Timing`.

## How it works

**Read path (hot, strict).** It runs in this order:
1. A gate skips greetings and acknowledgements.
2. Working memory, brief, memory version, prefetch pool and the local query embedding are fetched concurrently.
3. Known entities in the prompt are found, and spreading activation runs over the project graph (in-process cache).
4. A single SQL round trip unions the recall channels: vector (exact cosine over the project), full-text,
   associated entities, time window, recent episodes, and open threads.
5. Candidates are scored:
   `relevance (noisy-OR of channels) × strength (retrievability, importance, confidence, scope)`.
6. Selection is strict:
   - each memory needs a real relevance signal and must clear `min_score`;
   - MMR removes near-duplicates;
   - a hard token budget caps the result. Injecting nothing is normal.
7. When recall looks insufficient (temporal, multi-hop, unresolved reference, weak or ambiguous
   results), an optional **memory agent** runs with a hard timeout and falls back to the deterministic result.

**Write path (async).**
- **Stage A** runs in the request and takes about 1 ms, Redis only. It updates working memory and enqueues the turn.
- **Stage B** runs in the worker:
  1. Stores the verbatim turn.
  2. Detects signals and scores salience (novelty against what was recalled).
  3. Extracts memory ops: an LLM when `MEMORY_LLM_MODEL` is set, deterministic heuristics otherwise.
  4. Validates the ops. Assistant claims become *beliefs* until the user adopts them.
  5. Resolves them: dedupe, contradiction (`valid_until` and `superseded_by`), reconsolidation of recalled memories,
     outcome learning, and spaced reinforcement of the memories that were actually used.
  6. Writes provenance and a version for every change.
- **Stage C** runs when a chat goes idle, is closed, or reaches N turns. It turns the chat into an episode gist and refreshes the brief.
- **Stage D** is the nightly "sleep":
  - forgetting curve: active → weakening → archived;
  - merges duplicates;
  - promotes repeatedly observed beliefs;
  - fades stale open threads;
  - promotes cross-project preferences to user level.

| Human memory | Here |
|---|---|
| Working memory | Redis per chat, capped injection budget |
| Selective encoding | Salience gate (signals × novelty) |
| Gist vs verbatim | Facts/episodes are retrieved; verbatim turns are kept and fetched on demand |
| Forgetting curve / spacing | Stability S, retrievability `R = e^(-t/S)`; useful recall at low R strengthens most |
| Association / priming | Entity graph plus spreading activation seeded by the prompt and chat focus |
| Reconsolidation | Recalled memories are passed to the extractor and updated with versions |
| Interference | Contradictions close the old fact in time instead of deleting it |
| Prospective memory | Open commitments and questions stay retrievable until closed |
| Source memory | Provenance on every op (AI advantage) |

## Layout

`backend/app`:
- `auth/`: JWT, API keys, principals
- `api/v1/`: HTTP and WebSocket endpoints
- `runtime/`: read path (orchestrator, retrieval, agent, graph, working memory)
- `write/`: Stages A and B
- `consolidation/`: Stages C and D, and the brief
- `memory/`: strength and salience models
- `workers/`: stream consumer and scheduler

`frontend/src`:
- `app/(app)/`: pages (home, chat, memory, projects, analytics, settings)
- `components/`: squid, composer, debugger, explain drawer, graph, charts, UI primitives
- `lib/`: API client with SSE streaming and prefetch socket, auth, workspace and chat stores
