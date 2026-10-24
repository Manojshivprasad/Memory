"""Memory Runtime API.

    uvicorn app.main:app --port 8080
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from sqlalchemy import text

from app.api.v1 import chat_completions, chats, memory, prefetch, projects
from app.auth import routes as auth_routes
from app.cache import redis
from app.config import settings
from app.db.session import dispose, engine
from app.embeddings import warmup
from app.gateway import providers


@asynccontextmanager
async def lifespan(_: FastAPI):  # noqa: ANN201
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    await asyncio.to_thread(warmup)  # load the ONNX model before the first request
    await asyncio.to_thread(providers.warm)  # LiteLLM import is slow; do it before traffic
    async with engine.connect() as conn:  # open a pooled connection (vector codec registration)
        await conn.execute(text("SELECT 1"))
    yield
    await redis.aclose()
    await dispose()


app = FastAPI(title="Memory Runtime", version="0.1.0", default_response_class=ORJSONResponse, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Server-Timing", "X-Memory-Chat-Id", "X-Memory-Trace-Id", "X-Memory-Turn-Id",
                    "X-Memory-Count"],
)
for r in (auth_routes.router, projects.router, chats.router, memory.router, prefetch.router,
          chat_completions.router):
    app.include_router(r)


@app.get("/healthz", tags=["ops"])
async def healthz() -> dict:
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
    await redis.ping()
    return {"status": "ok"}
