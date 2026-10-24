"""Request principal + tenant/user/project resolution (cached; used on the hot path)."""

import time
import uuid
from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from fastapi import Depends, Header, HTTPException, Query, status
from sqlalchemy import text

from app.auth.jwt import TokenError, verify
from app.db.session import tenant_session

Kind = Literal["admin", "service", "subject"]
_TYPE_TO_KIND: dict[str, Kind] = {"access": "admin", "service": "service", "subject": "subject"}


@dataclass(frozen=True)
class Principal:
    tenant_id: UUID
    kind: Kind
    subject: str
    scopes: frozenset[str]
    user_id: UUID | None = None                 # set for subject tokens
    project_ids: frozenset[UUID] | None = None  # optional restriction on subject tokens

    def require_scope(self, scope: str) -> None:
        if self.kind == "service" or scope in self.scopes:
            return
        raise HTTPException(status.HTTP_403_FORBIDDEN, f"missing scope: {scope}")

    def check_project(self, project_id: UUID) -> None:
        if self.project_ids is not None and project_id not in self.project_ids:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "project not allowed for this token")


def principal_from_token(token: str) -> Principal:
    try:
        p = verify(token, {"access", "service", "subject"})
    except TokenError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"invalid token: {e}") from e
    pids = p.get("pids")
    return Principal(
        tenant_id=UUID(p["tid"]),
        kind=_TYPE_TO_KIND[p["typ"]],
        subject=str(p.get("sub", "")),
        scopes=frozenset(p.get("scopes", [])),
        user_id=UUID(p["uid"]) if p.get("uid") else None,
        project_ids=frozenset(UUID(x) for x in pids) if pids else None,
    )


async def get_principal(authorization: str = Header(default="")) -> Principal:
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing bearer token")
    return principal_from_token(token)


async def require_admin(p: Principal = Depends(get_principal)) -> Principal:
    if p.kind != "admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "dashboard token required")
    return p


async def require_app(p: Principal = Depends(get_principal)) -> Principal:
    if p.kind not in ("service", "subject"):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "service or subject token required")
    return p


# ---------------------------------------------------------------- user resolution (cached)
_user_cache: dict[tuple[UUID, str], UUID] = {}


async def ensure_user(tenant_id: UUID, external_id: str) -> UUID:
    key = (tenant_id, external_id)
    if key in _user_cache:
        return _user_cache[key]
    async with tenant_session(tenant_id) as conn:
        row = await conn.execute(
            text(
                "INSERT INTO users (id, tenant_id, external_id) VALUES (:id, :t, :e) "
                "ON CONFLICT (tenant_id, external_id) DO UPDATE SET external_id = EXCLUDED.external_id "
                "RETURNING id"
            ),
            {"id": uuid.uuid4(), "t": tenant_id, "e": external_id},
        )
        uid = row.scalar_one()
    if len(_user_cache) > 100_000:
        _user_cache.clear()
    _user_cache[key] = uid
    return uid


async def resolve_user(p: Principal, external_user_id: str | None) -> UUID:
    """Subject tokens carry their user; service tokens must name the end user explicitly."""
    if p.kind == "subject":
        if external_user_id and external_user_id != p.subject:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "subject token cannot act for another user")
        assert p.user_id is not None
        return p.user_id
    if not external_user_id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "user_id is required with a service token")
    return await ensure_user(p.tenant_id, external_user_id)


async def user_param(
    user_id: str | None = Query(default=None, description="External end-user id (service tokens)"),
    p: Principal = Depends(require_app),
) -> UUID:
    return await resolve_user(p, user_id)


# ---------------------------------------------------------------- project / chat meta (cached)
@dataclass(frozen=True)
class ProjectMeta:
    id: UUID
    tenant_id: UUID
    user_id: UUID
    isolated: bool


@dataclass(frozen=True)
class ChatMeta:
    id: UUID
    project_id: UUID


_META_TTL = 60.0
_project_cache: dict[UUID, tuple[float, ProjectMeta]] = {}
_chat_cache: dict[UUID, tuple[float, ChatMeta]] = {}


def forget_project(project_id: UUID) -> None:
    _project_cache.pop(project_id, None)


async def project_for(p: Principal, user_id: UUID, project_id: UUID) -> ProjectMeta:
    """Ownership check: the project must belong to this tenant AND this end user."""
    p.check_project(project_id)
    now = time.monotonic()
    hit = _project_cache.get(project_id)
    if hit and now - hit[0] < _META_TTL:
        meta = hit[1]
    else:
        async with tenant_session(p.tenant_id) as conn:
            row = (
                await conn.execute(
                    text("SELECT id, tenant_id, user_id, isolated FROM projects WHERE id = :id"),
                    {"id": project_id},
                )
            ).mappings().first()
        if not row:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "project not found")
        meta = ProjectMeta(row["id"], row["tenant_id"], row["user_id"], row["isolated"])
        _project_cache[project_id] = (now, meta)
    if meta.tenant_id != p.tenant_id or meta.user_id != user_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "project not found")
    return meta


async def chat_for(p: Principal, project: ProjectMeta, chat_id: UUID) -> ChatMeta:
    now = time.monotonic()
    hit = _chat_cache.get(chat_id)
    if hit and now - hit[0] < _META_TTL:
        meta = hit[1]
    else:
        async with tenant_session(p.tenant_id) as conn:
            row = (
                await conn.execute(text("SELECT id, project_id FROM chats WHERE id = :id"), {"id": chat_id})
            ).mappings().first()
        if not row:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "chat not found")
        meta = ChatMeta(row["id"], row["project_id"])
        _chat_cache[chat_id] = (now, meta)
    if meta.project_id != project.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "chat not found in this project")
    return meta


def remember_chat(chat_id: UUID, project_id: UUID) -> None:
    _chat_cache[chat_id] = (time.monotonic(), ChatMeta(chat_id, project_id))
