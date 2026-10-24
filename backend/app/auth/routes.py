import uuid
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import text

from app.auth import jwt as tokens
from app.auth.deps import Principal, ensure_user, require_admin, require_app
from app.auth.secrets import hash_secret, new_api_key, parse_prefix, verify_secret
from app.config import settings
from app.db.session import auth_session

router = APIRouter(prefix="/v1", tags=["auth"])

SUBJECT_SCOPES = {"prefetch", "chat", "memory:read", "memory:write"}


class RegisterIn(BaseModel):
    tenant_name: str = Field(min_length=1, max_length=200)
    email: str = Field(min_length=3, max_length=320, pattern=r"^[^@\s]+@[^@\s]+$")
    password: str = Field(min_length=8, max_length=200)


class LoginIn(BaseModel):
    email: str
    password: str


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class RefreshIn(BaseModel):
    refresh_token: str


class ApiKeyIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)


class ApiKeyExchangeIn(BaseModel):
    api_key: str


class SubjectTokenIn(BaseModel):
    user_id: str = Field(min_length=1, max_length=256, description="Your end-user id")
    project_ids: list[UUID] | None = None
    scopes: list[str] = ["prefetch", "chat", "memory:read"]
    ttl_s: int | None = Field(default=None, ge=60, le=3600)


class AccessToken(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


def _pair(tenant_id: UUID, user_id: UUID) -> TokenPair:
    claims = {"tid": str(tenant_id), "sub": str(user_id), "scopes": ["admin"]}
    return TokenPair(
        access_token=tokens.issue("access", claims, settings.access_ttl_s),
        refresh_token=tokens.issue("refresh", claims, settings.refresh_ttl_s),
        expires_in=settings.access_ttl_s,
    )


@router.post("/auth/register", response_model=TokenPair, status_code=201)
async def register(body: RegisterIn) -> TokenPair:
    tenant_id, user_id = uuid.uuid4(), uuid.uuid4()
    email = body.email.lower()
    async with auth_session() as conn:
        exists = await conn.execute(text("SELECT 1 FROM dashboard_users WHERE email = :e"), {"e": email})
        if exists.first():
            raise HTTPException(status.HTTP_409_CONFLICT, "email already registered")
        await conn.execute(
            text("INSERT INTO tenants (id, name) VALUES (:id, :n)"), {"id": tenant_id, "n": body.tenant_name}
        )
        await conn.execute(
            text("INSERT INTO dashboard_users (id, tenant_id, email, password_hash) VALUES (:id, :t, :e, :h)"),
            {"id": user_id, "t": tenant_id, "e": email, "h": hash_secret(body.password)},
        )
    return _pair(tenant_id, user_id)


@router.post("/auth/login", response_model=TokenPair)
async def login(body: LoginIn) -> TokenPair:
    async with auth_session() as conn:
        row = (
            await conn.execute(
                text("SELECT id, tenant_id, password_hash FROM dashboard_users WHERE email = :e"),
                {"e": body.email.lower()},
            )
        ).mappings().first()
    if not row or not verify_secret(row["password_hash"], body.password):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid credentials")
    return _pair(row["tenant_id"], row["id"])


@router.post("/auth/refresh", response_model=TokenPair)
async def refresh(body: RefreshIn) -> TokenPair:
    try:
        p = tokens.verify(body.refresh_token, {"refresh"})
    except tokens.TokenError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(e)) from e
    return _pair(UUID(p["tid"]), UUID(p["sub"]))


@router.post("/api-keys", status_code=201)
async def create_api_key(body: ApiKeyIn, p: Principal = Depends(require_admin)) -> dict:
    key, prefix = new_api_key()
    key_id = uuid.uuid4()
    async with auth_session() as conn:
        await conn.execute(
            text(
                "INSERT INTO api_keys (id, tenant_id, name, prefix, key_hash, scopes) "
                "VALUES (:id, :t, :n, :p, :h, :s)"
            ),
            {"id": key_id, "t": p.tenant_id, "n": body.name, "p": prefix, "h": hash_secret(key), "s": ["service"]},
        )
    return {"id": str(key_id), "name": body.name, "prefix": prefix, "api_key": key, "note": "shown once"}


@router.get("/api-keys")
async def list_api_keys(p: Principal = Depends(require_admin)) -> list[dict]:
    async with auth_session() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT id, name, prefix, created_at, last_used_at, revoked_at FROM api_keys "
                    "WHERE tenant_id = :t ORDER BY created_at DESC"
                ),
                {"t": p.tenant_id},
            )
        ).mappings().all()
    return [dict(r) for r in rows]


@router.delete("/api-keys/{key_id}", status_code=204)
async def revoke_api_key(key_id: UUID, p: Principal = Depends(require_admin)) -> None:
    async with auth_session() as conn:
        await conn.execute(
            text("UPDATE api_keys SET revoked_at = now() WHERE id = :id AND tenant_id = :t"),
            {"id": key_id, "t": p.tenant_id},
        )


@router.post("/auth/token", response_model=AccessToken)
async def exchange_api_key(body: ApiKeyExchangeIn) -> AccessToken:
    prefix = parse_prefix(body.api_key)
    if not prefix:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "malformed api key")
    async with auth_session() as conn:
        row = (
            await conn.execute(
                text("SELECT id, tenant_id, key_hash, revoked_at FROM api_keys WHERE prefix = :p"), {"p": prefix}
            )
        ).mappings().first()
        if not row or row["revoked_at"] is not None or not verify_secret(row["key_hash"], body.api_key):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid api key")
        await conn.execute(text("UPDATE api_keys SET last_used_at = now() WHERE id = :id"), {"id": row["id"]})
    token = tokens.issue(
        "service",
        {"tid": str(row["tenant_id"]), "sub": f"key:{row['id']}", "scopes": ["service"]},
        settings.service_ttl_s,
    )
    return AccessToken(access_token=token, expires_in=settings.service_ttl_s)


@router.post("/auth/subject-token", response_model=AccessToken)
async def subject_token(body: SubjectTokenIn, p: Principal = Depends(require_app)) -> AccessToken:
    if p.kind != "service":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "service token required")
    bad = set(body.scopes) - SUBJECT_SCOPES
    if bad:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"unknown scopes: {sorted(bad)}")
    uid = await ensure_user(p.tenant_id, body.user_id)
    ttl = body.ttl_s or settings.subject_ttl_s
    claims = {"tid": str(p.tenant_id), "sub": body.user_id, "uid": str(uid), "scopes": body.scopes}
    if body.project_ids:
        claims["pids"] = [str(x) for x in body.project_ids]
    return AccessToken(access_token=tokens.issue("subject", claims, ttl), expires_in=ttl)


@router.get("/auth/me")
async def me(p: Principal = Depends(require_admin)) -> dict:
    async with auth_session() as conn:
        row = (await conn.execute(
            text("SELECT u.email, u.role, t.name AS tenant_name FROM dashboard_users u "
                 "JOIN tenants t ON t.id = u.tenant_id WHERE u.id = :id"), {"id": UUID(p.subject)},
        )).mappings().first()
    if not row:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "user not found")
    return {"id": p.subject, "tenant_id": str(p.tenant_id), **dict(row)}


@router.post("/auth/session", response_model=AccessToken)
async def dashboard_session(p: Principal = Depends(require_admin)) -> AccessToken:
    """First-party UI: a dashboard user acts as their OWN end user (memory owner ``dash:<user id>``).

    Returns a subject token with every end-user scope, so the web app can chat, prefetch and manage memory
    without ever holding an API key.
    """
    external = f"dash:{p.subject}"
    uid = await ensure_user(p.tenant_id, external)
    ttl = 3600
    token = tokens.issue("subject", {"tid": str(p.tenant_id), "sub": external, "uid": str(uid),
                                     "scopes": sorted(SUBJECT_SCOPES)}, ttl)
    return AccessToken(access_token=token, expires_in=ttl)
