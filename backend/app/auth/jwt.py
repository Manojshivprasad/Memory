"""JWT issue/verify.

Token types (``typ`` claim):
* ``access``  - dashboard user (tenant admin), short-lived
* ``refresh`` - dashboard refresh token
* ``service`` - server-side app credential, exchanged from an API key; may act for any user of the tenant
* ``subject`` - browser-safe token bound to ONE end user (optionally to specific projects), used for
  type-ahead prefetch and direct chat from a frontend without exposing the API key
"""

import time
import uuid
from typing import Any

import jwt

from app.config import settings

ALGORITHM = "HS256"


class TokenError(Exception):
    pass


def issue(typ: str, claims: dict[str, Any], ttl_s: int) -> str:
    now = int(time.time())
    payload = {
        "iss": settings.jwt_issuer,
        "iat": now,
        "exp": now + ttl_s,
        "jti": uuid.uuid4().hex,
        "typ": typ,
        **claims,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=ALGORITHM, headers={"kid": settings.jwt_kid})


def verify(token: str, allowed_types: set[str]) -> dict[str, Any]:
    try:
        header = jwt.get_unverified_header(token)
        if header.get("kid") != settings.jwt_kid:
            raise TokenError("unknown key id")
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[ALGORITHM],
            issuer=settings.jwt_issuer,
            options={"require": ["exp", "iat", "typ", "tid"]},
        )
    except jwt.PyJWTError as e:
        raise TokenError(str(e)) from e
    if payload.get("typ") not in allowed_types:
        raise TokenError("wrong token type")
    return payload
