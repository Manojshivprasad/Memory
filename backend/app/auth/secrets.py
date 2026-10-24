"""Password + API key hashing (argon2). Never on the hot path: only login and key exchange."""

import secrets

from argon2 import PasswordHasher
from argon2.exceptions import VerificationError

_ph = PasswordHasher()

API_KEY_PREFIX = "mk"


def hash_secret(value: str) -> str:
    return _ph.hash(value)


def verify_secret(hashed: str, value: str) -> bool:
    try:
        return _ph.verify(hashed, value)
    except VerificationError:
        return False


def new_api_key() -> tuple[str, str]:
    """Returns (full_key, prefix). Format: ``mk_<prefix>_<secret>``; only the hash is stored."""
    prefix = secrets.token_hex(6)
    return f"{API_KEY_PREFIX}_{prefix}_{secrets.token_urlsafe(32)}", prefix


def parse_prefix(key: str) -> str | None:
    parts = key.split("_", 2)
    if len(parts) != 3 or parts[0] != API_KEY_PREFIX:
        return None
    return parts[1]
