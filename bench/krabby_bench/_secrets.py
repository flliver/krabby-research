"""Fernet-encrypted device IAM key storage."""
from __future__ import annotations

import base64
import hashlib
import json
import secrets
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

_CONF_DIR = Path("/etc/krabby-bench")
_SALT_PATH = _CONF_DIR / "salt"
_SECRETS_ENC = _CONF_DIR / "secrets.enc"


def _machine_id() -> str:
    return Path("/etc/machine-id").read_text().strip()


def _fernet_key(salt_path: Path = _SALT_PATH) -> bytes:
    salt = salt_path.read_text().strip()
    return base64.urlsafe_b64encode(hashlib.sha256(f"{_machine_id()}{salt}".encode()).digest())


def write_device_key(
    key_id: str,
    key_secret: str,
    salt_path: Path = _SALT_PATH,
    enc_path: Path = _SECRETS_ENC,
) -> None:
    """Encrypt and persist the IAM access key. Generates a new salt on each call."""
    salt_path.parent.mkdir(parents=True, exist_ok=True)
    salt = secrets.token_hex(32)
    salt_path.write_text(salt)
    salt_path.chmod(0o600)
    payload = json.dumps({"key_id": key_id, "key_secret": key_secret})
    enc_path.write_bytes(Fernet(_fernet_key(salt_path)).encrypt(payload.encode()))
    enc_path.chmod(0o600)


def read_device_key(
    salt_path: Path = _SALT_PATH,
    enc_path: Path = _SECRETS_ENC,
) -> tuple[str, str] | None:
    """Return (key_id, key_secret), or None if unavailable or corrupted."""
    if not enc_path.exists() or not salt_path.exists():
        return None
    try:
        payload = Fernet(_fernet_key(salt_path)).decrypt(enc_path.read_bytes())
        d = json.loads(payload)
        return d["key_id"], d["key_secret"]
    except (InvalidToken, KeyError, ValueError, OSError):
        return None
