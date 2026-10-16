"""Keyring-backed storage for the optional provider API keys.

Every key this application can hold is optional -- filings, prices and news
all come from sources that need no account -- so *looking* for a key must
never be able to fail. On a headless machine (a CI runner, a container)
there is no keyring backend at all, and `keyring.get_password` raises
`NoKeyringError` rather than returning nothing.

That distinction cost a full pipeline run: the price layer asks whether a
Twelve Data key exists before building its provider chain, the raise
propagated, and the stage produced no prices at all. A missing key and a
missing keyring must both simply mean "no key".

Storing a key is different: if that fails the user needs to know, so it
raises.
"""

from __future__ import annotations

import contextlib
import os

import keyring
import keyring.errors

_ACCOUNT = "api-key"


class KeyringUnavailableError(RuntimeError):
    """No keyring backend is available to store a secret on this machine."""


def resolve_api_key(service_name: str, env_var: str) -> str | None:
    """A key from the environment, else the keychain.

    CI has no keychain but does have secrets, and a developer machine has a
    keychain and usually no exported variable. Checking both means the same
    code path serves each without the caller caring which it found.
    """
    return os.environ.get(env_var) or get_api_key(service_name)


def get_api_key(service_name: str) -> str | None:
    """Return a stored key, or None -- including when there is no keyring.

    Deliberately swallows backend errors. The caller is asking "is there an
    optional key?", and on a headless runner the honest answer is no, not an
    exception.
    """
    try:
        return keyring.get_password(service_name, _ACCOUNT)
    except keyring.errors.NoKeyringError:
        return None
    except Exception:
        return None


def set_api_key(service_name: str, api_key: str) -> None:
    """Store a key, raising clearly if this machine cannot keep secrets."""
    try:
        keyring.set_password(service_name, _ACCOUNT, api_key)
    except keyring.errors.NoKeyringError as exc:
        raise KeyringUnavailableError(
            "no system keyring is available to store an API key on this machine"
        ) from exc


def delete_api_key(service_name: str) -> None:
    with contextlib.suppress(
        keyring.errors.PasswordDeleteError, keyring.errors.NoKeyringError
    ):
        keyring.delete_password(service_name, _ACCOUNT)


def keyring_available() -> bool:
    """Whether this machine can store a key at all, for the Secrets page."""
    try:
        keyring.get_keyring()
    except Exception:
        return False
    return True
