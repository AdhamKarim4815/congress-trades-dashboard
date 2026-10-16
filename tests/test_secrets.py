"""Optional keys, including on machines that cannot store them.

Every source this dashboard depends on is keyless, so *asking* whether an
optional key exists must never raise. On a headless CI runner there is no
keyring backend at all, and the raise from that took out the price stage —
which silently produced no prices, no scores, and a green build.
"""

from __future__ import annotations

import keyring.errors
import pytest

from trade_bot import secrets


def test_get_api_key_is_none_when_nothing_is_stored() -> None:
    assert secrets.get_api_key("twelve-data") is None


def test_set_then_get_round_trips() -> None:
    secrets.set_api_key("twelve-data", "super-secret-key")
    assert secrets.get_api_key("twelve-data") == "super-secret-key"


def test_delete_removes_a_stored_key() -> None:
    secrets.set_api_key("twelve-data", "super-secret-key")
    secrets.delete_api_key("twelve-data")
    assert secrets.get_api_key("twelve-data") is None


def test_delete_of_a_missing_key_does_not_raise() -> None:
    secrets.delete_api_key("never-set")


def test_services_are_isolated_from_each_other() -> None:
    secrets.set_api_key("twelve-data", "twelve-data-key")
    secrets.set_api_key("fred", "fred-key")
    assert secrets.get_api_key("twelve-data") == "twelve-data-key"
    assert secrets.get_api_key("fred") == "fred-key"


def _no_backend(*args, **kwargs):
    raise keyring.errors.NoKeyringError("No recommended backend was available")


def test_get_returns_none_when_there_is_no_keyring_backend(monkeypatch) -> None:
    """The CI failure, in one assertion.

    A headless runner has no backend; "no key" is the correct answer, not an
    exception that propagates into the price provider chain.
    """
    monkeypatch.setattr(keyring, "get_password", _no_backend)
    assert secrets.get_api_key("twelve-data") is None


def test_get_survives_an_otherwise_broken_backend(monkeypatch) -> None:
    monkeypatch.setattr(
        keyring, "get_password", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("locked"))
    )
    assert secrets.get_api_key("twelve-data") is None


def test_delete_is_quiet_without_a_backend(monkeypatch) -> None:
    monkeypatch.setattr(keyring, "delete_password", _no_backend)
    secrets.delete_api_key("twelve-data")


def test_storing_without_a_backend_raises_clearly(monkeypatch) -> None:
    """Storing is different: silence there would lose the user's key."""
    monkeypatch.setattr(keyring, "set_password", _no_backend)
    with pytest.raises(secrets.KeyringUnavailableError):
        secrets.set_api_key("twelve-data", "k")


def test_price_providers_build_without_a_keyring(settings, monkeypatch) -> None:
    """The actual regression: the provider chain must survive a headless box."""
    monkeypatch.setattr(keyring, "get_password", _no_backend)
    from trade_bot.prices import build_providers

    names = [provider.name for provider in build_providers(settings)]
    assert "nasdaq" in names
    assert "twelve-data" not in names
