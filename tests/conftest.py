from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest

from trade_bot.config import Settings
from trade_bot.database import Store
from trade_bot.domain import Chamber, Politician
from trade_bot.prices import PriceRepository


class _InMemoryKeyring:
    """Stand-in for the system keychain.

    Autouse below so no test can ever read, write or delete a key in the
    developer's real macOS keychain.
    """

    priority = 1

    def __init__(self) -> None:
        self.store: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> str | None:
        return self.store.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self.store[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        import keyring.errors

        if (service, username) not in self.store:
            raise keyring.errors.PasswordDeleteError(service)
        del self.store[(service, username)]


@pytest.fixture(autouse=True)
def fake_keyring(monkeypatch):
    import keyring

    backend = _InMemoryKeyring()
    monkeypatch.setattr(keyring, "get_keyring", lambda: backend)
    monkeypatch.setattr(keyring, "get_password", backend.get_password)
    monkeypatch.setattr(keyring, "set_password", backend.set_password)
    monkeypatch.setattr(keyring, "delete_password", backend.delete_password)
    return backend


@pytest.fixture
def settings(tmp_path, monkeypatch) -> Settings:
    # `project_root` is a computed property, so assigning it on the instance
    # silently does nothing -- which is what the previous line here did, and a
    # test that built a `PriceRepository(settings)` wrote Parquet straight into
    # the developer's real `var/prices`. Redirecting the class for the duration
    # of the test is what actually keeps a test's artefacts in its temp
    # directory.
    monkeypatch.setattr(Settings, "project_root", property(lambda self: tmp_path))
    config = Settings(
        database_url=f"sqlite:///{tmp_path / 'test.sqlite3'}",
        starting_equity=10_000.0,
    )
    config.ensure_runtime_dirs()
    return config


@pytest.fixture
def store(settings: Settings) -> Store:
    store = Store(settings.database_url)
    store.initialize()
    return store


class FakePriceProvider:
    """Deterministic prices: a straight ramp, so returns are exactly checkable."""

    name = "fake"

    def __init__(self, series: dict[str, list[tuple[date, float]]]) -> None:
        self.series = series

    def daily(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        rows = self.series.get(symbol.upper())
        if not rows:
            return pd.DataFrame(
                columns=["date", "open", "high", "low", "close", "volume"]
            )
        return pd.DataFrame(
            {
                "date": pd.to_datetime([row[0] for row in rows]),
                "open": [row[1] for row in rows],
                "high": [row[1] for row in rows],
                "low": [row[1] for row in rows],
                "close": [row[1] for row in rows],
                "volume": [1000.0] * len(rows),
            }
        )


def build_series(start: date, days: int, first: float, step: float) -> list[tuple[date, float]]:
    """Weekday-only closes rising by `step` each session."""
    rows: list[tuple[date, float]] = []
    cursor = start
    price = first
    while len(rows) < days:
        if cursor.weekday() < 5:
            rows.append((cursor, price))
            price += step
        cursor += timedelta(days=1)
    return rows


@pytest.fixture
def prices(settings: Settings) -> PriceRepository:
    """A repository preloaded with two symbols and a flat benchmark.

    ACME climbs, SPY is flat, so excess return is unambiguous in assertions.
    """
    start = date(2026, 1, 5)
    provider = FakePriceProvider(
        {
            "ACME": build_series(start, 200, 100.0, 0.5),
            "FLAT": build_series(start, 200, 50.0, 0.0),
            "SPY": build_series(start, 200, 400.0, 0.0),
        }
    )
    repository = PriceRepository(settings, providers=[provider])
    for symbol in ("ACME", "FLAT", "SPY"):
        frame, _ = repository.fetch(symbol)
        repository.save(symbol, frame)
    return repository


@pytest.fixture
def member(store: Store) -> Politician:
    politician = Politician(
        politician_id="house-tx-doe-jane",
        full_name="Jane Doe",
        first_name="Jane",
        last_name="Doe",
        chamber=Chamber.HOUSE,
        state="TX",
        district="21",
        party="Republican",
    )
    store.upsert_politicians([politician])
    return politician


def utc(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=UTC)
