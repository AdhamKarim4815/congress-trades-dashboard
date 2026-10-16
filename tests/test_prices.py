"""Price provider failover, caching and date-aware lookups."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from trade_bot.prices import (
    NasdaqDailyProvider,
    PriceRepository,
    PriceUnavailableError,
    _to_float,
    build_providers,
)


class Boom:
    name = "boom"

    def daily(self, symbol, start, end):
        raise RuntimeError("provider down")


class Empty:
    name = "empty"

    def daily(self, symbol, start, end):
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])


class Works:
    name = "works"

    def daily(self, symbol, start, end):
        return pd.DataFrame(
            {
                "date": pd.to_datetime([date(2026, 1, 5), date(2026, 1, 6)]),
                "open": [10.0, 11.0],
                "high": [10.0, 11.0],
                "low": [10.0, 11.0],
                "close": [10.0, 11.0],
                "volume": [1.0, 1.0],
            }
        )


def test_money_strings_parse():
    assert _to_float("$313.33") == 313.33
    assert _to_float("34,437,190") == 34437190.0
    assert _to_float("N/A") != _to_float("N/A")  # NaN
    assert _to_float(None) != _to_float(None)


def test_failover_moves_past_broken_and_empty_providers(settings, tmp_path):
    repository = PriceRepository(settings, providers=[Boom(), Empty(), Works()])
    repository.root = tmp_path / "p"
    repository.root.mkdir()
    frame, provider = repository.fetch("ACME")
    assert provider == "works"
    assert len(frame) == 2


def test_total_failure_raises_with_every_reason(settings, tmp_path):
    repository = PriceRepository(settings, providers=[Boom(), Empty()])
    repository.root = tmp_path / "p"
    repository.root.mkdir()
    with pytest.raises(PriceUnavailableError) as excinfo:
        repository.fetch("ACME")
    message = str(excinfo.value)
    assert "boom" in message and "empty" in message


def test_ensure_reports_per_symbol_failures_without_aborting(settings, tmp_path):
    class Selective:
        name = "selective"

        def daily(self, symbol, start, end):
            if symbol == "BAD":
                return pd.DataFrame(columns=["date", "close"])
            return Works().daily(symbol, start, end)

    repository = PriceRepository(settings, providers=[Selective()])
    repository.root = tmp_path / "p"
    repository.root.mkdir()
    result = repository.ensure(["GOOD", "BAD"])
    assert result["fetched"] == 1
    assert result["failed"] == 1
    assert "BAD" in result["failures"]


def test_a_symbol_holding_the_last_close_is_skipped(settings, tmp_path, monkeypatch):
    """Skipping is now decided by the data, not by the file's timestamp.

    The cache is current when it holds the last final close, so the test has
    to say which session that is -- previously it only had to write the file
    recently, which said nothing about whether the data was up to date.
    """
    import trade_bot.prices as prices_module

    repository = PriceRepository(settings, providers=[Works()])
    repository.root = tmp_path / "p"
    repository.root.mkdir()
    assert repository.ensure(["ACME"])["fetched"] == 1

    held = pd.Timestamp(repository.load("ACME")["date"].iloc[-1]).date()
    monkeypatch.setattr(prices_module, "last_completed_session", lambda: held)
    assert repository.ensure(["ACME"])["skipped"] == 1
    assert repository.ensure(["ACME"], force=True)["fetched"] == 1

    # A session the cache has not caught up to means there is work to do.
    monkeypatch.setattr(
        prices_module, "last_completed_session", lambda: held + timedelta(days=1)
    )
    assert repository.ensure(["ACME"])["fetched"] == 1


def test_close_on_or_after_rolls_forward_over_a_gap(prices):
    """2026-01-10 is a Saturday."""
    price, when = prices.close_on_or_after("ACME", date(2026, 1, 10))
    assert when == date(2026, 1, 12)
    assert price is not None


def test_close_on_or_before_rolls_backward(prices):
    price, when = prices.close_on_or_before("ACME", date(2026, 1, 11))
    assert when == date(2026, 1, 9)
    assert price is not None


def test_lookups_on_an_unknown_symbol_return_none(prices):
    assert prices.close_on_or_after("NOSUCH", date(2026, 1, 5)) == (None, None)
    assert prices.latest_close("NOSUCH") == (None, None)


def test_return_between_computes_from_the_first_available_close(prices):
    result, entry, exit_price = prices.return_between("ACME", date(2026, 1, 5), date(2026, 1, 9))
    assert entry == pytest.approx(100.0)
    assert exit_price == pytest.approx(102.0)  # four sessions at +0.5
    assert result == pytest.approx(0.02)


def test_flat_series_has_zero_return(prices):
    result, _, _ = prices.return_between("SPY", date(2026, 1, 5), date(2026, 6, 1))
    assert result == pytest.approx(0.0)


def test_keyed_providers_are_omitted_when_no_key_is_stored(settings, monkeypatch):
    """The app must work at zero cost with no account anywhere."""
    monkeypatch.setattr("trade_bot.prices.get_api_key", lambda service: None)
    names = [provider.name for provider in build_providers(settings)]
    assert "twelve-data" not in names
    assert "nasdaq" in names


def test_keyed_provider_appears_once_a_key_exists(settings, monkeypatch):
    monkeypatch.setattr("trade_bot.prices.get_api_key", lambda service: "secret")
    names = [provider.name for provider in build_providers(settings)]
    assert "twelve-data" in names


def test_nasdaq_rows_parse_into_a_tidy_frame(monkeypatch):
    provider = NasdaqDailyProvider()
    rows = [
        {
            "date": "08/07/2026",
            "close": "$313.33",
            "open": "$311.45",
            "high": "$314.81",
            "low": "$310.74",
            "volume": "34,437,190",
        }
    ]
    monkeypatch.setattr(provider, "_request", lambda *args, **kwargs: rows)
    frame = provider.daily("AAPL", date(2026, 1, 1), date(2026, 8, 8))
    assert len(frame) == 1
    assert frame.iloc[0]["close"] == pytest.approx(313.33)
    assert frame.iloc[0]["volume"] == pytest.approx(34437190.0)


def test_the_last_final_close_ignores_days_with_no_session():
    """Freshness has to mean "holds the last close", not "written recently".

    A modification time cannot tell stale apart from unchanged: on a Saturday
    every cached file is old by any clock, but there is no new close to fetch,
    and the old check re-read two thousand symbols to receive exactly what was
    already on disk.
    """
    from datetime import UTC, date, datetime

    from trade_bot.market import last_completed_session

    def at(iso: str) -> date:
        return last_completed_session(datetime.fromisoformat(iso).replace(tzinfo=UTC))

    # A weekend resolves back to Friday, not forward to a session that has
    # not happened.
    assert at("2026-08-08T12:00:00") == date(2026, 8, 7)
    assert at("2026-08-09T12:00:00") == date(2026, 8, 7)
    # Today only counts once the bell has rung.
    assert at("2026-08-07T18:00:00") == date(2026, 8, 6)
    assert at("2026-08-07T21:00:00") == date(2026, 8, 7)
    # Independence Day 2026 falls on a Saturday and is observed on the Friday,
    # so the last final close is the Thursday before it.
    assert at("2026-07-04T18:00:00") == date(2026, 7, 2)
