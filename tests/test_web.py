"""Dashboard routes, filters and the status API."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from fastapi.testclient import TestClient

from trade_bot.database import Store
from trade_bot.domain import AssetClass, Chamber, DisclosedTrade, Politician, TransactionType
from trade_bot.web import _ago, _money, _percent, create_app


@pytest.fixture
def client(settings, prices, monkeypatch):
    import trade_bot.web as web

    monkeypatch.setattr(web, "get_settings", lambda: settings)
    app = create_app()
    with TestClient(app) as test_client:
        # Share the fixture's price cache so pages render against known data.
        app.state.prices = prices
        store: Store = app.state.store
        store.upsert_politicians(
            [
                Politician(
                    politician_id="house-tx-doe-jane",
                    full_name="Jane Doe",
                    first_name="Jane",
                    last_name="Doe",
                    chamber=Chamber.HOUSE,
                    state="TX",
                    district="21",
                    party="Republican",
                )
            ]
        )
        store.add_trades(
            [
                DisclosedTrade(
                    trade_id="t1",
                    filing_id="house:2026:1",
                    politician_id="house-tx-doe-jane",
                    chamber=Chamber.HOUSE,
                    symbol="ACME",
                    asset_name="ACME Common Stock",
                    asset_class=AssetClass.STOCK,
                    transaction_type=TransactionType.PURCHASE,
                    transaction_date=date(2026, 1, 5),
                    disclosure_date=date(2026, 2, 5),
                    amount_low=1001,
                    amount_high=15000,
                )
            ]
        )
        yield test_client


@pytest.mark.parametrize(
    "path",
    ["/", "/members", "/trades", "/watchlists", "/politics", "/news", "/data", "/secrets"],
)
def test_every_page_renders(client, path):
    response = client.get(path)
    assert response.status_code == 200
    assert "<html" in response.text


def test_member_and_ticker_detail_pages_render(client):
    assert client.get("/members/house-tx-doe-jane").status_code == 200
    assert client.get("/tickers/ACME").status_code == 200


def test_unknown_member_is_a_404_not_a_crash(client):
    assert client.get("/members/nobody").status_code == 404


def test_status_api_reports_activity_and_counts(client):
    payload = client.get("/api/status").json()
    assert "activity" in payload and "market" in payload
    assert payload["counts"]["total"] == 1


def test_activity_api_paginates_by_event_id(client):
    first = client.get("/api/activity?since=0").json()
    assert "events" in first
    later = client.get(f"/api/activity?since={first['last_id']}").json()
    assert later["events"] == []


def test_board_api_returns_json(client):
    payload = client.get("/api/board").json()
    assert "items" in payload


def test_trades_filters_apply(client):
    assert "ACME" in client.get("/trades?symbol=ACME").text
    assert "ACME" not in client.get("/trades?symbol=NOPE").text


def test_creating_a_watchlist_redirects_and_persists(client):
    response = client.post(
        "/watchlists/create",
        data={
            "name": "Jane",
            "politician_ids": ["house-tx-doe-jane"],
            "allocation_mode": "fixed_dollars",
            "allocation_value": "1000",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert "Jane" in client.get("/watchlists").text


def test_refresh_action_starts_a_job_and_redirects(client):
    response = client.post(
        "/actions/refresh", data={"stages": ["board"]}, follow_redirects=False
    )
    assert response.status_code == 303


def test_money_filter_scales_and_handles_none():
    assert _money(None) == "—"
    assert _money(1500) == "$1.5k"
    assert _money(2_500_000) == "$2.5M"
    assert _money(12.5) == "$12.50"


def test_percent_filter_signs_and_handles_none():
    assert _percent(None) == "—"
    assert _percent(0.1234) == "+12.3%"
    assert _percent(-0.02) == "-2.0%"


def test_ago_filter_accepts_datetimes_dates_and_iso_strings():
    """State blobs store ISO strings while rows return datetimes; the filter
    previously crashed on the former."""
    assert _ago(None) == "—"
    assert _ago(datetime.now(UTC)) == "just now"
    assert _ago(datetime.now(UTC).isoformat()) == "just now"
    assert _ago(date.today()).endswith("ago") or _ago(date.today()) == "just now"
    assert _ago("not a timestamp") == "not a timestamp"
