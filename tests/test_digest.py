"""The daily digest: what changed, filtered to what is worth a message."""

from __future__ import annotations

from datetime import date

from trade_bot.database import Store
from trade_bot.digest import build_digest, load_digest, publish_digest, render_markdown
from trade_bot.domain import (
    AssetClass,
    Chamber,
    DisclosedTrade,
    MemberScore,
    Politician,
    TradeSide,
    TransactionType,
)

TODAY = date(2026, 8, 9)


def _member(store: Store, politician_id: str, name: str) -> None:
    store.upsert_politicians(
        [
            Politician(
                politician_id=politician_id,
                full_name=name,
                last_name=name.split()[-1],
                chamber=Chamber.HOUSE,
            )
        ]
    )


def _score(
    store: Store, politician_id: str, score: float, side: TradeSide = TradeSide.BUY
) -> None:
    """Give a member a record on one side, keeping any already stored."""
    keep = [
        MemberScore(
            politician_id=row["politician_id"],
            horizon_days=90,
            side=TradeSide(row["side"]),
            trades_scored=row["trades_scored"],
            score=row["score"],
            rankable=True,
        )
        for existing_side in (TradeSide.ALL, TradeSide.BUY, TradeSide.SELL)
        for row in store.list_member_scores(horizon_days=90, side=existing_side.value)
    ]
    keep.append(
        MemberScore(
            politician_id=politician_id,
            horizon_days=90,
            side=side,
            trades_scored=20,
            score=score,
            rankable=True,
        )
    )
    store.replace_member_scores(keep)


def _trade(
    store: Store,
    trade_id: str,
    politician_id: str,
    symbol: str,
    kind: TransactionType = TransactionType.PURCHASE,
    filed: date = date(2026, 8, 8),
) -> None:
    store.add_trades(
        [
            DisclosedTrade(
                trade_id=trade_id,
                filing_id=f"house:2026:{trade_id}",
                politician_id=politician_id,
                chamber=Chamber.HOUSE,
                symbol=symbol,
                asset_name=f"{symbol} Common Stock",
                asset_class=AssetClass.STOCK,
                transaction_type=kind,
                transaction_date=date(2026, 7, 10),
                disclosure_date=filed,
                amount_low=1001.0,
                amount_high=15000.0,
            )
        ]
    )


def test_only_top_ranked_members_reach_the_headline(settings, store):
    """A digest that lists every filing is a second feed to ignore.

    Ranked per direction: a purchase is worth reporting when the member is
    near the top *at buying*. How well they exit says nothing about it.
    """
    _member(store, "m1", "Ann One")
    _member(store, "m2", "Bob Two")
    _score(store, "m1", 0.09)
    _score(store, "m2", -0.04)
    _trade(store, "a", "m1", "ACME")
    _trade(store, "b", "m2", "FLAT")

    payload = build_digest(settings, store, since=date(2026, 8, 1), top_rank=1, today=TODAY)
    assert payload["new_disclosures"] == 2
    assert [row["symbol"] for row in payload["headline"]] == ["ACME"]


def test_two_members_moving_the_same_way_is_reported_without_any_ranking(settings, store):
    """Agreement is the one signal that needs nobody's track record."""
    _member(store, "m1", "Ann One")
    _member(store, "m2", "Bob Two")
    _trade(store, "a", "m1", "ACME")
    _trade(store, "b", "m2", "ACME")

    payload = build_digest(settings, store, since=date(2026, 8, 1), today=TODAY)
    assert payload["headline"] == []
    assert payload["clusters"][0]["symbol"] == "ACME"
    assert payload["clusters"][0]["members"] == ["Ann One", "Bob Two"]


def test_one_member_filing_twice_is_not_a_cluster(settings, store):
    _member(store, "m1", "Ann One")
    _trade(store, "a", "m1", "ACME")
    _trade(store, "b", "m1", "ACME")
    payload = build_digest(settings, store, since=date(2026, 8, 1), today=TODAY)
    assert payload["clusters"] == []


def test_a_quiet_day_says_so_rather_than_going_silent(settings, store):
    """A digest that is empty when broken and empty when calm is untrustworthy."""
    payload = build_digest(settings, store, since=date(2026, 8, 1), today=TODAY)
    assert payload["quiet"] is True
    assert "normal state on most days" in render_markdown(payload)


def test_the_window_resumes_from_the_last_digest(settings, store):
    _member(store, "m1", "Ann One")
    _trade(store, "a", "m1", "ACME", filed=date(2026, 8, 5))
    build_digest(settings, store, since=date(2026, 8, 1), today=date(2026, 8, 5))

    _trade(store, "b", "m1", "FLAT", filed=date(2026, 8, 7))
    payload = build_digest(settings, store, today=date(2026, 8, 7))
    assert payload["covers_from"] == "2026-08-06", "the day after the last digest"
    assert payload["new_disclosures"] == 1, "the already-reported trade must not repeat"


def test_publishing_writes_a_file_and_stores_the_payload(settings, store):
    _member(store, "m1", "Ann One")
    _score(store, "m1", 0.09)
    _trade(store, "a", "m1", "ACME")
    result = publish_digest(settings, store, since=date(2026, 8, 1), today=TODAY)
    written = settings.project_root / "var" / "reports" / f"digest-{TODAY.isoformat()}.md"
    assert written.exists()
    assert "ACME" in written.read_text()
    assert result["new_disclosures"] == 1
    assert load_digest(store) is not None


def test_delivery_to_a_job_summary_when_one_exists(settings, store, tmp_path, monkeypatch):
    """The scheduled run already produces a page a person opens.

    That makes it a real delivery channel with no mail server, no webhook and
    no new credential, which is the whole reason the digest is deliverable at
    all under a zero-cost constraint.
    """
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    _member(store, "m1", "Ann One")
    _trade(store, "a", "m1", "ACME")
    result = publish_digest(settings, store, since=date(2026, 8, 1), today=TODAY)
    assert result["delivered_to_job_summary"] is True
    assert "Congressional disclosures" in summary.read_text()


def test_a_member_good_at_selling_does_not_headline_their_purchases(settings, store):
    """Ranked per direction, because the two records answer different things."""
    _member(store, "m1", "Ann One")
    _score(store, "m1", 0.09, side=TradeSide.SELL)
    _trade(store, "buy", "m1", "ACME")
    _trade(store, "sell", "m1", "FLAT", kind=TransactionType.SALE)

    payload = build_digest(settings, store, since=date(2026, 8, 1), top_rank=1, today=TODAY)
    headline = {row["symbol"]: row["side"] for row in payload["headline"]}
    assert headline == {"FLAT": "sell"}, "their sale headlines; their purchase does not"


def test_trades_far_apart_in_time_are_not_a_cluster(settings, store):
    """Filings landing in the same window is an artefact of the lag.

    Five members "buying MSFT together" whose transactions were 1,497 days
    apart is not agreement about anything, and the digest presented it as the
    one signal that needs nobody's track record.
    """
    _member(store, "m1", "Ann One")
    _member(store, "m2", "Bob Two")
    store.add_trades(
        [
            DisclosedTrade(
                trade_id=f"far-{index}",
                filing_id=f"house:2026:{index}",
                politician_id=politician_id,
                chamber=Chamber.HOUSE,
                symbol="ACME",
                asset_name="ACME Common Stock",
                asset_class=AssetClass.STOCK,
                transaction_type=TransactionType.PURCHASE,
                transaction_date=traded,
                disclosure_date=date(2026, 8, 8),
                amount_low=1001.0,
                amount_high=15000.0,
            )
            for index, (politician_id, traded) in enumerate(
                [("m1", date(2026, 7, 10)), ("m2", date(2022, 6, 1))]
            )
        ]
    )
    payload = build_digest(settings, store, since=date(2026, 8, 1), today=TODAY)
    assert payload["clusters"] == []


def test_members_acting_within_the_window_still_cluster(settings, store):
    _member(store, "m1", "Ann One")
    _member(store, "m2", "Bob Two")
    _trade(store, "a", "m1", "ACME")
    _trade(store, "b", "m2", "ACME")
    payload = build_digest(settings, store, since=date(2026, 8, 1), today=TODAY)
    assert payload["clusters"][0]["symbol"] == "ACME"
    assert payload["clusters"][0]["span_days"] == 0
