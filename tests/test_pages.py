"""Every route rendered, and checked for the thing it exists to say.

The gap this closes is specific. The validation stage checks the data and the
rest of the suite checks the code; between them sits the template layer, which
was verified only by a person looking at it. That is how `/evidence` shipped a
guard reading `committee_edge.buy.n` when the shape is
`committee_edge.buy.conflicted.n`: every test passed, the page returned 200,
and the entire section silently rendered nothing.

So a 200 is not the assertion. Each case names a fact the page exists to
communicate and fails if that fact is missing from the HTML.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from trade_bot import web
from trade_bot.committees import EDGE_STATE_KEY, save_sic_cache
from trade_bot.database import Store
from trade_bot.domain import (
    AssetClass,
    Chamber,
    DisclosedTrade,
    MemberScore,
    Politician,
    TradeSide,
    TransactionType,
)
from trade_bot.recommend import refresh_recommendations

MEMBER = "senate-one-ann"
SYMBOL = "ACME"
# Relative to the clock on purpose. The board only ranks disclosures inside a
# rolling window, and `refresh_recommendations` reads the real date, so fixed
# dates would quietly stop producing a board and every board assertion here
# would pass vacuously from the day the window moved past them.
FILED = date.today() - timedelta(days=7)
TRADED = FILED - timedelta(days=21)


def _series(days: int, first: float, step: float) -> pd.DataFrame:
    """A rising weekday series ending today, so every lookup resolves."""
    stamps = pd.bdate_range(end=pd.Timestamp(date.today()), periods=days)
    closes = [first + step * index for index in range(days)]
    return pd.DataFrame(
        {
            "date": stamps,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1_000_000.0] * days,
        }
    )


@pytest.fixture
def seeded(settings, prices):
    """One member, one committee seat, both directions traded, scores stored."""
    store = Store(settings.database_url)
    store.initialize()
    store.upsert_politicians(
        [
            Politician(
                politician_id=MEMBER,
                full_name="Ann One",
                first_name="Ann",
                last_name="One",
                chamber=Chamber.SENATE,
                state="KS",
                party="Republican",
                bioguide_id="O000001",
            )
        ]
    )
    store.add_trades(
        [
            DisclosedTrade(
                trade_id=f"t{index}",
                filing_id="senate:2026:1",
                politician_id=MEMBER,
                chamber=Chamber.SENATE,
                symbol=SYMBOL,
                asset_name="Acme Corporation Common Stock (ACME)",
                asset_class=AssetClass.STOCK,
                transaction_type=kind,
                transaction_date=TRADED,
                disclosure_date=FILED,
                amount_low=1001.0,
                amount_high=15000.0,
            )
            for index, kind in enumerate((TransactionType.PURCHASE, TransactionType.SALE))
        ]
    )
    store.replace_member_scores(
        [
            MemberScore(
                politician_id=MEMBER,
                horizon_days=horizon,
                side=side,
                trades_scored=20,
                win_rate=0.6,
                raw_excess_return=0.08,
                score=0.06,
                rankable=True,
                best_symbol=SYMBOL,
                best_symbol_excess=0.2,
                best_symbol_side=TradeSide.SELL if side is TradeSide.SELL else TradeSide.BUY,
            )
            for horizon in (30, 90, 180, 365)
            for side in (TradeSide.ALL, TradeSide.BUY, TradeSide.SELL)
        ]
    )
    store.replace_committee_memberships(
        [
            {
                "politician_id": MEMBER,
                "committee_id": "SSHR",
                "committee_name": "Senate Committee on Health, Education, Labor, and Pensions",
                "chamber": "senate",
                "title": "Chairman",
                "rank": 1,
                "jurisdiction": "pharmaceuticals and health services",
            }
        ]
    )
    save_sic_cache(
        settings,
        {SYMBOL: {"sic": "2834", "industry": "Pharmaceutical Preparations", "name": "Acme"}},
    )
    store.set_state(
        EDGE_STATE_KEY,
        {
            "horizon_days": 90,
            "lookback_days": 1095,
            "measured_at": datetime.now(UTC).isoformat(),
            "buy": {
                "conflicted": {"n": 510, "median_excess": -0.0271, "beat_rate": 0.40},
                "other": {"n": 10435, "median_excess": -0.0157, "beat_rate": 0.45},
                "edge": -0.0114,
            },
            "sell": {
                "conflicted": {"n": 540, "median_excess": 0.0319, "beat_rate": 0.63},
                "other": {"n": 6455, "median_excess": 0.0175, "beat_rate": 0.56},
                "edge": 0.0144,
            },
        },
    )
    prices.save(SYMBOL, _series(200, 100.0, 0.5))
    refresh_recommendations(settings, store, prices)
    return settings, store


@pytest.fixture
def client(seeded):
    settings, store = seeded
    application = web.create_app()
    with TestClient(application) as opened:
        application.state.store = store
        application.state.settings = settings
        yield opened


def _html(client: TestClient, route: str) -> str:
    response = client.get(route)
    assert response.status_code == 200, f"{route} returned {response.status_code}"
    return response.text


@pytest.mark.parametrize(
    "route",
    [
        "/",
        "/digest",
        "/evidence",
        "/members",
        "/trades",
        "/watchlists",
        "/politics",
        "/news",
        "/data",
        "/secrets",
        f"/members/{MEMBER}",
        f"/tickers/{SYMBOL}",
    ],
)
def test_every_route_renders(client, route):
    body = _html(client, route)
    assert "<title>" in body
    # A Jinja undefined renders as the empty string, so a broken expression is
    # silent. A literal brace in the output means one was left unrendered.
    assert "{{" not in body and "{%" not in body


def test_the_board_shows_both_directions(client):
    """A sell-only ticker must reach the board, and never as a buy."""
    body = _html(client, "/")
    assert 'data-side="sell"' in body
    assert 'data-side="buy"' in body


def test_the_ticker_page_carries_the_position_ledger(client):
    body = _html(client, f"/tickers/{SYMBOL}")
    assert "Who is building a position" in body
    assert "flow flow--" in body, "the direction word is the point of the ledger"
    assert "Pharmaceutical Preparations" in body


def test_the_member_page_names_the_committee_without_accusing(client):
    body = _html(client, f"/members/{MEMBER}")
    assert "Health, Education, Labor, and Pensions" in body
    assert "not because it implies anything" in body


def test_the_evidence_page_renders_the_committee_measurement(client):
    """The regression this whole file exists for.

    The guard read `committee_edge.buy.n` against a shape of
    `committee_edge.buy.conflicted.n`, so the section vanished and the page
    still returned 200. Asserting on a number from the payload is the only
    thing that catches it.
    """
    body = _html(client, "/evidence")
    assert "Does sitting on the committee help?" in body
    assert "+3.2%" in body, "the sell-side median must reach the page"
    assert "-2.7%" in body, "and so must the buy-side one, which is negative"
    assert "Dividends are not." in body


def test_the_data_page_renders_the_validation_report(client, seeded):
    from trade_bot.prices import PriceRepository
    from trade_bot.validate import run_validation

    settings, store = seeded
    run_validation(settings, store, PriceRepository(settings))
    body = _html(client, "/data")
    assert "Plausibility checks" in body
    assert "finding finding--" in body


def test_a_page_with_no_data_still_explains_itself(settings):
    """An empty deployment must not render as a blank page."""
    store = Store(settings.database_url)
    store.initialize()
    application = web.create_app()
    with TestClient(application) as opened:
        application.state.store = store
        application.state.settings = settings
        assert "Nothing on the board yet" in _html(opened, "/")
        assert "has not run yet" in _html(opened, "/digest")
        assert "Not measured yet" in _html(opened, "/evidence")


def test_the_ranking_names_who_to_copy_and_where_the_line_is(client, seeded):
    """The page's whole job is the cut, not the ordering.

    A hundred members sorted by score does not answer "who do I copy" — the
    measured edge sits in the top three and inverts below rank six, so the
    three are lifted out and the bands are separated by what copying each one
    actually returned. If the backtest disappears the page must fall back to
    saying the order is a hypothesis rather than asserting a cut it cannot
    support.
    """
    _settings, store = seeded
    store.set_state(
        "board.backtest",
        {
            "summary": {"positions": 0},
            "by_rank": [
                {"rank": "1-3", "positions": 99, "mean_excess": 0.0663, "beat_rate": 0.636},
                {"rank": "4-6", "positions": 48, "mean_excess": 0.0007, "beat_rate": 0.458},
                {"rank": "7+", "positions": 101, "mean_excess": -0.0112, "beat_rate": 0.347},
            ],
        },
    )
    body = _html(client, "/members")
    assert "Best at buying" in body
    assert "Ann One" in body
    # Unsigned: a rate is not a gain, and "+64% of the time" reads as one.
    assert ">64%</strong>" in body and "of the time" in body

    # Without the measurement there is no cut to claim, and the page has to
    # say so rather than presenting the same order as though it were tested.
    store.set_state("board.backtest", {"summary": {"positions": 0}})
    fallback = _html(client, "/members?sort=all")
    assert "treat this order as a hypothesis" in fallback

    # And the sell ranking must never borrow the buy measurement's authority.
    selling = _html(client, "/members?sort=sell")
    assert "The backtest does not cover this side." in selling


def test_the_ranking_flags_a_score_built_on_too_few_trades(client, seeded):
    """The biggest number must not read as the safest bet.

    A member barely past the minimum has a score that is mostly prior, and
    saying so is the difference between a ranking and a recommendation.
    """
    _settings, store = seeded
    store.replace_member_scores(
        [
            MemberScore(
                politician_id=MEMBER,
                horizon_days=horizon,
                side=side,
                trades_scored=5,
                win_rate=0.8,
                raw_excess_return=0.30,
                score=0.11,
                rankable=True,
            )
            for horizon in (30, 90, 180, 365)
            for side in (TradeSide.ALL, TradeSide.BUY, TradeSide.SELL)
        ]
    )
    body = _html(client, "/members")
    assert "thin sample" in body.lower()
    assert "Thin sample." in body, "and the frame says why, not just that"


def test_no_page_still_uses_the_purple_annotation_token(client):
    """The microfilm palette is a negative, not this palette inverted.

    `--violet` was renamed to `--annot` because it holds film blue on the
    negative and violet on paper. A template still asking for `--violet` gets
    an undefined custom property and silently falls back to the initial value,
    which renders as black text on a black ground.
    """
    for route in ("/", "/members", "/data", f"/tickers/{SYMBOL}"):
        assert "--violet" not in _html(client, route)


def test_a_member_with_no_purchases_is_not_offered_as_someone_to_copy(client, seeded):
    """The finding that started the audit.

    A member ranked second on the "who to copy" page had twenty-five sales and
    no purchases at all, beneath a backtest that only ever takes buy
    positions — so not one of their trades could be in the number vouching for
    them. They now cannot appear in the buy ranking, and where they do appear
    the page says what following them would and would not mean.
    """
    _settings, store = seeded
    store.replace_member_scores(
        [
            MemberScore(
                politician_id=MEMBER,
                horizon_days=horizon,
                side=side,
                trades_scored=25,
                win_rate=0.7,
                raw_excess_return=0.12,
                score=0.09,
                rankable=True,
            )
            for horizon in (30, 90, 180, 365)
            for side in (TradeSide.ALL, TradeSide.SELL)
        ]
    )
    buying = _html(client, "/members?sort=buy")
    assert "Ann One" in buying, "they are still listed"
    assert "Best at buying" not in buying or "001" not in buying.split("Ann One")[0][-400:]

    combined = _html(client, "/members?sort=all")
    assert "Nothing to copy on the buy side." in combined
    assert "sells only" in combined


def test_the_scorecard_says_it_measured_purchases_only(client, seeded):
    """Half the cards below it are sales, and a short is not copyable."""
    _settings, store = seeded
    store.set_state(
        "board.backtest",
        {
            "summary": {
                "positions": 188,
                "top_n": 3,
                "horizon_days": 90,
                "mean_excess_return": 0.034,
                "median_excess_return": 0.026,
                "excess_ci_low": 0.0014,
                "excess_ci_high": 0.0676,
                "conclusive": True,
                "beat_benchmark_rate": 0.548,
                "beat_benchmark_wins": 103,
                "beat_rate_probability": 0.215,
                "side": "buy",
            }
        },
    )
    body = _html(client, "/")
    assert "past <em>purchases</em>" in body
    assert "not covered by this" in body


def test_the_evidence_page_states_the_interval_and_the_control(client, seeded):
    """A point estimate reads as a fact; the interval is what qualifies it.

    And the control is the only thing on the page that can say whether the
    ranking beats picking names out of the eligible pool at random.
    """
    _settings, store = seeded
    store.set_state(
        "board.backtest",
        {
            "summary": {
                "positions": 188, "top_n": 3, "horizon_days": 90, "rebalances": 22,
                "start": "2024-08-09", "end": "2026-05-11",
                "mean_return": 0.089, "median_return": 0.089,
                "mean_benchmark_return": 0.061,
                "mean_excess_return": 0.0343, "median_excess_return": 0.026,
                "excess_ci_low": 0.0014, "excess_ci_high": 0.0676, "conclusive": True,
                "beat_benchmark_rate": 0.548, "beat_benchmark_wins": 103,
                "beat_rate_probability": 0.215,
                "unique_symbols": 89, "unique_members": 11, "skipped_unpriced": 9,
                "side": "buy", "best": "ACME", "worst": "FLAT",
            },
            "control": {
                "trials": 25, "actual": 0.0343, "median": -0.0037,
                "best": 0.0276, "worst": -0.0568,
                "percentile": 1.0, "beats_chance": True,
            },
            "by_horizon": [
                {"horizon_days": 90, "positions": 188, "mean_excess_return": 0.0343,
                 "excess_ci_low": 0.0014, "excess_ci_high": 0.0676, "conclusive": True,
                 "beat_benchmark_rate": 0.548, "beat_rate_probability": 0.215},
                {"horizon_days": 365, "positions": 341, "mean_excess_return": 0.002,
                 "excess_ci_low": -0.04, "excess_ci_high": 0.046, "conclusive": False,
                 "beat_benchmark_rate": 0.45, "beat_rate_probability": 0.083},
            ],
            "coverage": {
                "actionable_trades": 22558, "unpriced_trades": 1613,
                "unpriced_share": 0.0715, "unpriced_symbols": 639,
                "by_year": [
                    {"year": 2023, "priced": 4420, "unpriced": 802, "unpriced_share": 0.153},
                    {"year": 2026, "priced": 4044, "unpriced": 135, "unpriced_share": 0.032},
                ],
            },
        },
    )
    body = _html(client, "/evidence")
    # The interval, beside the number it qualifies.
    assert "95% confidence" in body
    assert "+0.1%" in body and "+6.8%" in body
    # The beat rate, measured against a coin rather than asserted.
    assert "what a fair coin produces" in body
    # The control, and its verdict.
    assert "picking three from eighty" in body
    assert "the ranking earns its place" in body
    # Every horizon, with the inconclusive one saying so.
    assert "The same test at every horizon" in body
    assert "contains zero" in body
    # And what the measurement could not see.
    assert "What the measurement could not see" in body
    assert "15%" in body, "the oldest year's dropped share"


def test_a_measurement_that_cannot_clear_zero_says_so_on_the_board(client, seeded):
    """An inconclusive result must not be presented as a headline figure."""
    _settings, store = seeded
    store.set_state(
        "board.backtest",
        {
            "summary": {
                "positions": 120, "top_n": 3, "horizon_days": 90,
                "mean_excess_return": 0.004, "median_excess_return": 0.001,
                "excess_ci_low": -0.031, "excess_ci_high": 0.038, "conclusive": False,
                "beat_benchmark_rate": 0.50, "beat_benchmark_wins": 60,
                "beat_rate_probability": 1.0, "side": "buy",
            },
            "control": {"trials": 25, "actual": 0.004, "median": 0.002,
                        "best": 0.03, "worst": -0.04,
                        "percentile": 0.52, "beats_chance": False},
        },
    )
    body = _html(client, "/")
    assert "Tested, and inconclusive" in body
    assert "ranking is not separating anyone" in body
    assert "scorecard--untested" in body


def test_the_roster_row_reports_coverage_of_members_we_track(client, seeded):
    """"165 enriched of 537 on the roster" read as two thirds of Congress to go.

    It never was. Nothing creates a member: the set this application cares
    about is exactly the set who have filed a transaction report, so the
    several hundred sitting members who never have are not a gap to close.
    The denominator has to be the members we track.
    """
    _settings, store = seeded
    store.set_state(
        "sync.disclosures",
        {
            "sources": {
                "congress-legislators": {
                    "ok": True,
                    "filings_seen": 0,
                    "filings_parsed": 0,
                    "trades_added": 0,
                    "politicians_added": 0,
                    "details": {
                        "enriched": 195,
                        "tracked": 231,
                        "unmatched_members": 36,
                        "roster_size": 537,
                    },
                }
            }
        },
    )
    body = _html(client, "/data")
    assert "195 of 231 tracked members matched" in body
    assert "36 unmatched" in body
    assert "are not a gap" in body, "and the roster size is explained, not used as a target"
    assert "195 of 537" not in body and "of 537 on the roster" not in body
