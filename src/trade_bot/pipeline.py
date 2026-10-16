"""The one-button refresh: everything, in dependency order.

Each stage feeds the next, so the order is not arbitrary:

1. **disclosures** -- new filings define the ticker universe
2. **prices** -- fetched for exactly those tickers plus the benchmark
3. **industries** -- each ticker's SIC code, so committee remits can be
   matched against what a member actually bought
4. **ranking** -- needs prices to score members
5. **news** -- member/ticker feeds need to know who and what matters, so it
   runs after ranking picks the interesting ones
6. **mentions** -- extracted from the figure news fetched in stage 5
7. **watchlists** -- mirrored once new disclosures and prices are in
8. **committees** -- re-measures whether a committee overlap predicts
   anything, because the board's sell side is weighted on the answer
9. **board** -- the opportunity ranking consumes all of the above
10. **validate** -- judges everything the others produced
11. **digest** -- writes down what changed, so the product does not require
    a visit to deliver its one job

Every stage reports through the shared activity log, which is what the
dashboard's status bar renders live. A stage that fails is recorded and the
pipeline continues: a dead news feed must not stop the disclosure sync from
being useful.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from trade_bot.activity import JobReporter, NullReporter
from trade_bot.backtest import refresh_backtest
from trade_bot.committees import measure_committee_edge, sync_industries
from trade_bot.config import Settings
from trade_bot.database import Store
from trade_bot.digest import publish_digest
from trade_bot.disclosures import sync_all
from trade_bot.follow import refresh_all
from trade_bot.news import sync_figure_news, sync_member_news, sync_ticker_news, sync_world_news
from trade_bot.ocr import backfill_scanned_filings
from trade_bot.politics import extract_mentions
from trade_bot.prices import PriceRepository
from trade_bot.ranking import rank_members
from trade_bot.recommend import refresh_recommendations
from trade_bot.validate import run_validation

STAGES = (
    "disclosures",
    "scans",
    "prices",
    "industries",
    "ranking",
    "news",
    "mentions",
    "watchlists",
    "committees",
    "board",
    "backtest",
    "validate",
    "digest",
)


def run_refresh(
    settings: Settings,
    store: Store,
    reporter: JobReporter | None = None,
    stages: list[str] | None = None,
    filing_limit: int | None = None,
    news_symbol_limit: int = 40,
    # Each uncached scan is several vision requests against a per-minute token
    # budget, so this is the stage most able to stretch a run unpredictably.
    # The on-disk cache means a repeat run reads none of them.
    scan_limit: int = 8,
    news_member_limit: int = 40,
) -> dict[str, Any]:
    reporter = reporter or NullReporter()
    wanted = set(stages or STAGES)
    repository = PriceRepository(settings)
    results: dict[str, Any] = {}
    errors: dict[str, str] = {}

    def _stage(name: str, work: Any) -> None:
        if name not in wanted:
            return
        reporter.step(f"{name}: starting")
        try:
            results[name] = work()
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
            reporter.note(f"{name} failed: {exc}", level="error")

    _stage("disclosures", lambda: sync_all(settings, store, reporter, limit=filing_limit))

    def _prices() -> dict[str, Any]:
        symbols = [*store.distinct_trade_symbols(), settings.prices.benchmark_symbol]
        return repository.ensure(symbols, reporter)

    _stage("prices", _prices)
    # Which industry each ticker belongs to, so a committee's remit can be
    # matched against it. One free SEC request per company, cached for ever,
    # capped per run — after the first pass this stage does nothing.
    _stage("industries", lambda: sync_industries(settings, store, reporter))
    _stage("ranking", lambda: rank_members(settings, store, repository, reporter))

    def _news() -> dict[str, Any]:
        board_symbols = [
            trade["symbol"]
            for trade in store.list_trades(
                actionable_only=True,
                disclosed_since=None,
                limit=2000,
            )
            if trade["symbol"]
        ]
        deduped = list(dict.fromkeys(board_symbols))[:news_symbol_limit]
        return {
            "world": sync_world_news(settings, store, reporter),
            "figures": sync_figure_news(settings, store, reporter),
            "members": sync_member_news(
                settings, store, reporter, limit_members=news_member_limit
            ),
            "tickers": sync_ticker_news(settings, store, deduped, reporter),
        }

    # Straight after the sync, so the scanned filings it could not read are
    # recovered before prices, ranking and the board are computed from them.
    # A modest per-run cap keeps a backlog from becoming one large bill; the
    # remainder is picked up by subsequent runs.
    _stage(
        "scans",
        lambda: backfill_scanned_filings(settings, store, reporter, limit=scan_limit),
    )

    _stage("news", _news)
    _stage("mentions", lambda: extract_mentions(settings, store, reporter))
    _stage("watchlists", lambda: refresh_all(settings, store, repository, reporter))
    # Before the board, because the board's sell side weights the committee
    # overlap and the weight is only defensible while the measurement holds.
    _stage(
        "committees",
        lambda: measure_committee_edge(settings, store, repository, reporter),
    )
    _stage("board", lambda: refresh_recommendations(settings, store, repository, reporter))
    # Last, and deliberately part of every run: the claim that following these
    # members works should be re-measured with the data, not asserted once.
    _stage(
        "backtest",
        lambda: refresh_backtest(settings, store, repository, reporter),
    )
    # After everything, because it judges what the other stages produced. The
    # suite can be green while the data is nonsense; this is the stage that
    # notices. It never rewrites a row -- it records what does not add up.
    _stage("validate", lambda: run_validation(settings, store, repository, reporter))
    # Last of all, because the digest reports on the board and the board is
    # not final until validation has had its say about the data behind it.
    _stage("digest", lambda: publish_digest(settings, store, reporter))

    summary = {
        "stages": results,
        "errors": errors,
        "finished_at": datetime.now(UTC).isoformat(),
        "ok": not errors,
    }
    store.set_state("sync.last_refresh", summary)
    return summary
