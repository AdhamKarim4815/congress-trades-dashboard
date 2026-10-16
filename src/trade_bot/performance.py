"""How disclosed trades actually turned out.

Two returns are computed for every trade and they are not interchangeable:

* **member return** runs from the transaction date. It is what the member
  got, and it is the number every "Congress beats the market" headline uses.
  It is a *price* move, unsigned by direction -- averaging it across a
  member's buys and sells describes nothing, which is why `summarize` no
  longer offers a mean of it.
* **achievable return** runs from the disclosure date -- the first day the
  public could have known. It is the only number a copier could have earned,
  and it is what member ranking and the mirror portfolios are built on.

The gap between them is the reporting lag, which averages weeks and is
routinely the difference between a member's spectacular trade and a
copier's mediocre one. Reporting only the first would flatter every member
in the application.

Excess return is signed by intent: a purchase that beat the benchmark scores
positive, and so does a sale that dodged a fall. That way "did this member
make a good decision" is one comparable number across buys and sells.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import date, timedelta
from typing import Any

from trade_bot.activity import JobReporter, NullReporter
from trade_bot.domain import MAX_PLAUSIBLE_LAG_DAYS, TradePerformance, TransactionType
from trade_bot.prices import PriceRepository


def _direction(transaction_type: TransactionType) -> int:
    """+1 when the member was buying exposure, -1 when shedding it."""
    if transaction_type.is_sell:
        return -1
    return 1


def evaluate_trade(
    trade: dict[str, Any],
    repository: PriceRepository,
    benchmark_symbol: str,
    horizon_days: int | None = None,
    today: date | None = None,
) -> TradePerformance:
    """Price one disclosed trade over one horizon.

    A horizon of `None` measures to the latest close (an open-ended "how has
    it done since"), otherwise the window closes `horizon_days` after the
    disclosure date. A trade whose horizon has not elapsed yet is still
    measured to today, and `note` records that it is incomplete.
    """
    today = today or date.today()
    symbol = (trade.get("symbol") or "").upper()
    transaction_type = TransactionType(trade["transaction_type"])
    transaction_date = trade["transaction_date"]
    disclosure_date = trade["disclosure_date"]

    base = {
        "trade_id": trade["trade_id"],
        "politician_id": trade["politician_id"],
        "symbol": symbol,
        "transaction_type": transaction_type,
        "transaction_date": transaction_date,
        "disclosure_date": disclosure_date,
        "amount_midpoint": trade.get("amount_midpoint", 0.0),
        "horizon_days": horizon_days,
    }
    if not symbol:
        return TradePerformance(**base, priced=False, note="no ticker on this disclosure line")

    window_end: date | None = None
    note: str | None = None
    if horizon_days is not None:
        window_end = disclosure_date + timedelta(days=horizon_days)
        if window_end > today:
            window_end = today
            note = f"only {(today - disclosure_date).days}d of the {horizon_days}d horizon elapsed"

    # A transaction date the filing cannot possibly mean must not be priced.
    # `repair_transaction_date` fixes the century misreads; what survives is a
    # date nothing can resolve, and entering a position at a 1975 close and
    # exiting at a 2026 one is not a member return, it is a fiction. The
    # copier's return is unaffected -- it starts at the disclosure date, which
    # is taken from the filing index and is never in doubt.
    believable_entry = 0 <= (disclosure_date - transaction_date).days <= MAX_PLAUSIBLE_LAG_DAYS
    entry_price = (
        repository.close_on_or_after(symbol, transaction_date)[0] if believable_entry else None
    )
    disclosure_price, priced_on = repository.close_on_or_after(symbol, disclosure_date)
    if disclosure_price is None:
        return TradePerformance(
            **base,
            entry_price=entry_price,
            priced=False,
            note=f"no price history for {symbol}",
        )

    exit_price, _ = (
        repository.latest_close(symbol)
        if window_end is None
        else repository.close_on_or_before(symbol, window_end)
    )
    if exit_price is None:
        exit_price, _ = repository.latest_close(symbol)
    if exit_price is None:
        return TradePerformance(
            **base,
            entry_price=entry_price,
            disclosure_price=disclosure_price,
            priced=False,
            note=f"no closing price for {symbol}",
        )

    direction = _direction(transaction_type)
    member_return = (
        (exit_price - entry_price) / entry_price if entry_price and entry_price > 0 else None
    )
    achievable_return = (exit_price - disclosure_price) / disclosure_price

    benchmark_return: float | None = None
    if benchmark_symbol:
        benchmark_start = priced_on or disclosure_date
        benchmark_return, _, _ = repository.return_between(
            benchmark_symbol, benchmark_start, window_end
        )

    excess_return = (
        direction * (achievable_return - benchmark_return)
        if benchmark_return is not None
        else direction * achievable_return
    )

    return TradePerformance(
        **base,
        entry_price=entry_price,
        disclosure_price=disclosure_price,
        exit_price=exit_price,
        member_return=member_return,
        achievable_return=achievable_return,
        benchmark_return=benchmark_return,
        excess_return=excess_return,
        priced=True,
        note=note,
    )


def evaluate_trades(
    trades: Sequence[dict[str, Any]],
    repository: PriceRepository,
    benchmark_symbol: str,
    horizon_days: int | None = None,
    today: date | None = None,
    reporter: JobReporter | None = None,
) -> list[TradePerformance]:
    reporter = reporter or NullReporter()
    total = len(trades)
    results: list[TradePerformance] = []
    for index, trade in enumerate(trades, start=1):
        if total > 200 and index % 200 == 0:
            reporter.step(f"pricing trade {index}/{total}", index, total)
        results.append(
            evaluate_trade(trade, repository, benchmark_symbol, horizon_days, today)
        )
    return results


def summarize(performances: Iterable[TradePerformance]) -> dict[str, Any]:
    """Aggregate stats over a set of priced trades."""
    priced = [item for item in performances if item.priced and item.excess_return is not None]
    if not priced:
        return {
            "trades": 0,
            "priced": 0,
            "win_rate": 0.0,
            "mean_excess": 0.0,
            "median_excess": 0.0,
            "mean_achievable": 0.0,
            "best": None,
            "worst": None,
        }
    excesses = sorted(item.excess_return or 0.0 for item in priced)
    middle = len(excesses) // 2
    median = (
        excesses[middle]
        if len(excesses) % 2
        else (excesses[middle - 1] + excesses[middle]) / 2
    )
    achievable = [item.achievable_return for item in priced if item.achievable_return is not None]
    best = max(priced, key=lambda item: item.excess_return or 0.0)
    worst = min(priced, key=lambda item: item.excess_return or 0.0)
    return {
        "trades": len(list(priced)),
        "priced": len(priced),
        "win_rate": sum(1 for value in excesses if value > 0) / len(excesses),
        "mean_excess": sum(excesses) / len(excesses),
        "median_excess": median,
        "mean_achievable": sum(achievable) / len(achievable) if achievable else 0.0,
        "best": best,
        "worst": worst,
    }
