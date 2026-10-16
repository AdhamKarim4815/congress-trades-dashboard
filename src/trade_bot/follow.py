"""Watchlists that mirror a member's trades into a notional portfolio.

Nothing here touches a broker. A watchlist names one or more members, and
every disclosed trade of theirs becomes a *notional* fill so the dashboard
can answer the question that matters: if I had copied this person, where
would I be now?

Two rules make the answer honest rather than flattering:

1. Fills are priced at the **disclosure date**, never the member's
   transaction date. The transaction date is unobtainable — you did not know.
2. A sell only closes what the watchlist actually holds. Members routinely
   disclose selling positions they acquired long before the watchlist
   existed, and mirroring those as if you held them would invent gains.

Cost basis is average-cost, which keeps partial sales (the most common
disclosure type) simple and stable.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any
from uuid import uuid4

from trade_bot.activity import JobReporter, NullReporter
from trade_bot.config import Settings
from trade_bot.database import Store
from trade_bot.domain import (
    MirrorFill,
    MirrorPortfolio,
    MirrorPosition,
    TransactionType,
    Watchlist,
)
from trade_bot.prices import PriceRepository


def new_watchlist_id() -> str:
    return uuid4().hex[:16]


def fill_id(watchlist_id: str, trade_id: str) -> str:
    """Deterministic, so re-running a mirror never double-buys."""
    return hashlib.sha256(f"{watchlist_id}|{trade_id}".encode()).hexdigest()[:32]


def create_watchlist(
    settings: Settings,
    store: Store,
    name: str,
    politician_ids: list[str],
    allocation_mode: str | None = None,
    allocation_value: float | None = None,
    mirror_sells: bool | None = None,
    starting_equity: float | None = None,
) -> Watchlist:
    watchlist = Watchlist(
        watchlist_id=new_watchlist_id(),
        name=name.strip() or "Untitled watchlist",
        politician_ids=politician_ids,
        allocation_mode=allocation_mode or settings.follow.default_allocation_mode,
        allocation_value=allocation_value or settings.follow.default_allocation_dollars,
        mirror_sells=(
            settings.follow.mirror_sells if mirror_sells is None else mirror_sells
        ),
        min_amount_low=settings.follow.min_amount_low,
        starting_equity=starting_equity or settings.starting_equity,
    )
    store.create_watchlist(watchlist)
    store.append_event(watchlist, subject=watchlist.watchlist_id)
    return watchlist


def _target_notional(
    watchlist: dict[str, Any], trade: dict[str, Any], equity: float
) -> float:
    """How many dollars this watchlist puts into one mirrored buy."""
    mode = watchlist["allocation_mode"]
    value = float(watchlist["allocation_value"])
    if mode == "proportional_to_amount":
        # Scale the member's disclosed bracket midpoint down to the
        # watchlist's own size. The disclosed range is wide, so this is a
        # rough proportionality, not a claim about their real position.
        midpoint = float(trade.get("amount_midpoint", 0.0) or 0.0)
        return max(value * (midpoint / 50_000.0), value * 0.1)
    if mode == "percent_of_equity":
        return max(equity, 0.0) * (value / 100.0)
    return value


@dataclass
class _Holding:
    shares: float = 0.0
    cost_basis: float = 0.0
    realized: float = 0.0
    fills: int = 0
    opened_on: date | None = None

    @property
    def average_cost(self) -> float:
        return self.cost_basis / self.shares if self.shares > 0 else 0.0


def mirror_watchlist(
    settings: Settings,
    store: Store,
    repository: PriceRepository,
    watchlist_id: str,
    reporter: JobReporter | None = None,
) -> dict[str, Any]:
    """Generate any fills this watchlist is missing for its members' trades."""
    reporter = reporter or NullReporter()
    watchlist = store.get_watchlist(watchlist_id)
    if watchlist is None:
        raise KeyError(f"unknown watchlist {watchlist_id}")
    if not watchlist["politician_ids"]:
        return {"fills_added": 0, "skipped": 0, "unpriced": 0, "reason": "no members followed"}

    already = store.mirrored_trade_ids(watchlist_id)
    trades: list[dict[str, Any]] = []
    for politician_id in watchlist["politician_ids"]:
        trades.extend(
            store.list_trades(politician_id=politician_id, actionable_only=True, limit=10_000)
        )
    # Chronological by disclosure: a sell must never be processed before the
    # buy that gave the watchlist something to sell.
    trades.sort(key=lambda trade: (trade["disclosure_date"], trade["trade_id"]))

    holdings: dict[str, _Holding] = defaultdict(_Holding)
    for fill in store.list_mirror_fills(watchlist_id):
        holding = holdings[fill["symbol"]]
        if TransactionType(fill["side"]).is_buy:
            holding.shares += fill["shares"]
            holding.cost_basis += fill["notional"]
        else:
            holding.shares = max(holding.shares - fill["shares"], 0.0)
            holding.cost_basis = holding.average_cost * holding.shares

    equity = float(watchlist["starting_equity"])
    minimum = float(watchlist["min_amount_low"])
    new_fills: list[MirrorFill] = []
    skipped = unpriced = 0
    total = len(trades)

    for index, trade in enumerate(trades, start=1):
        if index % 200 == 0:
            reporter.step(f"mirroring trade {index}/{total}", index, total)
        if trade["trade_id"] in already:
            continue
        if trade["amount_low"] < minimum:
            skipped += 1
            continue

        transaction_type = TransactionType(trade["transaction_type"])
        symbol = trade["symbol"]
        price, priced_on = repository.close_on_or_after(symbol, trade["disclosure_date"])
        if price is None or price <= 0:
            unpriced += 1
            continue

        holding = holdings[symbol]
        if transaction_type.is_buy:
            notional = _target_notional(watchlist, trade, equity)
            shares = notional / price
            holding.shares += shares
            holding.cost_basis += notional
            holding.fills += 1
            if holding.opened_on is None:
                holding.opened_on = priced_on
        elif transaction_type.is_sell:
            if not watchlist["mirror_sells"]:
                skipped += 1
                continue
            if holding.shares <= 0:
                # The member is selling something this watchlist never bought.
                skipped += 1
                continue
            # A full sale closes the position; a partial sale halves it,
            # since the filing discloses no quantity to be more precise with.
            fraction = 1.0 if transaction_type is TransactionType.SALE else 0.5
            shares = holding.shares * fraction
            notional = shares * price
            holding.realized += notional - holding.average_cost * shares
            holding.shares -= shares
            holding.cost_basis = holding.average_cost * holding.shares
            holding.fills += 1
        else:
            skipped += 1
            continue

        if shares <= 0:
            skipped += 1
            continue

        new_fills.append(
            MirrorFill(
                fill_id=fill_id(watchlist_id, trade["trade_id"]),
                watchlist_id=watchlist_id,
                trade_id=trade["trade_id"],
                politician_id=trade["politician_id"],
                symbol=symbol,
                side=transaction_type,
                shares=shares,
                price=price,
                notional=abs(notional),
                priced_on=priced_on or trade["disclosure_date"],
                price_basis="disclosure_date",
                note=(
                    f"member {transaction_type.value} on {trade['transaction_date']}, "
                    f"disclosed {trade['disclosure_date']}"
                ),
            )
        )

    added = store.add_mirror_fills(new_fills)
    reporter.note(
        f"{watchlist['name']}: {added} new mirrored fills "
        f"({skipped} skipped, {unpriced} unpriceable)"
    )
    return {
        "fills_added": added,
        "skipped": skipped,
        "unpriced": unpriced,
        "trades_considered": total,
    }


def build_portfolio(
    settings: Settings,
    store: Store,
    repository: PriceRepository,
    watchlist_id: str,
) -> MirrorPortfolio:
    """Replay a watchlist's fills into positions marked to the latest close."""
    watchlist = store.get_watchlist(watchlist_id)
    if watchlist is None:
        raise KeyError(f"unknown watchlist {watchlist_id}")

    fills = store.list_mirror_fills(watchlist_id)
    holdings: dict[str, _Holding] = defaultdict(_Holding)
    cash = float(watchlist["starting_equity"])
    realized_total = 0.0

    for fill in fills:
        holding = holdings[fill["symbol"]]
        holding.fills += 1
        if TransactionType(fill["side"]).is_buy:
            holding.shares += fill["shares"]
            holding.cost_basis += fill["notional"]
            cash -= fill["notional"]
            if holding.opened_on is None:
                holding.opened_on = fill["priced_on"]
        else:
            average = holding.average_cost
            sold = min(fill["shares"], holding.shares)
            proceeds = sold * fill["price"]
            realized = proceeds - average * sold
            holding.realized += realized
            realized_total += realized
            holding.shares -= sold
            holding.cost_basis = average * holding.shares
            cash += proceeds

    positions: list[MirrorPosition] = []
    market_value = 0.0
    unrealized_total = 0.0
    unpriced = 0
    for symbol, holding in sorted(holdings.items()):
        if holding.shares <= 1e-9:
            continue
        last_price, _ = repository.latest_close(symbol)
        if last_price is None:
            unpriced += 1
        value = (last_price or 0.0) * holding.shares
        unrealized = value - holding.cost_basis if last_price else None
        market_value += value
        unrealized_total += unrealized or 0.0
        positions.append(
            MirrorPosition(
                watchlist_id=watchlist_id,
                symbol=symbol,
                shares=holding.shares,
                average_cost=holding.average_cost,
                cost_basis=holding.cost_basis,
                last_price=last_price,
                market_value=value if last_price else None,
                unrealized_pnl=unrealized,
                unrealized_return=(
                    unrealized / holding.cost_basis
                    if unrealized is not None and holding.cost_basis > 0
                    else None
                ),
                realized_pnl=holding.realized,
                opened_on=holding.opened_on,
                fill_count=holding.fills,
            )
        )

    positions.sort(key=lambda position: position.market_value or 0.0, reverse=True)
    starting = float(watchlist["starting_equity"])
    equity = cash + market_value
    return MirrorPortfolio(
        watchlist_id=watchlist_id,
        name=watchlist["name"],
        starting_equity=starting,
        cash=cash,
        positions=positions,
        realized_pnl=realized_total,
        unrealized_pnl=unrealized_total,
        market_value=market_value,
        equity=equity,
        total_return=(equity - starting) / starting if starting else 0.0,
        fill_count=len(fills),
        unpriced_trades=unpriced,
        as_of=datetime.now(UTC),
    )


def refresh_all(
    settings: Settings,
    store: Store,
    repository: PriceRepository,
    reporter: JobReporter | None = None,
) -> dict[str, Any]:
    """Bring every active watchlist up to date with the latest disclosures."""
    reporter = reporter or NullReporter()
    watchlists = store.list_watchlists(active_only=True)
    results = {}
    for watchlist in watchlists:
        reporter.step(f"mirroring {watchlist['name']}")
        results[watchlist["watchlist_id"]] = mirror_watchlist(
            settings, store, repository, watchlist["watchlist_id"], reporter
        )
    return {
        "watchlists": len(watchlists),
        "fills_added": sum(result["fills_added"] for result in results.values()),
        "results": results,
    }
