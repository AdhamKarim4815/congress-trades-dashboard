"""A member's running position in a name, rather than a list of events.

Every disclosure in this application has been an isolated fact: on this day
this person bought somewhere between $1,001 and $15,000 of this company. The
question a reader actually has is the one no single row answers — *is this
person building a position or leaving one?* Four purchases across six months
and one sale read very differently from one purchase and four sales, and a
flat feed shows both as "five trades".

Congress does not disclose share counts, so a real ledger is impossible. What
is possible is a bracketed one, and the arithmetic has to be honest about
that:

* **exposure** is tracked in disclosed dollars, signed by direction, summing
  bracket midpoints. It is an estimate built on estimates and is labelled as
  one everywhere it is shown.
* **bounds** carry the same walk done with the bracket floors and ceilings,
  so "between $16k and $265k of net buying" can be stated instead of a
  midpoint pretending to be a measurement.
* **a sale of an unknown fraction cannot be netted against a purchase.** A
  member who bought band A and sold band A may have exited entirely or
  trimmed a tenth of a much larger holding they never had to disclose,
  because holdings below $1,000 and positions established before they took
  office do not appear here at all. `round_trip` therefore means "sold at
  least as much as we ever saw them buy", not "owns none" -- and a member with
  only sales on record is `selling_down` rather than `trimming`, because
  nothing here watched them build the position they are leaving.

What the ledger can say without qualification is the *sequence*: when they
started, whether they have added since, whether they have begun selling, and
how long the position has been open. That is the part no other view provides.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from trade_bot.domain import TransactionType
from trade_bot.prices import PriceRepository

# Every state the sequence can be in, named for what it is rather than for
# what it implies about a holding we cannot see.
# "accumulating" -- bought more than once and never sold, or last move was a buy
# "held"          -- exactly one purchase on record, no sales
# "trimming"      -- sold some of what we watched them build
# "selling_down"  -- only sales on record; the holding predates what we can see
# "round_trip"    -- sold at least as much as they were ever seen buying
Direction = str


@dataclass
class Lot:
    """One disclosure line, kept whole so the ledger can be audited."""

    trade_id: str
    transaction_date: date
    disclosure_date: date
    side: str
    amount_low: float
    amount_high: float
    amount_midpoint: float
    price_at_disclosure: float | None = None

    @property
    def signed_midpoint(self) -> float:
        return self.amount_midpoint if self.side == "buy" else -self.amount_midpoint


@dataclass
class Position:
    """One member's running exposure to one symbol."""

    politician_id: str
    member_name: str
    symbol: str
    lots: list[Lot] = field(default_factory=list)
    bought: float = 0.0
    sold: float = 0.0
    bought_low: float = 0.0
    bought_high: float = 0.0
    sold_low: float = 0.0
    sold_high: float = 0.0
    buy_count: int = 0
    sell_count: int = 0
    first_seen: date | None = None
    last_action: date | None = None
    last_side: str = ""
    # Disclosed-dollar-weighted average of the price on each buy's disclosure
    # date. Not a cost basis -- the member's own basis is their transaction
    # price, which is unknowable in size -- but it is the basis a copier
    # following this member would actually have.
    copier_basis: float | None = None
    last_price: float | None = None

    @property
    def net_exposure(self) -> float:
        return self.bought - self.sold

    @property
    def net_low(self) -> float:
        """The most conservative reading: least bought, most sold."""
        return self.bought_low - self.sold_high

    @property
    def net_high(self) -> float:
        return self.bought_high - self.sold_low

    @property
    def days_open(self) -> int | None:
        if self.first_seen is None or self.last_action is None:
            return None
        return (self.last_action - self.first_seen).days

    @property
    def direction(self) -> Direction:
        """What the sequence of lots is doing, in one word.

        Deliberately about sequence rather than size: the sizes are brackets
        and the sequence is fact.

        Two of these words were wrong. `closed` was unreachable -- by the time
        the size test ran, both counts were already non-zero -- so the state it
        named could never be shown. And a member who has only ever sold was
        called `trimming`, which claims we watched them reduce a position we
        never saw them open; they are selling down a holding that predates the
        record, which is a different and weaker fact.
        """
        if not self.sell_count:
            return "accumulating" if self.buy_count > 1 else "held"
        if not self.buy_count:
            return "selling_down"
        if self.sold >= self.bought:
            return "round_trip"
        return "trimming" if self.last_side == "sell" else "accumulating"

    @property
    def unrealized_return(self) -> float | None:
        """What a copier who followed every buy would be showing.

        Uses `copier_basis` -- the disclosure-date price, which is the first
        price a reader could have paid -- never the member's own entry.
        """
        if not self.copier_basis or not self.last_price or self.copier_basis <= 0:
            return None
        return (self.last_price - self.copier_basis) / self.copier_basis

    def as_dict(self) -> dict[str, Any]:
        return {
            "politician_id": self.politician_id,
            "member_name": self.member_name,
            "symbol": self.symbol,
            "direction": self.direction,
            "bought": self.bought,
            "sold": self.sold,
            "net_exposure": self.net_exposure,
            "net_low": self.net_low,
            "net_high": self.net_high,
            "buy_count": self.buy_count,
            "sell_count": self.sell_count,
            "first_seen": self.first_seen,
            "last_action": self.last_action,
            "last_side": self.last_side,
            "days_open": self.days_open,
            "copier_basis": self.copier_basis,
            "last_price": self.last_price,
            "unrealized_return": self.unrealized_return,
            "lots": [
                {
                    "trade_id": lot.trade_id,
                    "transaction_date": lot.transaction_date,
                    "disclosure_date": lot.disclosure_date,
                    "side": lot.side,
                    "amount_low": lot.amount_low,
                    "amount_high": lot.amount_high,
                    "price_at_disclosure": lot.price_at_disclosure,
                }
                for lot in self.lots
            ],
        }


def _side(transaction_type: str) -> str | None:
    kind = TransactionType(transaction_type)
    if kind.is_buy:
        return "buy"
    if kind.is_sell:
        return "sell"
    # An exchange moves exposure between two names and discloses neither leg's
    # size usefully, so it is recorded nowhere rather than guessed at.
    return None


def build_positions(
    trades: list[dict[str, Any]],
    members: dict[str, dict[str, Any]],
    repository: PriceRepository | None = None,
    symbol: str | None = None,
) -> list[Position]:
    """Fold disclosure lines into one running position per member per symbol.

    Lots are walked oldest first by *disclosure* date, because that is the
    order a reader learned them in, and the copier basis is only meaningful
    in that order.
    """
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for trade in trades:
        ticker = trade.get("symbol")
        if not ticker or (symbol and ticker.upper() != symbol.upper()):
            continue
        if _side(trade["transaction_type"]) is None:
            continue
        grouped[(trade["politician_id"], ticker.upper())].append(trade)

    positions: list[Position] = []
    for (politician_id, ticker), rows in grouped.items():
        rows.sort(key=lambda row: (row["disclosure_date"], row["transaction_date"]))
        position = Position(
            politician_id=politician_id,
            member_name=members.get(politician_id, {}).get("full_name", politician_id),
            symbol=ticker,
        )
        weighted = 0.0
        weight = 0.0
        for row in rows:
            side = _side(row["transaction_type"])
            if side is None:
                continue
            price = (
                repository.close_on_or_after(ticker, row["disclosure_date"])[0]
                if repository is not None
                else None
            )
            lot = Lot(
                trade_id=row["trade_id"],
                transaction_date=row["transaction_date"],
                disclosure_date=row["disclosure_date"],
                side=side,
                amount_low=row.get("amount_low", 0.0),
                amount_high=row.get("amount_high", 0.0),
                amount_midpoint=row.get("amount_midpoint", 0.0),
                price_at_disclosure=price,
            )
            position.lots.append(lot)
            if side == "buy":
                position.bought += lot.amount_midpoint
                position.bought_low += lot.amount_low
                position.bought_high += lot.amount_high
                position.buy_count += 1
                if price:
                    weighted += price * lot.amount_midpoint
                    weight += lot.amount_midpoint
            else:
                position.sold += lot.amount_midpoint
                position.sold_low += lot.amount_low
                position.sold_high += lot.amount_high
                position.sell_count += 1
            if position.first_seen is None:
                position.first_seen = lot.disclosure_date
            position.last_action = lot.disclosure_date
            position.last_side = side

        position.copier_basis = weighted / weight if weight else None
        if repository is not None:
            position.last_price = repository.latest_close(ticker)[0]
        positions.append(position)

    # Largest net exposure first: "who is accumulating" is the question this
    # exists to answer, and the answer should be at the top.
    positions.sort(key=lambda item: item.net_exposure, reverse=True)
    return positions


def accumulation_summary(positions: list[Position]) -> dict[str, Any]:
    """Headline counts for a ticker page: who is building, who is leaving."""
    counts: dict[str, int] = defaultdict(int)
    for position in positions:
        counts[position.direction] += 1
    net = sum(position.net_exposure for position in positions)
    return {
        "members": len(positions),
        "accumulating": counts["accumulating"],
        "trimming": counts["trimming"],
        "selling_down": counts["selling_down"],
        "closed": counts["round_trip"],
        "held": counts["held"],
        "net_exposure": net,
        "net_low": sum(position.net_low for position in positions),
        "net_high": sum(position.net_high for position in positions),
    }
