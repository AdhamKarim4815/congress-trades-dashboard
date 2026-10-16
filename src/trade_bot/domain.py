"""Immutable records for congressional copy-trading.

Everything the dashboard shows is derived from three kinds of fact: a
*filing* (a member submitted a periodic transaction report), a *disclosed
trade* (one line inside that filing), and a *price* (what the market did
afterwards). Mirror portfolios, member rankings and recommendations are all
computed on top of those, never stored as ground truth.

A note on dates that runs through the whole application: a disclosed trade
has both a `transaction_date` (when the member actually traded) and a
`disclosure_date` (when the public could first know about it, up to 45 days
later). The member's return is measured from the former; the return *you*
could have achieved by copying is measured from the latter. Conflating them
is the single easiest way to make a copy-trading dashboard lie, so the two
are kept distinct everywhere.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# A periodic transaction report is due within 45 days of the trade. Filers run
# late and an amendment can reach further back, so a year is the outer edge of
# honest lateness -- past it the date is wrong rather than tardy. It lives here
# because three layers need the same line and they must never disagree: the
# parser repairs up to it, performance refuses to price beyond it, and
# validation quarantines what is left.
MAX_PLAUSIBLE_LAG_DAYS = 365


def utc_now() -> datetime:
    return datetime.now(UTC)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(UTC)


class Chamber(StrEnum):
    HOUSE = "house"
    SENATE = "senate"


class TransactionType(StrEnum):
    PURCHASE = "purchase"
    SALE = "sale"
    PARTIAL_SALE = "partial_sale"
    EXCHANGE = "exchange"
    UNKNOWN = "unknown"

    @property
    def is_buy(self) -> bool:
        return self is TransactionType.PURCHASE

    @property
    def is_sell(self) -> bool:
        return self in (TransactionType.SALE, TransactionType.PARTIAL_SALE)


class Owner(StrEnum):
    SELF = "self"
    SPOUSE = "spouse"
    CHILD = "child"
    JOINT = "joint"
    UNKNOWN = "unknown"


class Stance(StrEnum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class AssetClass(StrEnum):
    STOCK = "stock"
    OPTION = "option"
    FUND = "fund"
    BOND = "bond"
    CRYPTO = "crypto"
    OTHER = "other"


class DomainRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = "2"
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    occurred_at: datetime = Field(default_factory=utc_now)
    correlation_id: str = Field(default_factory=lambda: str(uuid4()))
    causation_id: str | None = None

    @field_validator("occurred_at")
    @classmethod
    def validate_utc(cls, value: datetime) -> datetime:
        return _utc(value)


class Politician(DomainRecord):
    """A member of Congress whose disclosures are tracked.

    `politician_id` is a stable slug derived from chamber plus name, because
    the House filing index carries no bioguide id -- `bioguide_id` is filled
    in opportunistically when a name matches the congress-legislators roster.
    """

    politician_id: str
    full_name: str
    first_name: str = ""
    last_name: str = ""
    chamber: Chamber
    state: str = ""
    district: str | None = None
    party: str | None = None
    bioguide_id: str | None = None
    photo_url: str | None = None

    @property
    def display_state(self) -> str:
        if self.district:
            return f"{self.state}-{self.district}"
        return self.state


class Filing(DomainRecord):
    """One periodic transaction report, before or after its lines are parsed."""

    filing_id: str
    politician_id: str
    chamber: Chamber
    doc_id: str
    year: int
    filing_type: str = "P"
    filed_at: date
    source_url: str
    parsed: bool = False
    parse_error: str | None = None
    transaction_count: int = Field(default=0, ge=0)


class DisclosedTrade(DomainRecord):
    """One transaction line inside a filing.

    `amount_low`/`amount_high` bound the disclosed range (Congress reports
    brackets, never exact sizes), so every dollar figure downstream is an
    estimate built from the bracket midpoint.
    """

    trade_id: str
    filing_id: str
    politician_id: str
    chamber: Chamber
    symbol: str | None = None
    asset_name: str
    asset_class: AssetClass = AssetClass.OTHER
    transaction_type: TransactionType
    owner: Owner = Owner.UNKNOWN
    transaction_date: date
    disclosure_date: date
    amount_low: float = Field(default=0, ge=0)
    amount_high: float = Field(default=0, ge=0)
    comment: str | None = None
    source_url: str = ""

    @model_validator(mode="after")
    def validate_amounts(self) -> DisclosedTrade:
        if self.amount_high and self.amount_high < self.amount_low:
            raise ValueError("amount_high cannot be below amount_low")
        if self.disclosure_date < self.transaction_date:
            raise ValueError("a trade cannot be disclosed before it happened")
        return self

    @property
    def amount_midpoint(self) -> float:
        if self.amount_high:
            return (self.amount_low + self.amount_high) / 2
        return self.amount_low

    @property
    def reporting_lag_days(self) -> int:
        return (self.disclosure_date - self.transaction_date).days

    @property
    def is_actionable(self) -> bool:
        """Whether this line can be priced, ranked and mirrored at all."""
        return bool(self.symbol) and self.asset_class in (AssetClass.STOCK, AssetClass.FUND)


def build_trade_id(
    filing_id: str,
    asset: str,
    transaction_type: str,
    transaction_date: date,
    owner: str,
    amount_low: float,
    ordinal: int,
) -> str:
    """Deterministic id so re-parsing a filing never duplicates its lines.

    The ordinal is part of the hash because a single filing legitimately
    repeats an otherwise identical line (the same asset sold partially twice
    on the same day); without it those collapse into one.
    """
    payload = "|".join(
        [
            filing_id,
            asset.strip().upper(),
            transaction_type,
            transaction_date.isoformat(),
            owner,
            f"{amount_low:.2f}",
            str(ordinal),
        ]
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


class TradePerformance(BaseModel):
    """How one disclosed trade has done, measured two ways.

    `member_return` runs from the transaction date -- what the member got.
    `achievable_return` runs from the disclosure date -- what a copier could
    have got, since that is the first day the trade was public. `excess` is
    the achievable return net of the benchmark over the same window, which is
    what member ranking is actually built on.
    """

    model_config = ConfigDict(frozen=True)

    trade_id: str
    politician_id: str
    symbol: str
    transaction_type: TransactionType
    transaction_date: date
    disclosure_date: date
    amount_midpoint: float
    horizon_days: int | None = None
    entry_price: float | None = None
    disclosure_price: float | None = None
    exit_price: float | None = None
    member_return: float | None = None
    achievable_return: float | None = None
    benchmark_return: float | None = None
    excess_return: float | None = None
    priced: bool = False
    note: str | None = None

    @property
    def result_return(self) -> float | None:
        """The achievable return, signed by intent rather than by the price.

        `achievable_return` is what the *stock* did from the filing date. For
        a sale that is backwards as a verdict: a member who sold before a 12%
        rise did not make 12%, they gave it up. Every score in this
        application is signed by intent, so the figure a page puts next to a
        sale has to be too — otherwise the colour and the sign tell a reader
        opposite things about the same trade.
        """
        if self.achievable_return is None:
            return None
        return (-1 if self.transaction_type.is_sell else 1) * self.achievable_return


class TradeSide(StrEnum):
    """Which half of a member's record a score covers.

    One score across both directions answers "is this person a good
    decision-maker". It cannot answer "should I copy their buying", and using
    it for that recommended a member whose entire record was sales against a
    backtest that only ever bought. Every score is now computed three times,
    and the caller has to say which question it is asking.
    """

    ALL = "all"
    BUY = "buy"
    SELL = "sell"


class MemberScore(DomainRecord):
    """A member's track record over one horizon and one side of the book.

    `raw_excess_return` is the plain average; `score` applies Bayesian
    shrinkage so a member with three lucky trades does not outrank one with
    forty consistent ones. `model_probability` is filled in only when the
    optional pooled LightGBM edge model has been trained.
    """

    politician_id: str
    horizon_days: int
    side: TradeSide = TradeSide.ALL
    trades_scored: int = Field(default=0, ge=0)
    win_rate: float = Field(default=0, ge=0, le=1)
    raw_excess_return: float = 0.0
    median_excess_return: float = 0.0
    score: float = 0.0
    total_disclosed_dollars: float = Field(default=0, ge=0)
    average_reporting_lag_days: float = Field(default=0, ge=0)
    best_symbol: str | None = None
    best_symbol_excess: float | None = None
    # Which way they traded it. A "best call" of +36% that was a *sale* reads
    # as a pick that rose 36% unless the side travels with it.
    best_symbol_side: TradeSide | None = None
    # How many of their actionable trades could not be priced, and are
    # therefore in none of the numbers above.
    trades_unpriced: int = Field(default=0, ge=0)
    model_probability: float | None = None
    rankable: bool = False
    computed_at: datetime = Field(default_factory=utc_now)


class PoliticalMention(DomainRecord):
    """A tracked figure publicly said something touching a ticker.

    Strictly weaker evidence than a disclosed trade: it comes from headline
    text matching, so `confidence` and `matched_terms` are carried alongside
    so the dashboard can show *why* a ticker was attached.
    """

    mention_id: str
    figure: str
    figure_slug: str
    symbol: str
    company_name: str | None = None
    stance: Stance = Stance.NEUTRAL
    headline: str
    url: str
    source: str = ""
    published_at: datetime
    confidence: float = Field(default=0.0, ge=0, le=1)
    matched_terms: list[str] = Field(default_factory=list)

    @field_validator("published_at")
    @classmethod
    def validate_published_at(cls, value: datetime) -> datetime:
        return _utc(value)


class Watchlist(DomainRecord):
    """A standing instruction to mirror one or more members' trades.

    Mirroring is notional only: it produces `MirrorFill` records priced from
    historical closes so the dashboard can show what following this member
    would have done to a portfolio. Nothing is ever sent to a broker.
    """

    watchlist_id: str
    name: str
    politician_ids: list[str] = Field(default_factory=list)
    allocation_mode: str = "fixed_dollars"
    allocation_value: float = Field(default=1000.0, gt=0)
    mirror_sells: bool = True
    min_amount_low: float = Field(default=0, ge=0)
    starting_equity: float = Field(default=10_000.0, gt=0)
    active: bool = True
    created_at: datetime = Field(default_factory=utc_now)


class MirrorFill(DomainRecord):
    """A notional fill generated by a watchlist copying a disclosed trade.

    `priced_on` is the disclosure date, not the member's transaction date --
    see the module docstring. `price_basis` records which it was so a fill
    can never silently claim an unobtainable entry.
    """

    fill_id: str
    watchlist_id: str
    trade_id: str
    politician_id: str
    symbol: str
    side: TransactionType
    shares: float = Field(gt=0)
    price: float = Field(gt=0)
    notional: float = Field(gt=0)
    priced_on: date
    price_basis: str = "disclosure_date"
    note: str | None = None


class MirrorPosition(BaseModel):
    """Current notional holding of one symbol inside one watchlist."""

    model_config = ConfigDict(frozen=True)

    watchlist_id: str
    symbol: str
    shares: float
    average_cost: float
    cost_basis: float
    last_price: float | None = None
    market_value: float | None = None
    unrealized_pnl: float | None = None
    unrealized_return: float | None = None
    realized_pnl: float = 0.0
    opened_on: date | None = None
    fill_count: int = 0


class MirrorPortfolio(BaseModel):
    """Aggregate state of one watchlist's notional portfolio."""

    model_config = ConfigDict(frozen=True)

    watchlist_id: str
    name: str
    starting_equity: float
    cash: float
    positions: list[MirrorPosition] = Field(default_factory=list)
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    market_value: float = 0.0
    equity: float = 0.0
    total_return: float = 0.0
    fill_count: int = 0
    unpriced_trades: int = 0
    as_of: datetime = Field(default_factory=utc_now)


class RecommendationBuyer(BaseModel):
    """One member's participation in a recommended ticker, either direction.

    A card that says "6 buyers" and then quotes a single date is hiding the
    only detail that matters: they each traded on different days, disclosed
    on different days, and are therefore up by different amounts. Each member
    carries their own two returns -- from their trade, and from the day their
    filing became public.
    """

    model_config = ConfigDict(frozen=True)

    politician_id: str
    name: str
    chamber: str = ""
    party: str | None = None
    transaction_date: date
    disclosure_date: date
    lag_days: int = 0
    amount: float = 0.0
    amount_low: float = 0.0
    amount_high: float = 0.0
    trade_count: int = 1
    # `move_*` is what the *price* did. `result_*` is what that was worth to
    # the member, signed by intent: a sale followed by a rise is a bad exit
    # and reads negative, exactly as the member's score does. Both are kept,
    # because "the stock went up" and "the trade went badly" are different
    # facts and a sell card has to be able to say both.
    price_at_trade: float | None = None
    price_at_disclosure: float | None = None
    move_since_trade: float | None = None
    move_since_disclosure: float | None = None
    result_since_trade: float | None = None
    result_since_disclosure: float | None = None
    score: float | None = None
    rankable: bool = False
    rank: int | None = None
    # Committee seats this member holds whose remit covers this company.
    committees: list[str] = Field(default_factory=list)


class Recommendation(BaseModel):
    """A ranked idea, with the evidence that produced it.

    `reasons` is the point of the whole application: never surface a ticker
    without saying which members traded it, when it became public, and what
    their record is.

    `action` says which direction the card is about. Both are ranked: cluster
    selling ahead of a fall is the same evidence read the other way, and it
    measures stronger than cluster buying.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str
    company_name: str | None = None
    # SIC description from the SEC registry, which is what a committee's
    # jurisdiction is matched against.
    industry: str | None = None
    action: str = "buy"
    conviction: float = Field(default=0, ge=0)
    buyer_count: int = 0
    seller_count: int = 0
    total_disclosed_dollars: float = 0.0
    latest_disclosure_date: date | None = None
    earliest_transaction_date: date | None = None
    top_member: str | None = None
    top_member_score: float | None = None
    best_rank: int | None = None
    # trusted | credible | unproven | unranked -- see `recommend._tier`.
    tier: str = "unranked"
    # Price move since the member traded, versus since it became public. The
    # gap between the two is the reporting lag expressed in return.
    move_since_trade: float | None = None
    move_since_disclosure: float | None = None
    political_mentions: int = 0
    # How many members on this side sit on a committee whose remit covers the
    # company. Counted on both sides and scored on neither by default; see
    # `recommend.WEIGHT_COMMITTEE_SELL` for what the measurement says.
    committee_conflicts: int = 0
    # Only one of these is populated, according to `action`. The other side's
    # participants are counted in `buyer_count`/`seller_count` and named in
    # the counter-signal reason.
    buyers: list[RecommendationBuyer] = Field(default_factory=list)
    sellers: list[RecommendationBuyer] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    supporting_trade_ids: list[str] = Field(default_factory=list)
    as_of: datetime = Field(default_factory=utc_now)


class SyncReport(BaseModel):
    """What one disclosure sync actually managed to do, per source."""

    model_config = ConfigDict(frozen=True)

    source: str
    ok: bool
    filings_seen: int = 0
    filings_parsed: int = 0
    filings_failed: int = 0
    trades_added: int = 0
    politicians_added: int = 0
    error: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime | None = None
