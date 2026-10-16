"""The opportunity board: which tickers are worth a look, and why.

The board ranks **both directions**. Buying was ranked first because it is
the obvious thing to copy, but cluster selling ahead of a fall is the same
evidence read the other way, and the data says it is the stronger of the two:
sales beat the index 56% of the time over 90 days against 45% for purchases.
Discarding half the disclosures was never defensible; `seller_count` sat
computed and unused for exactly that long.

Conviction is a weighted blend of independent pieces of evidence, each
normalised to roughly 0..1 before weighting:

* **member quality** -- the best shrunk track record among the members on
  this side. The score is signed by intent, so a member who is good at
  selling scores well on a sell card without their buys flattering them.
* **cluster** -- how many distinct members acted, which is the strongest
  structural signal in this dataset because independent actors rarely
  coincide by chance
* **size** -- the disclosed dollar brackets behind them
* **recency** -- decayed over the disclosure window, since a disclosure from
  six weeks ago is largely priced in
* **news sentiment** -- recent coverage tone, read in the direction of the
  trade: negative coverage supports a sale as much as positive supports a buy
* **political attention** -- tracked figures naming the company
* **committee overlap** -- on sell cards only, and only because it was
  measured. See `WEIGHT_COMMITTEE_SELL`.

The output always carries `reasons`: plain sentences naming the members, the
dates and the record behind the number. A conviction score with no reasons
attached would be worse than useless here, because the entire premise is
that you can see who is trading and decide for yourself.

A word on what this is not: disclosure lags the trade by weeks by law, so
this ranks *newly public* information, not fresh trades. `days_since_traded`
is surfaced on every card so that gap is never hidden.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import UTC, date, datetime, timedelta
from typing import Any

from trade_bot.activity import JobReporter, NullReporter
from trade_bot.committees import ConflictIndex
from trade_bot.config import Settings
from trade_bot.database import Store
from trade_bot.domain import Recommendation, RecommendationBuyer, TransactionType
from trade_bot.news import company_names, sentiment_window
from trade_bot.politics import mention_summary
from trade_bot.prices import PriceRepository

# Weights sum to 1.0 before the optional political bonus is layered on.
#
# Member quality dominates because the walk-forward test says it should: rank
# 1-3 returned +3.43% mean excess at a 55% beat rate against +1.83% at 49% for
# everyone below. Clustering was weighted nearly as heavily as quality in the
# first cut, which floated ideas whose only merit was several low-ranked
# members agreeing -- and agreeing members who individually lose to the index
# do not stop losing to it by coinciding.
#
# Those figures move every time the backtest re-runs, and `/evidence` is the
# copy that cannot go stale. They are quoted here only to show which way the
# evidence pointed when these weights were chosen.
# Rebalanced once, on evidence, and it is worth being precise about what kind
# of change that was.
#
# The blended score was measured against outcomes for the first time and did
# not order them: sorted into quartiles over 678 past positions, the *lowest*
# quartile returned the most. Clustering was the obvious suspect -- it carried
# 22%, second only to the buyer's record, on the argument that independent
# buyers rarely coincide by chance -- and measured on its own it is noise:
# +1.9%, +3.0%, -1.3%, +1.5% as the number of buyers rises.
#
# So cluster weight is cut to a residual and the freed weight goes to the one
# component with evidence behind it. What was *not* done matters as much:
# nothing here was tuned to maximise the measured spread. Searching the weight
# space for the combination that orders 678 positions best would fit this
# window and nothing else, and would be a worse mistake than the one being
# corrected. This removes weight that the evidence does not support; it does
# not add weight that the evidence recommends, because it does not.
#
# The caveat that keeps this honest: the cluster figure comes from positions
# taken in top-10 members only, which is a restricted range. Agreement may
# still carry information further down the ranking, where nothing measures it.
#
# `/evidence` recomputes the ordering test every run, and the board draws the
# conviction bar faint whenever it fails. If this rebalance did not help, the
# page says so on its own.
WEIGHT_MEMBER_QUALITY = 0.62
WEIGHT_CLUSTER = 0.08
WEIGHT_SIZE = 0.06
WEIGHT_RECENCY = 0.12
WEIGHT_NEWS = 0.12

# Committee overlap, and why it is on one side of the board only.
#
# The premise was that a member trading a company their committee oversees is
# the strongest signal congressional data can produce. Measured over three
# years of disclosures at the 90-day horizon, that turned out to be true in
# one direction and *backwards* in the other:
#
#     purchases inside a member's remit   median -2.71% excess, beat 40%
#     purchases outside it                median -1.57% excess, beat 45%
#     sales inside a member's remit       median +3.19% excess, beat 63%
#     sales outside it                    median +1.75% excess, beat 56%
#
# So the overlap earns weight on sells and none on buys. Weighting the buy
# side for it would have made the board measurably worse while feeling like
# the cleverest thing in the application. The live numbers are recomputed
# every run by `committees.measure_committee_edge` and published on
# `/evidence`; if they invert, this weight should move with them.
WEIGHT_COMMITTEE_SELL = 0.14

# How far apart two transactions can be and still read as one cluster. A
# quarter is generous for "the same decision, arrived at independently";
# beyond it the members simply happened to file in the same window, which is
# an artefact of the reporting lag rather than agreement about anything.
CLUSTER_SPAN_DAYS = 90


def _tier(rank: int | None, settings: Any) -> str:
    """Plain label for how much the buyers' record is worth.

    The board sorts on a blended number, and a number alone cannot say "the
    only person buying this has a measured record that loses to the index".
    """
    if rank is None:
        return "unranked"
    if rank <= settings.trusted_rank:
        return "trusted"
    if rank <= settings.credible_rank:
        return "credible"
    return "unproven"


def _rank_quality(rank: int | None, trusted: int, credible: int) -> float:
    """How much a member's standing should count, by rank rather than score.

    Walk-forward testing finds the edge concentrated in the top band and
    *unordered* below it: ranks 1-3 returned +3.43% mean excess at a 55% beat
    rate, ranks 4-6 -0.11% at 45%, and ranks 7 and lower +1.83% at 49%. The
    top band is reliably the best one; the rest are not in order.

    This used to descend in four steps, which read as a gradient and was
    justified in its own docstring by a measurement that had a gradient. The
    corrected backtest -- ranked on purchases, which is what the strategy then
    copies -- does not. So the curve is now flat below the top: separating the
    top from the rest is a claim the data supports, and ordering the rest is
    not.
    """
    if rank is None:
        return 0.30
    if rank <= trusted:
        return 1.0
    if rank <= credible:
        return 0.70
    # Ranked, but past the point where rank predicts anything.
    return 0.45


def _cluster_strength(buyers: int) -> float:
    """Saturating curve: two buyers is a lot, six is not three times as much."""
    if buyers <= 0:
        return 0.0
    return min(1.0, math.log1p(buyers) / math.log(6))


def _size_strength(dollars: float) -> float:
    if dollars <= 0:
        return 0.0
    return min(1.0, math.log1p(dollars) / math.log1p(500_000))


def _recency_strength(days_ago: int, window: int) -> float:
    if window <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - (days_ago / window)))


def _news_strength(mean_sentiment: float) -> float:
    return max(0.0, min(1.0, 0.5 + mean_sentiment))


def structural_conviction(
    settings: Settings,
    rank: int | None,
    buyers: int,
    dollars: float,
    days_ago: int,
    window: int,
) -> float:
    """The part of conviction that can be rebuilt for a date in the past.

    Shared with `backtest.conviction_as_of`, and it has to be: the two had
    separate copies of this arithmetic for exactly one afternoon, and in that
    time removing a damper from the live board left the measurement scoring
    the old formula and reporting no change. A score and the test of that
    score cannot be two pieces of code.

    News tone and political mentions are deliberately not here. They are
    fetched in the present and cannot be known as of a past disclosure, so
    they belong to the live blend only.
    """
    return (
        WEIGHT_MEMBER_QUALITY
        * _rank_quality(
            rank,
            settings.recommendations.trusted_rank,
            settings.recommendations.credible_rank,
        )
        + WEIGHT_CLUSTER * _cluster_strength(buyers)
        + WEIGHT_SIZE * _size_strength(dollars)
        + WEIGHT_RECENCY * _recency_strength(days_ago, window)
    )


def damp_unproven(settings: Settings, score: float, rank: int | None) -> float:
    """Nothing presents as a strong idea on a record worse than the index."""
    if _tier(rank, settings.recommendations) in ("unproven", "unranked"):
        return score * 0.72
    return score


def build_recommendations(
    settings: Settings,
    store: Store,
    repository: PriceRepository,
    reporter: JobReporter | None = None,
    today: date | None = None,
) -> list[Recommendation]:
    """Rank recently-disclosed trading, both directions, into one board."""
    reporter = reporter or NullReporter()
    today = today or date.today()
    window = settings.recommendations.disclosure_window_days
    cutoff = today - timedelta(days=window)
    horizon = settings.ranking.primary_horizon_days

    trades = store.list_trades(actionable_only=True, disclosed_since=cutoff, limit=20_000)
    if not trades:
        return []

    # One ranking per side of the book. A sell card asking "who has the best
    # record here" means the best record *at selling*, and the overall score
    # cannot answer that: the member leading one sell card on +5.9% overall
    # has a sell record of -3.3%.
    scores: dict[str, dict[str, dict[str, Any]]] = {}
    ranks: dict[str, dict[str, int]] = {}
    for side in ("buy", "sell"):
        side_scores = store.list_member_scores(horizon_days=horizon, side=side)
        scores[side] = {score["politician_id"]: score for score in side_scores}
        ranked = sorted(
            (score for score in side_scores if score["rankable"]),
            key=lambda score: score["score"],
            reverse=True,
        )
        ranks[side] = {
            score["politician_id"]: position for position, score in enumerate(ranked, start=1)
        }
    members = {member["politician_id"]: member for member in store.list_politicians()}
    names = company_names(store)
    news = sentiment_window(store, "ticker", days=14)
    mentions = mention_summary(store, days=settings.politics.lookback_days)

    by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trade in trades:
        by_symbol[trade["symbol"]].append(trade)

    conflicts = ConflictIndex(settings, store)

    reporter.step(f"scoring {len(by_symbol)} tickers", 0, len(by_symbol))
    by_action: dict[str, list[Recommendation]] = {"buy": [], "sell": []}

    for index, (symbol, symbol_trades) in enumerate(sorted(by_symbol.items()), start=1):
        if index % 50 == 0:
            reporter.step(f"scoring ticker {index}/{len(by_symbol)}", index, len(by_symbol))

        buys = [
            trade
            for trade in symbol_trades
            if TransactionType(trade["transaction_type"]).is_buy
        ]
        sells = [
            trade
            for trade in symbol_trades
            if TransactionType(trade["transaction_type"]).is_sell
        ]
        for action, subject, opposing in (("buy", buys, sells), ("sell", sells, buys)):
            if not subject:
                continue
            card = _score_side(
                action=action,
                symbol=symbol,
                subject=subject,
                opposing=opposing,
                settings=settings,
                repository=repository,
                conflicts=conflicts,
                members=members,
                scores=scores[action],
                ranks=ranks[action],
                names=names,
                news=news.get(symbol, {}),
                mention_entry=mentions.get(symbol, {}),
                horizon=horizon,
                window=window,
                today=today,
            )
            by_action[action].append(card)

    # Capped per side rather than overall. A single cap would let a strong day
    # of buying push every sell card off the board, and the sell side is the
    # half that has been missing.
    limit = settings.recommendations.max_results
    limited: list[Recommendation] = []
    for action in ("buy", "sell"):
        by_action[action].sort(key=lambda item: item.conviction, reverse=True)
        limited.extend(by_action[action][:limit])
    limited.sort(key=lambda item: item.conviction, reverse=True)
    reporter.note(
        f"built {sum(1 for item in limited if item.action == 'buy')} buy and "
        f"{sum(1 for item in limited if item.action == 'sell')} sell ideas "
        f"from {len(by_symbol)} tickers"
    )
    return limited


def _score_side(
    *,
    action: str,
    symbol: str,
    subject: list[dict[str, Any]],
    opposing: list[dict[str, Any]],
    settings: Settings,
    repository: PriceRepository,
    conflicts: ConflictIndex,
    members: dict[str, dict[str, Any]],
    scores: dict[str, dict[str, Any]],
    ranks: dict[str, int],
    names: dict[str, str],
    news: dict[str, Any],
    mention_entry: dict[str, Any],
    horizon: int,
    window: int,
    today: date,
) -> Recommendation:
    """Build one card for one direction of trading in one ticker.

    Both sides run through here rather than through two near-identical
    functions, because the only real differences are which way the news tilt
    is read and whether committee overlap counts — and a copy of this logic
    would drift from the original within a release.
    """
    is_sell = action == "sell"
    actors = sorted({trade["politician_id"] for trade in subject})
    against = sorted({trade["politician_id"] for trade in opposing})
    dollars = sum(trade.get("amount_midpoint", 0.0) for trade in subject)
    latest_disclosure = max(trade["disclosure_date"] for trade in subject)
    earliest_transaction = min(trade["transaction_date"] for trade in subject)
    days_ago = (today - latest_disclosure).days

    actor_scores = [
        scores[politician_id]
        for politician_id in actors
        if politician_id in scores and scores[politician_id]["rankable"]
    ]
    best = max(actor_scores, key=lambda item: item["score"]) if actor_scores else None
    best_rank = ranks.get(best["politician_id"]) if best else None

    # Coverage tone read in the direction of the trade. Positive news is
    # evidence for a purchase and evidence against a sale, and treating it as
    # bullish on both sides would have made the sell board an inverted echo
    # of the buy board.
    tone = news.get("mean_score", 0.0)
    # Cluster credit goes to members who acted close together, not to everyone
    # who ever touched the name. Two people trading it two years apart happen
    # to appear in the same disclosure window because the reporting lag varies;
    # that is an artefact, and scoring it as agreement rewarded coincidence.
    latest_transaction = max(trade["transaction_date"] for trade in subject)
    clustered = {
        trade["politician_id"]
        for trade in subject
        if (latest_transaction - trade["transaction_date"]).days <= CLUSTER_SPAN_DAYS
    }
    conviction = structural_conviction(
        settings, best_rank, len(clustered), dollars, days_ago, window
    ) + WEIGHT_NEWS * _news_strength(-tone if is_sell else tone)

    conflicted = [
        politician_id for politician_id in actors if conflicts.overlaps(politician_id, symbol)
    ]
    if is_sell and conflicted:
        strength = max(conflicts.strength(politician_id, symbol) for politician_id in conflicted)
        conviction += WEIGHT_COMMITTEE_SELL * strength

    if mention_entry:
        bullish = mention_entry.get("bullish", 0)
        bearish = mention_entry.get("bearish", 0)
        count = max(mention_entry.get("count", 1), 1)
        tilt = max(0.0, ((bearish - bullish) if is_sell else (bullish - bearish)) / count)
        # A single headline is not worth the full political weight; it takes a
        # few before the tilt means anything.
        conviction += settings.recommendations.politics_weight * tilt * min(1.0, count / 3)

    # There used to be a damper here: conviction was multiplied down by up to
    # 45% when members went the other way, on the reasoning that opposition is
    # "real counter-evidence". Measured over 678 past positions, it is the
    # opposite of that -- purchases with nobody selling returned -0.25% and
    # beat the index 45% of the time, one seller against them +3.05% at 54%,
    # two or more +3.82% at 54%.
    #
    # It was also the single reason the blended score inverted: it pushed
    # top-ranked buyers with opposition into the bottom quartile, and the
    # bottom quartile was the one performing.
    #
    # So it is removed, not reversed. Turning a penalty into a bonus because
    # this window says so would be fitting three years of one market, and a
    # more confident mistake than the one being corrected. The counts are
    # still shown on the card -- a reader can see who went the other way and
    # decide -- they simply no longer move the number.
    #
    # The sell side is unmeasured here (every tested position is a purchase),
    # so it loses the damper too rather than keeping one on inertia.

    tier = _tier(best_rank, settings.recommendations)
    conviction = damp_unproven(settings, conviction, best_rank)

    disclosure_price, _ = repository.close_on_or_after(symbol, latest_disclosure)
    trade_price, _ = repository.close_on_or_after(symbol, earliest_transaction)
    last_price, _ = repository.latest_close(symbol)
    move_since_disclosure = (
        (last_price - disclosure_price) / disclosure_price
        if disclosure_price and last_price and disclosure_price > 0
        else None
    )
    move_since_trade = (
        (last_price - trade_price) / trade_price
        if trade_price and last_price and trade_price > 0
        else None
    )

    details = _member_details(
        subject, members, scores, ranks, repository, symbol, conflicts, is_sell
    )
    reasons = _build_reasons(
        action=action,
        symbol=symbol,
        trades=subject,
        actors=actors,
        against=against,
        members=members,
        best=best,
        latest_disclosure=latest_disclosure,
        news_entry=news,
        mention_entry=mention_entry,
        horizon=horizon,
        best_rank=best_rank,
        repository=repository,
        conflicts=conflicts,
        conflicted=conflicted,
    )

    return Recommendation(
        symbol=symbol,
        company_name=names.get(symbol),
        action=action,
        conviction=round(conviction, 4),
        buyer_count=len(against) if is_sell else len(actors),
        seller_count=len(actors) if is_sell else len(against),
        committee_conflicts=len(conflicted),
        industry=conflicts.industry(symbol) or None,
        total_disclosed_dollars=dollars,
        latest_disclosure_date=latest_disclosure,
        earliest_transaction_date=earliest_transaction,
        top_member=(members.get(best["politician_id"], {}).get("full_name") if best else None),
        top_member_score=best["score"] if best else None,
        political_mentions=mention_entry.get("count", 0),
        move_since_trade=move_since_trade,
        move_since_disclosure=move_since_disclosure,
        buyers=[] if is_sell else details,
        sellers=details if is_sell else [],
        best_rank=best_rank,
        tier=tier,
        reasons=reasons,
        supporting_trade_ids=[trade["trade_id"] for trade in subject][:25],
        as_of=datetime.now(UTC),
    )


def _member_details(
    trades: list[dict[str, Any]],
    members: dict[str, dict[str, Any]],
    scores: dict[str, dict[str, Any]],
    ranks: dict[str, int],
    repository: PriceRepository,
    symbol: str,
    conflicts: ConflictIndex,
    is_sell: bool,
) -> list[RecommendationBuyer]:
    """One row per member, each with their own dates and their own returns.

    A member who traded the same name twice is folded into a single row at
    their earliest transaction, with the amounts summed -- the question the
    card answers is "when did this person act and where is it now", not "how
    many separate tickets did they file". `trade_count` keeps that visible, so a
    member who filed one line per account is not silently flattened.
    """
    last_price, _ = repository.latest_close(symbol)
    # Signed by intent, exactly as the member's score is: for a seller a rising
    # price is the outcome they avoided having, so it counts against them.
    # Without this a bad exit rendered green on a sell card and read as a win.
    sign = -1.0 if is_sell else 1.0
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trade in trades:
        grouped[trade["politician_id"]].append(trade)

    details: list[RecommendationBuyer] = []
    for politician_id, member_trades in grouped.items():
        earliest = min(member_trades, key=lambda trade: trade["transaction_date"])
        member = members.get(politician_id, {})
        score = scores.get(politician_id)

        price_at_trade, _ = repository.close_on_or_after(
            symbol, earliest["transaction_date"]
        )
        price_at_disclosure, _ = repository.close_on_or_after(
            symbol, earliest["disclosure_date"]
        )
        move_trade = (
            (last_price - price_at_trade) / price_at_trade
            if price_at_trade and last_price and price_at_trade > 0
            else None
        )
        move_disclosure = (
            (last_price - price_at_disclosure) / price_at_disclosure
            if price_at_disclosure and last_price and price_at_disclosure > 0
            else None
        )
        details.append(
            RecommendationBuyer(
                politician_id=politician_id,
                name=member.get("full_name", politician_id),
                chamber=member.get("chamber", ""),
                party=member.get("party"),
                transaction_date=earliest["transaction_date"],
                disclosure_date=earliest["disclosure_date"],
                lag_days=earliest.get("reporting_lag_days", 0),
                amount=sum(trade.get("amount_midpoint", 0.0) for trade in member_trades),
                amount_low=sum(trade.get("amount_low", 0.0) for trade in member_trades),
                amount_high=sum(trade.get("amount_high", 0.0) for trade in member_trades),
                trade_count=len(member_trades),
                price_at_trade=price_at_trade,
                price_at_disclosure=price_at_disclosure,
                move_since_trade=move_trade,
                move_since_disclosure=move_disclosure,
                result_since_trade=None if move_trade is None else sign * move_trade,
                result_since_disclosure=(
                    None if move_disclosure is None else sign * move_disclosure
                ),
                score=score["score"] if score else None,
                rankable=bool(score and score["rankable"]),
                rank=ranks.get(politician_id),
                committees=[
                    seat["committee_name"]
                    for seat in conflicts.overlaps(politician_id, symbol)
                ],
            )
        )
    # Earliest in first: it reads as the order they got involved.
    details.sort(key=lambda buyer: buyer.transaction_date)
    return details


def _build_reasons(
    *,
    action: str,
    symbol: str,
    trades: list[dict[str, Any]],
    actors: list[str],
    against: list[str],
    members: dict[str, dict[str, Any]],
    best: dict[str, Any] | None,
    latest_disclosure: date,
    news_entry: dict[str, Any],
    mention_entry: dict[str, Any],
    horizon: int,
    best_rank: int | None,
    repository: PriceRepository,
    conflicts: ConflictIndex,
    conflicted: list[str],
) -> list[str]:
    is_sell = action == "sell"
    verb, noun = ("sold", "sellers") if is_sell else ("bought", "buyers")
    reasons: list[str] = []

    if best is not None:
        name = members.get(best["politician_id"], {}).get("full_name", best["politician_id"])
        standing = f", ranked #{best_rank} of all scored members" if best_rank else ""
        reasons.append(
            f"Best record among the {noun} is {name}{standing}: {best['score']:+.1%} shrunk "
            f"excess return over {horizon} days across {best['trades_scored']} scored "
            f"{'sales' if is_sell else 'purchases'} ({best['win_rate']:.0%} beat the "
            f"benchmark). That is their record at {'selling' if is_sell else 'buying'} "
            "specifically — a member can be good at one and poor at the other, and their "
            "combined score hides which."
        )
    else:
        reasons.append(
            f"None of the {noun} has enough scored {'sales' if is_sell else 'purchases'} "
            "yet to rank, so this rests on the disclosure itself rather than on a track "
            "record."
        )

    if len(actors) > 1:
        # What the per-member rows cannot say on their own: whether these were
        # independent decisions or one cluster, and how differently the lag
        # treated them.
        span = (
            max(trade["transaction_date"] for trade in trades)
            - min(trade["transaction_date"] for trade in trades)
        ).days
        lags = [trade.get("reporting_lag_days", 0) for trade in trades]
        # A cluster is members acting close together. Two people trading the
        # same name two years apart and filing in the same window is not a
        # cluster, and calling it "within 766 days of each other" made the
        # card claim agreement it had no evidence for.
        if span <= CLUSTER_SPAN_DAYS:
            together = (
                "on the same day"
                if span == 0
                else f"within {span} day{'s' if span != 1 else ''} of each other"
            )
            reasons.append(
                f"{len(actors)} members {verb} {together}, then filed {min(lags)} to "
                f"{max(lags)} days later — so they became public at different prices."
            )
        else:
            reasons.append(
                f"{len(actors)} members {verb}, but {span} days apart — they became "
                "public together only because their filings landed in the same window. "
                "Treat this as separate decisions, not agreement."
            )

    # The committee overlap, named rather than scored silently. It carries
    # weight on a sell card and none on a buy card, and the card says which.
    for politician_id in conflicted[:2]:
        name = members.get(politician_id, {}).get("full_name", politician_id)
        note = conflicts.describe(politician_id, symbol, name)
        if note:
            reasons.append(
                note
                + (
                    ""
                    if is_sell
                    else " It adds nothing to this score: measured over three years, "
                    "purchases inside a member's remit did worse than purchases outside it."
                )
            )

    price, _ = repository.close_on_or_after(symbol, latest_disclosure)
    latest, _ = repository.latest_close(symbol)
    if price and latest and price > 0:
        move = (latest - price) / price
        reasons.append(
            f"Since the disclosure, {symbol} has moved {move:+.1%} "
            f"(${price:,.2f} to ${latest:,.2f})."
        )
    # The per-buyer table on the card carries each member's own two returns;
    # an aggregate "since the earliest trade" line collapsed six different
    # entry dates into one and said less than the table does.

    if news_entry:
        tone = news_entry.get("mean_score", 0.0)
        descriptor = "positive" if tone > 0.15 else "negative" if tone < -0.15 else "mixed"
        supports = (tone < -0.15) if is_sell else (tone > 0.15)
        reasons.append(
            f"Recent coverage is {descriptor} ({news_entry.get('items', 0)} headlines, "
            f"mean sentiment {tone:+.2f})"
            + (f", which {'supports' if supports else 'runs against'} the {action} case."
               if abs(tone) > 0.15 else ".")
        )

    if mention_entry:
        figures = ", ".join(mention_entry.get("figures", []))
        reasons.append(
            f"{mention_entry.get('count', 0)} political mention"
            f"{'s' if mention_entry.get('count', 0) != 1 else ''} "
            f"({figures}): {mention_entry.get('bullish', 0)} positive, "
            f"{mention_entry.get('bearish', 0)} negative."
        )

    if against:
        other_names = ", ".join(
            members.get(politician_id, {}).get("full_name", politician_id)
            for politician_id in against[:3]
        )
        reasons.append(
            f"Counter-signal: {len(against)} member"
            f"{'s' if len(against) != 1 else ''} went the other way on {symbol} in this "
            f"window ({other_names})."
        )

    return reasons


def refresh_recommendations(
    settings: Settings,
    store: Store,
    repository: PriceRepository,
    reporter: JobReporter | None = None,
) -> dict[str, Any]:
    """Recompute the board and cache it for the dashboard to read."""
    reporter = reporter or NullReporter()
    recommendations = build_recommendations(settings, store, repository, reporter)
    store.set_state(
        "board.recommendations",
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "items": [item.model_dump(mode="json") for item in recommendations],
        },
    )
    return {"recommendations": len(recommendations)}


def load_recommendations(store: Store) -> tuple[list[dict[str, Any]], str | None]:
    state = store.get_state("board.recommendations")
    if not state:
        return [], None
    return state.get("items", []), state.get("generated_at")
