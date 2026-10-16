"""Which members are actually worth following.

Ranking is statistical first. For each member and each horizon, their
disclosed trades are priced (see `trade_bot.performance`) and reduced to an
average excess return over the benchmark -- then shrunk toward zero by
sample size:

    score = mean_excess * n / (n + k)

That shrinkage is the whole point. Congressional disclosure is a small-sample
problem: plenty of members have three or four reported trades, and without
shrinkage a member who bought one lucky biotech outranks one with forty
consistently good decisions. `k` (`ranking.shrinkage_prior_trades`) sets how
much evidence it takes to be believed.

On top of that sits an optional pooled model. Rather than fitting one model
per member -- which the data cannot support -- it trains a single LightGBM
classifier across *every* member's trades to predict whether a disclosure
will beat the benchmark, using features known at disclosure time. A member's
model probability is then the average edge the model sees in their recent
trades. The member's own prior record is included as a feature, computed on
an expanding window so a trade is never scored using its own outcome.

The model is optional in the literal sense: without the `ml` extra installed
everything here still works and the dashboard just shows statistical scores.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from trade_bot.activity import JobReporter, NullReporter
from trade_bot.config import Settings
from trade_bot.database import Store
from trade_bot.domain import (
    MAX_PLAUSIBLE_LAG_DAYS,
    MemberScore,
    TradePerformance,
    TradeSide,
    TransactionType,
)
from trade_bot.performance import evaluate_trades
from trade_bot.prices import PriceRepository

OWNER_WEIGHTS = {"self": 1.0, "joint": 0.9, "spouse": 0.7, "child": 0.5, "unknown": 0.8}


class MissingMLDependencyError(RuntimeError):
    """The optional `ml` extra is required: `uv sync --extra ml`."""


def shrink(mean_excess: float, sample_size: int, prior: float) -> float:
    """Pull an average toward zero in proportion to how little evidence backs it."""
    if sample_size <= 0:
        return 0.0
    if prior <= 0:
        return mean_excess
    return mean_excess * (sample_size / (sample_size + prior))


def compute_member_scores(
    settings: Settings,
    store: Store,
    repository: PriceRepository,
    reporter: JobReporter | None = None,
    today: date | None = None,
) -> list[MemberScore]:
    """Score every member who has disclosed a priceable trade in the window."""
    reporter = reporter or NullReporter()
    today = today or date.today()
    cutoff = today - timedelta(days=settings.ranking.lookback_days)
    benchmark = settings.prices.benchmark_symbol

    trades = store.list_trades(actionable_only=True, limit=100_000)
    trades = [trade for trade in trades if trade["disclosure_date"] >= cutoff]
    reporter.step(f"scoring {len(trades)} disclosed trades", 0, len(trades))

    scores: list[MemberScore] = []
    for horizon in sorted({*settings.ranking.horizons_days, settings.ranking.primary_horizon_days}):
        performances = evaluate_trades(
            trades, repository, benchmark, horizon_days=horizon, today=today, reporter=reporter
        )
        # Keyed on (member, side). A member has three records, not one: their
        # buying, their selling, and everything together. Collapsing them into
        # a single number ranked one member second on twenty-five sales and no
        # purchases at all, directly beneath a backtest that only ever buys.
        by_member: dict[tuple[str, TradeSide], list[TradePerformance]] = defaultdict(list)
        # Counted, not discarded. A trade nobody can price is absent from every
        # figure below, and the companies that cannot be priced are the ones
        # that were acquired or delisted -- so a record that hides the count
        # hides its own survivorship.
        unpriced: dict[tuple[str, TradeSide], int] = defaultdict(int)
        for performance in performances:
            side = _side_of(performance.transaction_type)
            keys = [(performance.politician_id, TradeSide.ALL)]
            if side is not None:
                keys.append((performance.politician_id, side))
            if not performance.priced or performance.excess_return is None:
                for key in keys:
                    unpriced[key] += 1
                continue
            for key in keys:
                by_member[key].append(performance)

        for (politician_id, side), items in by_member.items():
            scores.append(
                _score_for(
                    settings, politician_id, horizon, side, items,
                    unpriced[(politician_id, side)],
                )
            )
        covered = len({politician_id for politician_id, _ in by_member})
        reporter.note(f"scored {covered} members over a {horizon}-day horizon")

    return scores


def _side_of(transaction_type: TransactionType) -> TradeSide | None:
    """Which half of the book a trade belongs to, or neither.

    An exchange is a direction the scoring cannot read, so it counts toward a
    member's overall record and toward neither side of it.
    """
    if transaction_type.is_buy:
        return TradeSide.BUY
    if transaction_type.is_sell:
        return TradeSide.SELL
    return None


def _score_for(
    settings: Settings,
    politician_id: str,
    horizon: int,
    side: TradeSide,
    items: list[TradePerformance],
    unpriced: int = 0,
) -> MemberScore:
    excesses = [item.excess_return or 0.0 for item in items]
    sample = len(excesses)
    mean_excess = sum(excesses) / sample
    ordered = sorted(excesses)
    middle = sample // 2
    median = (
        ordered[middle] if sample % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    )
    best = max(items, key=lambda item: item.excess_return or 0.0)
    # A misread transaction date produces a lag in the tens of thousands of
    # days, and one of those drags a member's average from three weeks to
    # three years. The trade still scores -- its disclosure date is sound --
    # but it cannot say anything about how promptly this member files.
    lags = [
        (item.disclosure_date - item.transaction_date).days
        for item in items
        if 0 <= (item.disclosure_date - item.transaction_date).days <= MAX_PLAUSIBLE_LAG_DAYS
    ]
    return MemberScore(
        politician_id=politician_id,
        horizon_days=horizon,
        side=side,
        trades_scored=sample,
        win_rate=sum(1 for value in excesses if value > 0) / sample,
        raw_excess_return=mean_excess,
        median_excess_return=median,
        score=shrink(mean_excess, sample, settings.ranking.shrinkage_prior_trades),
        total_disclosed_dollars=sum(item.amount_midpoint for item in items),
        average_reporting_lag_days=sum(lags) / len(lags) if lags else 0.0,
        best_symbol=best.symbol,
        best_symbol_excess=best.excess_return,
        # Which way they traded their best call. A +36% "best call" that was a
        # sale reads as a pick that rose 36% unless the side travels with it.
        best_symbol_side=_side_of(best.transaction_type),
        trades_unpriced=unpriced,
        rankable=sample >= settings.ranking.min_trades_to_rank,
    )


# -- optional pooled edge model -------------------------------------------


FEATURE_COLUMNS = (
    "log_amount",
    "reporting_lag_days",
    "owner_weight",
    "is_buy",
    "is_senate",
    "prior_trade_count",
    "prior_mean_excess",
    "momentum_20d",
    "momentum_60d",
    "volatility_20d",
    "benchmark_momentum_20d",
)


@dataclass(frozen=True)
class EdgeModel:
    booster: Any
    horizon_days: int
    trained_at: date
    train_rows: int
    eval_rows: int
    eval_auc: float
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS

    def predict(self, rows: list[dict[str, float]]) -> list[float]:
        import pandas as pd

        if not rows:
            return []
        frame = pd.DataFrame(rows, columns=list(self.feature_columns))
        return [float(value) for value in self.booster.predict(frame)]

    def save(self, path: Path) -> None:
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> EdgeModel:
        import joblib

        model = joblib.load(path)
        if not isinstance(model, cls):
            raise ValueError(f"{path} does not hold a trade-edge model")
        return model


def model_path(settings: Settings) -> Path:
    return settings.project_root / "var" / "models" / "trade_edge.joblib"


def _price_features(
    repository: PriceRepository, symbol: str, as_of: date, benchmark: str
) -> dict[str, float]:
    """Momentum and volatility known at disclosure time, never after it."""
    frame = repository.load(symbol)
    features = {
        "momentum_20d": 0.0,
        "momentum_60d": 0.0,
        "volatility_20d": 0.0,
        "benchmark_momentum_20d": 0.0,
    }
    if frame.empty:
        return features
    import pandas as pd

    history = frame[frame["date"] <= pd.Timestamp(as_of)]
    if len(history) >= 21:
        closes = history["close"].to_numpy()
        features["momentum_20d"] = float(closes[-1] / closes[-21] - 1)
        if len(closes) >= 22:
            returns = closes[-21:] / closes[-22:-1] - 1
            features["volatility_20d"] = float(returns.std())
    if len(history) >= 61:
        closes = history["close"].to_numpy()
        features["momentum_60d"] = float(closes[-1] / closes[-61] - 1)

    benchmark_frame = repository.load(benchmark)
    if not benchmark_frame.empty:
        benchmark_history = benchmark_frame[benchmark_frame["date"] <= pd.Timestamp(as_of)]
        if len(benchmark_history) >= 21:
            closes = benchmark_history["close"].to_numpy()
            features["benchmark_momentum_20d"] = float(closes[-1] / closes[-21] - 1)
    return features


def build_training_frame(
    settings: Settings,
    store: Store,
    repository: PriceRepository,
    horizon_days: int,
    reporter: JobReporter | None = None,
    today: date | None = None,
) -> Any:
    """One row per resolved disclosure, with leak-free member history.

    Rows are built in disclosure order and each member's prior statistics are
    accumulated *as the walk passes*, so a trade's features only ever reflect
    trades disclosed strictly before it.
    """
    import pandas as pd

    reporter = reporter or NullReporter()
    today = today or date.today()
    benchmark = settings.prices.benchmark_symbol

    trades = store.list_trades(actionable_only=True, limit=100_000)
    # Only trades whose horizon has fully elapsed carry a real label.
    resolved = [
        trade
        for trade in trades
        if trade["disclosure_date"] + timedelta(days=horizon_days) <= today
    ]
    resolved.sort(key=lambda trade: trade["disclosure_date"])
    reporter.step(f"labelling {len(resolved)} resolved trades", 0, len(resolved))

    performances = {
        performance.trade_id: performance
        for performance in evaluate_trades(
            resolved, repository, benchmark, horizon_days=horizon_days, today=today
        )
    }

    history: dict[str, list[float]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    for index, trade in enumerate(resolved, start=1):
        if index % 250 == 0:
            reporter.step(f"building features {index}/{len(resolved)}", index, len(resolved))
        performance = performances.get(trade["trade_id"])
        if performance is None or not performance.priced or performance.excess_return is None:
            continue
        prior = history[trade["politician_id"]]
        transaction_type = TransactionType(trade["transaction_type"])
        features = _price_features(
            repository, trade["symbol"], trade["disclosure_date"], benchmark
        )
        rows.append(
            {
                "log_amount": math.log1p(max(trade.get("amount_midpoint", 0.0), 0.0)),
                "reporting_lag_days": float(trade.get("reporting_lag_days", 0)),
                "owner_weight": OWNER_WEIGHTS.get(trade.get("owner", "unknown"), 0.8),
                "is_buy": 1.0 if transaction_type.is_buy else 0.0,
                "is_senate": 1.0 if trade.get("chamber") == "senate" else 0.0,
                "prior_trade_count": float(len(prior)),
                "prior_mean_excess": float(sum(prior) / len(prior)) if prior else 0.0,
                **features,
                "label": 1 if (performance.excess_return or 0.0) > 0 else 0,
                "politician_id": trade["politician_id"],
                "disclosure_date": trade["disclosure_date"],
            }
        )
        prior.append(performance.excess_return or 0.0)

    return pd.DataFrame(rows)


def train_edge_model(
    settings: Settings,
    store: Store,
    repository: PriceRepository,
    reporter: JobReporter | None = None,
    horizon_days: int | None = None,
    train_fraction: float = 0.75,
    minimum_rows: int = 200,
    today: date | None = None,
) -> EdgeModel:
    try:
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score
    except ImportError as exc:
        raise MissingMLDependencyError(
            "training the edge model requires the `ml` extra: `uv sync --extra ml`"
        ) from exc

    reporter = reporter or NullReporter()
    horizon = horizon_days or settings.ranking.primary_horizon_days
    frame = build_training_frame(settings, store, repository, horizon, reporter, today)
    if len(frame) < minimum_rows:
        raise ValueError(
            f"only {len(frame)} resolved trades are available; need at least {minimum_rows}. "
            "Sync more filing years, or wait for open horizons to close."
        )
    if frame["label"].nunique() < 2:
        raise ValueError("training data contains only one outcome class")

    # Time-ordered split: the model is always evaluated on disclosures that
    # happened after everything it learned from.
    frame = frame.sort_values("disclosure_date").reset_index(drop=True)
    split = int(len(frame) * train_fraction)
    train, evaluation = frame.iloc[:split], frame.iloc[split:]
    if train["label"].nunique() < 2 or evaluation["label"].nunique() < 2:
        raise ValueError("the time-ordered split left a fold with a single outcome class")

    columns = list(FEATURE_COLUMNS)
    reporter.step(f"training on {len(train)} trades, evaluating on {len(evaluation)}")
    booster = lgb.train(
        {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "num_leaves": 15,
            "min_data_in_leaf": 20,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
        },
        lgb.Dataset(train[columns], label=train["label"]),
        num_boost_round=200,
    )
    predictions = booster.predict(evaluation[columns])
    auc = float(roc_auc_score(evaluation["label"], predictions))
    reporter.note(f"edge model AUC {auc:.3f} on {len(evaluation)} held-out trades")
    return EdgeModel(
        booster=booster,
        horizon_days=horizon,
        trained_at=date.today(),
        train_rows=len(train),
        eval_rows=len(evaluation),
        eval_auc=auc,
    )


def apply_edge_model(
    settings: Settings,
    store: Store,
    repository: PriceRepository,
    model: EdgeModel,
    scores: list[MemberScore],
    recent_trades: int = 25,
    today: date | None = None,
) -> list[MemberScore]:
    """Attach each member's average model-predicted edge to their scores."""
    today = today or date.today()
    benchmark = settings.prices.benchmark_symbol
    cutoff = today - timedelta(days=settings.ranking.lookback_days)

    trades = [
        trade
        for trade in store.list_trades(actionable_only=True, limit=100_000)
        if trade["disclosure_date"] >= cutoff
    ]
    trades.sort(key=lambda trade: trade["disclosure_date"])
    by_member: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trade in trades:
        by_member[trade["politician_id"]].append(trade)

    probabilities: dict[str, float] = {}
    for politician_id, member_trades in by_member.items():
        window = member_trades[-recent_trades:]
        rows = []
        prior_count = float(len(member_trades) - len(window))
        for trade in window:
            transaction_type = TransactionType(trade["transaction_type"])
            rows.append(
                {
                    "log_amount": math.log1p(max(trade.get("amount_midpoint", 0.0), 0.0)),
                    "reporting_lag_days": float(trade.get("reporting_lag_days", 0)),
                    "owner_weight": OWNER_WEIGHTS.get(trade.get("owner", "unknown"), 0.8),
                    "is_buy": 1.0 if transaction_type.is_buy else 0.0,
                    "is_senate": 1.0 if trade.get("chamber") == "senate" else 0.0,
                    "prior_trade_count": prior_count,
                    "prior_mean_excess": 0.0,
                    **_price_features(
                        repository, trade["symbol"], trade["disclosure_date"], benchmark
                    ),
                }
            )
        predicted = model.predict(rows)
        if predicted:
            probabilities[politician_id] = sum(predicted) / len(predicted)

    return [
        score.model_copy(update={"model_probability": probabilities.get(score.politician_id)})
        for score in scores
    ]


def rank_members(
    settings: Settings,
    store: Store,
    repository: PriceRepository,
    reporter: JobReporter | None = None,
    use_model: bool = True,
    today: date | None = None,
) -> dict[str, Any]:
    """Recompute and persist every member's score, model included if trained."""
    reporter = reporter or NullReporter()
    scores = compute_member_scores(settings, store, repository, reporter, today)

    model_note = "statistical only"
    if use_model:
        path = model_path(settings)
        if path.exists():
            try:
                model = EdgeModel.load(path)
                scores = apply_edge_model(settings, store, repository, model, scores, today=today)
                model_note = f"edge model applied (AUC {model.eval_auc:.3f})"
            except Exception as exc:
                reporter.note(f"edge model could not be applied: {exc}", level="warn")
                model_note = f"edge model unavailable: {exc}"

    store.replace_member_scores(scores)
    primary = settings.ranking.primary_horizon_days
    rankable = [
        score for score in scores if score.horizon_days == primary and score.rankable
    ]
    reporter.note(
        f"ranked {len(rankable)} members with at least "
        f"{settings.ranking.min_trades_to_rank} scored trades"
    )
    return {
        "scores_written": len(scores),
        "rankable_members": len(rankable),
        "horizons": sorted({score.horizon_days for score in scores}),
        "model": model_note,
    }
