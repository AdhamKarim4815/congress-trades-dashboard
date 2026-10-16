"""Finance-tuned sentiment scoring with no dependencies and no API.

This is a lexicon scorer, not a neural model, and that is a deliberate
trade-off. It costs nothing, needs no key, downloads no weights, runs in
microseconds and — most importantly — can explain itself: every score comes
with the exact terms that produced it, which matters a great deal when the
output is nudging a stock up a "next to boom" board.

The word lists lean on the insight behind Loughran-McDonald's finance
dictionary: general-purpose sentiment lexicons badly misread financial text
("liability", "cost", "crude" are not negative in a market context, while
"dilution", "going concern" and "restatement" very much are). Weights are
hand-set in three tiers so that "plunges" outweighs "slips".

Scores land in [-1, 1] and are length-normalised, so a short headline and a
long summary containing the same signal score comparably.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

# Strong signals: unambiguous, large-magnitude market language.
STRONG_POSITIVE = {
    "surge", "surges", "surged", "soar", "soars", "soared", "skyrocket", "skyrockets",
    "skyrocketed", "record", "records", "breakthrough", "blowout", "smashes", "smashed",
    "outperform", "outperforms", "outperformed", "upgrade", "upgrades", "upgraded",
    "bullish", "rally", "rallies", "rallied", "boom", "booming", "windfall", "landmark",
    "approval", "approved", "wins", "won", "acquisition", "acquires", "buyout", "takeover",
}
STRONG_NEGATIVE = {
    "plunge", "plunges", "plunged", "crash", "crashes", "crashed", "collapse", "collapses",
    "collapsed", "plummet", "plummets", "plummeted", "bankruptcy", "bankrupt", "fraud",
    "fraudulent", "downgrade", "downgrades", "downgraded", "bearish", "probe", "probes",
    "investigation", "investigating", "subpoena", "lawsuit", "sued", "recall", "recalls",
    "delisting", "delisted", "halt", "halted", "restatement", "dilution", "default",
    "defaults", "layoffs", "slump", "slumps", "slumped", "tumble", "tumbles", "tumbled",
}
# Moderate signals: directional but ordinary.
POSITIVE = {
    "gain", "gains", "gained", "rise", "rises", "rose", "climb", "climbs", "climbed",
    "jump", "jumps", "jumped", "beat", "beats", "profit", "profits", "profitable",
    "growth", "grow", "grows", "grew", "strong", "stronger", "boost", "boosts", "boosted",
    "expand", "expands", "expansion", "raise", "raises", "raised", "exceed", "exceeds",
    "exceeded", "momentum", "optimistic", "optimism", "confidence", "recovery", "rebound",
    "rebounds", "rebounded", "buy", "buys", "accumulate", "positive", "improve", "improves",
    "improved", "advance", "advances", "advanced", "higher", "tops", "surpass",
    "surpasses", "demand", "partnership", "contract", "awarded", "launch", "launches",
    "dividend", "buyback", "buybacks", "shine", "shines", "shining", "upbeat", "robust",
    "accelerate", "accelerates", "accelerating", "outpace", "outpaces", "benefit",
    "benefits", "benefiting", "expanding", "soaring", "leads", "leading",
}
NEGATIVE = {
    "fall", "falls", "fell", "drop", "drops", "dropped", "decline", "declines", "declined",
    "loss", "losses", "lose", "loses", "lost", "miss", "misses", "missed", "weak", "weaker",
    "weakness", "slash", "slashes", "slashed", "sink", "sinks", "sank",
    "slip", "slips", "slipped", "concern", "concerns", "worry", "worries", "risk", "risks",
    "warn", "warns", "warned", "warning", "shortfall", "delay", "delays", "delayed",
    "selloff", "downturn", "fine", "fined", "penalty", "negative",
    "lower", "shrink", "shrinks", "struggle", "struggles", "struggling",
    "pressure", "headwind", "headwinds", "disappoint", "disappoints", "disappointing",
    "stalls", "stalled", "halts", "cancels", "cancelled", "scraps", "scrapped",
}
# Topic markers, not sentiment. Loughran-McDonald's core lesson is that a
# general lexicon misreads financial text; the same applies to policy nouns.
# A tariff, a ban or a rate decision is bullish for some names and bearish
# for others, and scoring them as inherently negative made every Trump
# trade-policy headline read bearish regardless of who it helped.
POLICY_NEUTRAL = {
    "tariff", "tariffs", "sanction", "sanctions", "ban", "bans", "banned", "inflation",
    "recession", "regulation", "regulations", "policy", "cut", "cuts", "cutting",
    "sell", "sells", "buy", "up", "down", "top", "rate", "rates",
}
NEGATORS = {"not", "no", "never", "without", "denies", "denied", "avoids", "avoided", "halts"}
INTENSIFIERS = {
    "very": 1.5, "sharply": 1.6, "massively": 1.8, "significantly": 1.4, "surprisingly": 1.3,
    "record": 1.5, "deeply": 1.5, "steeply": 1.6, "slightly": 0.5, "modestly": 0.6,
    "marginally": 0.5,
}

STRONG_WEIGHT = 2.0
BASE_WEIGHT = 1.0
# Divides the raw score; tuned so a typical headline with one strong term
# lands near 0.5 rather than pinning at 1.0.
NORMALISER = 3.0

TOKEN_RE = re.compile(r"[a-z']+")
# Negation scope stops at clause boundaries. Without this, "not strong" in
# "is not strong; profit growth slipped" flipped `profit` and `growth` too,
# turning a correctly-bearish headline into a doubly-wrong one.
CLAUSE_RE = re.compile(r"[.;:!?,]|\band\b|\bbut\b|\bwhile\b|\bafter\b|\bas\b")


@dataclass(frozen=True)
class SentimentResult:
    score: float
    label: str
    positive_terms: list[str] = field(default_factory=list)
    negative_terms: list[str] = field(default_factory=list)
    token_count: int = 0

    @property
    def is_bullish(self) -> bool:
        return self.label == "bullish"

    @property
    def is_bearish(self) -> bool:
        return self.label == "bearish"

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "label": self.label,
            "positive_terms": self.positive_terms,
            "negative_terms": self.negative_terms,
            "token_count": self.token_count,
        }


def _stems(token: str) -> list[str]:
    """Cheap morphological variants, so the lists need not spell every form.

    Enumerating rise/rises/rose/rising by hand guarantees gaps ("rebounding"
    and "topped" both slipped through). Trying a few suffix strips covers the
    regular cases; irregular forms still live in the lists explicitly.
    """
    variants = [token]
    for suffix, replacements in (
        ("ing", ("", "e")),
        ("ed", ("", "e")),
        ("es", ("", "e")),
        ("s", ("",)),
    ):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            stem = token[: -len(suffix)]
            variants.extend(stem + replacement for replacement in replacements)
            # Doubled consonant: "topped" -> "top", "slipped" -> "slip".
            if len(stem) > 2 and stem[-1] == stem[-2]:
                variants.append(stem[:-1])
            break
    return variants


def _classify(token: str) -> tuple[float, int] | None:
    """Weight and polarity for a token, or None when it carries no sentiment."""
    for candidate in _stems(token):
        if candidate in POLICY_NEUTRAL:
            return None
        if candidate in STRONG_POSITIVE:
            return STRONG_WEIGHT, 1
        if candidate in STRONG_NEGATIVE:
            return STRONG_WEIGHT, -1
        if candidate in POSITIVE:
            return BASE_WEIGHT, 1
        if candidate in NEGATIVE:
            return BASE_WEIGHT, -1
    return None


def _label(score: float) -> str:
    if score >= 0.15:
        return "bullish"
    if score <= -0.15:
        return "bearish"
    return "neutral"


def score_text(text: str) -> SentimentResult:
    """Score one piece of text, reporting which terms drove the result.

    A negator or intensifier in the three tokens before a sentiment word
    modifies it: "not strong" flips, "sharply lower" amplifies. Three tokens
    is enough for headline grammar without letting a negation leak across a
    whole sentence.
    """
    lowered = (text or "").lower()
    clauses = [clause for clause in CLAUSE_RE.split(lowered) if clause.strip()]
    if not clauses:
        return SentimentResult(score=0.0, label="neutral")

    total = 0.0
    token_count = 0
    positive_terms: list[str] = []
    negative_terms: list[str] = []

    for clause in clauses:
        tokens = TOKEN_RE.findall(clause)
        token_count += len(tokens)
        for index, token in enumerate(tokens):
            classified = _classify(token)
            if classified is None:
                continue
            weight, polarity = classified

            # Negation only ever precedes its target ("not strong"), and only
            # within this clause.
            for modifier in tokens[max(0, index - 3) : index]:
                if modifier in NEGATORS:
                    polarity = -polarity
                elif modifier in INTENSIFIERS:
                    weight *= INTENSIFIERS[modifier]
            # Adverbs of degree usually *follow* the verb in financial prose
            # ("fell sharply", "rose modestly"), so look one token ahead too.
            for modifier in tokens[index + 1 : index + 2]:
                if modifier in INTENSIFIERS:
                    weight *= INTENSIFIERS[modifier]

            total += weight * polarity
            (positive_terms if polarity > 0 else negative_terms).append(token)

    if total == 0:
        return SentimentResult(score=0.0, label="neutral", token_count=token_count)

    # Length normalisation keeps a long article from out-shouting a headline
    # purely by containing more words.
    scale = NORMALISER * math.sqrt(max(token_count, 1) / 12)
    score = max(-1.0, min(1.0, total / scale))
    return SentimentResult(
        score=score,
        label=_label(score),
        positive_terms=sorted(set(positive_terms))[:8],
        negative_terms=sorted(set(negative_terms))[:8],
        token_count=token_count,
    )


def aggregate(results: list[SentimentResult]) -> dict[str, Any]:
    """Roll several scored items into one view of a subject's news tone."""
    if not results:
        return {
            "count": 0,
            "mean_score": 0.0,
            "label": "neutral",
            "bullish": 0,
            "bearish": 0,
            "neutral": 0,
        }
    scores = [result.score for result in results]
    mean = sum(scores) / len(scores)
    return {
        "count": len(results),
        "mean_score": mean,
        "label": _label(mean),
        "bullish": sum(1 for result in results if result.is_bullish),
        "bearish": sum(1 for result in results if result.is_bearish),
        "neutral": sum(1 for result in results if result.label == "neutral"),
        "strongest": max(results, key=lambda result: abs(result.score)).as_dict(),
    }
