"""The finance lexicon, including the two mistakes it was built to avoid."""

from __future__ import annotations

from trade_bot.sentiment import aggregate, score_text


def test_clear_bullish_and_bearish_headlines():
    assert score_text("Apple announces record buyback, shares surge").is_bullish
    assert score_text("Nvidia shares plunge as chip probe widens").is_bearish


def test_neutral_text_scores_zero():
    result = score_text("Board meeting scheduled for Tuesday")
    assert result.label == "neutral"
    assert result.score == 0.0


def test_negation_flips_the_term_it_governs():
    plain = score_text("earnings were strong")
    negated = score_text("earnings were not strong")
    assert plain.score > 0 > negated.score


def test_negation_does_not_leak_across_a_clause_boundary():
    """"not strong; profit growth" must not mark profit and growth negative.

    An earlier version used a flat three-token lookback with no clause
    awareness and flipped every following term.
    """
    result = score_text("The outlook is not strong; profit and growth continue")
    assert "profit" in result.positive_terms
    assert "growth" in result.positive_terms
    assert "profit" not in result.negative_terms


def test_policy_nouns_carry_no_sentiment_of_their_own():
    """Tariffs help some companies and hurt others.

    Treating them as inherently negative made every trade-policy headline
    read bearish regardless of who it benefited.
    """
    assert score_text("Trump extends China tariffs").label == "neutral"
    assert score_text("Fed cuts rates as inflation cools").label == "neutral"
    assert score_text("Solar stocks shine after tariffs extended").is_bullish


def test_intensifiers_and_diminishers_scale_the_signal():
    assert abs(score_text("shares fell sharply").score) > abs(
        score_text("shares fell slightly").score
    )


def test_morphological_variants_are_recognised():
    for text in ("shares rebounding", "stock climbing", "profits topped estimates"):
        assert score_text(text).score > 0, text


def test_scores_stay_within_bounds_and_report_their_terms():
    result = score_text("surge soar record breakthrough rally boom smashes wins")
    assert -1.0 <= result.score <= 1.0
    assert result.positive_terms and not result.negative_terms


def test_length_normalisation_keeps_a_long_text_comparable():
    short = score_text("shares surge")
    padded = score_text("shares surge " + "the market opened on time " * 8)
    assert padded.score < short.score
    assert padded.score > 0


def test_aggregate_summarises_a_set_of_scores():
    results = [
        score_text("shares surge on record profit"),
        score_text("shares plunge on fraud probe"),
        score_text("meeting scheduled"),
    ]
    summary = aggregate(results)
    assert summary["count"] == 3
    assert summary["bullish"] == 1 and summary["bearish"] == 1 and summary["neutral"] == 1
    assert "strongest" in summary


def test_aggregate_of_nothing_is_neutral():
    assert aggregate([])["label"] == "neutral"


def test_empty_input_is_handled():
    assert score_text("").label == "neutral"
    assert score_text(None).label == "neutral"  # type: ignore[arg-type]
