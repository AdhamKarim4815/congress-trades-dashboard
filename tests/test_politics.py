"""Detecting which companies a political figure's coverage actually names."""

from __future__ import annotations

from trade_bot.politics import build_company_index, extract_symbols

INDEX = {
    "TSLA": "Tesla",
    "AAPL": "Apple",
    "BA": "Boeing",
    "X": "United States Steel",
    "ALL": "Allstate",
}


def test_cashtag_is_high_confidence():
    results = extract_symbols("$TSLA is up", INDEX)
    found = {symbol: confidence for symbol, confidence, _ in results}
    assert found["TSLA"] >= 0.9


def test_company_name_match_is_high_confidence():
    results = extract_symbols("Trump praises Tesla's new factory", INDEX)
    symbols = {symbol: confidence for symbol, confidence, _ in results}
    assert symbols["TSLA"] >= 0.8


def test_bare_ticker_is_low_confidence():
    results = extract_symbols("Analysts rate BA a buy", INDEX)
    confidence = {symbol: value for symbol, value, _ in results}
    # "Boeing" was never written, only the ticker, so trust it far less.
    assert confidence["BA"] < 0.5


def test_english_words_that_are_also_tickers_are_not_matched():
    """"ALL", "IT", "ON" and friends are real tickers and ordinary words.

    Matching them bare would attach a company to nearly every headline.
    """
    results = extract_symbols("Trump said ALL of IT was ON the table", INDEX)
    assert all(symbol != "ALL" for symbol, _, _ in results)


def test_single_letter_ticker_is_not_matched_from_prose():
    results = extract_symbols("The X factor in the deal", INDEX)
    assert all(symbol != "X" for symbol, _, _ in results)


def test_strongest_evidence_wins_when_a_headline_says_both():
    results = extract_symbols("Tesla ($TSLA) surges", INDEX)
    tesla = [item for item in results if item[0] == "TSLA"]
    assert len(tesla) == 1, "one mention per symbol, not one per matching form"
    assert tesla[0][1] >= 0.9
    assert len(tesla[0][2]) >= 2  # both forms recorded as evidence


def test_corporate_suffixes_are_ignored_when_matching():
    index = {"AAPL": "Apple Inc. - Common Stock"}
    results = extract_symbols("Apple unveils a new product", index)
    assert results and results[0][0] == "AAPL"


def test_company_index_merges_filings_with_the_baseline(store):
    index = build_company_index(store)
    # Baseline household names are present even with an empty database.
    assert "TSLA" in index and "AAPL" in index
