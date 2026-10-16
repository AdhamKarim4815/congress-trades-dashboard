"""Turning a written-out company name into a ticker.

Electronic filings carry the symbol in the asset name, so the parser reads it
straight off: ``KEYSIGHT TECHNOLOGIES INC. COMMON STOCK (KEYS)``. Paper filings
never do -- the PTR form instructs the filer to "Provide full name not ticker
symbol" -- so a scanned filing yields ``KEYSIGHT TECHNOLOGIES INC`` and nothing
to trade on.

Rather than pay a data vendor for a name-to-ticker map, this resolves against
one the application already has: every electronic filing ever parsed is a pair
of a company name and the symbol that filer wrote next to it. That corpus is
free, already local, and drawn from the same vocabulary the paper filings use,
which a general securities database is not.

The matching is deliberately strict. A wrong ticker here is worse than no
ticker: it invents a trade that a member never made and puts it on the board.
Anything short of a confident match resolves to None and the row is kept as an
unpriced disclosure rather than guessed at.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

# Legal and instrument suffixes carry no identifying information and differ
# between how a filer writes a name on paper and how the electronic form
# renders it ("DANAHER CORPORATION" vs "DANAHER CORPORATION COMMON STOCK").
_NOISE = re.compile(
    r"\b("
    r"COMMON\s+STOCK|COMMON\s+SHARES|ORDINARY\s+SHARES|CAPITAL\s+STOCK|"
    r"CMN|CMT|CNA|HYBRID|PERPETUAL|REIT|USD|"
    r"CLASS\s+[A-C]|CL\s+[A-C]|SERIES\s+[A-C]|"
    r"INCORPORATED|INC|CORPORATION|CORP|COMPANY|CO|LTD|LIMITED|PLC|"
    r"LP|LLC|NV|N\s+V|SA|AG|HOLDINGS|HLDGS|GROUP|GRP|THE"
    r")\b",
    re.IGNORECASE,
)
_PUNCT = re.compile(r"[^A-Z0-9 ]+")
_SPACE = re.compile(r"\s+")

# Below this, two names are different companies that happen to share a word.
_MIN_RATIO = 0.90
# A one-token key like "GENERAL" matches far too much to be trusted.
_MIN_KEY_LENGTH = 5


def normalise(name: str) -> str:
    """Reduce a company name to the part that identifies it."""
    text = name.upper()
    # A trailing "(KEYS)" is the symbol, not part of the name.
    text = re.sub(r"\([A-Z.]{1,6}\)\s*$", " ", text)
    text = _PUNCT.sub(" ", text)
    text = _NOISE.sub(" ", text)
    return _SPACE.sub(" ", text).strip()


def build_index(pairs: list[tuple[str, str]]) -> dict[str, str]:
    """A normalised-name to symbol index, dropping ambiguous names.

    A name that appeared against two different symbols in the corpus is not
    evidence of anything, so it is excluded rather than resolved by majority.
    """
    seen: dict[str, set[str]] = {}
    for name, symbol in pairs:
        if not name or not symbol:
            continue
        key = normalise(name)
        if len(key) >= _MIN_KEY_LENGTH:
            seen.setdefault(key, set()).add(symbol.upper())
    return {key: next(iter(symbols)) for key, symbols in seen.items() if len(symbols) == 1}


def resolve(name: str, index: dict[str, str]) -> str | None:
    """The ticker for a written-out company name, or None if unsure."""
    key = normalise(name)
    if len(key) < _MIN_KEY_LENGTH:
        return None
    if key in index:
        return index[key]

    # Fall back to closest match, but only among names that share a first
    # word: comparing against all 3,000 keys invites a high-scoring accident.
    head = key.split(" ")[0]
    best: tuple[float, str | None] = (0.0, None)
    for candidate, symbol in index.items():
        if not candidate.startswith(head):
            continue
        ratio = SequenceMatcher(None, key, candidate).ratio()
        if ratio > best[0]:
            best = (ratio, symbol)
    return best[1] if best[0] >= _MIN_RATIO else None
