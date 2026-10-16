"""Matching a committee's remit against what a member actually traded."""

from __future__ import annotations

from datetime import date

from trade_bot.committees import (
    JURISDICTIONS,
    ConflictIndex,
    jurisdiction_for,
    measure_committee_edge,
    save_sic_cache,
)
from trade_bot.database import Store
from trade_bot.domain import (
    AssetClass,
    Chamber,
    DisclosedTrade,
    Politician,
    TransactionType,
)

SEATS = [
    {
        "politician_id": "senate-hep",
        "committee_id": "SSHR",
        "committee_name": "Senate Committee on Health, Education, Labor, and Pensions",
        "chamber": "senate",
        "title": "",
        "rank": 4,
        "jurisdiction": "pharmaceuticals and health services",
    },
    {
        "politician_id": "senate-chair",
        "committee_id": "SSHR",
        "committee_name": "Senate Committee on Health, Education, Labor, and Pensions",
        "chamber": "senate",
        "title": "Chairman",
        "rank": 1,
        "jurisdiction": "pharmaceuticals and health services",
    },
    {
        "politician_id": "senate-farm",
        "committee_id": "SSAF",
        "committee_name": "Senate Committee on Agriculture, Nutrition, and Forestry",
        "chamber": "senate",
        "title": "",
        "rank": 6,
        "jurisdiction": "agriculture and food",
    },
]


def _prepare(settings, store: Store) -> ConflictIndex:
    store.replace_committee_memberships(SEATS)
    save_sic_cache(
        settings,
        {
            # Pfizer: pharmaceutical preparations.
            "PFE": {"sic": "2834", "industry": "Pharmaceutical Preparations", "name": "Pfizer"},
            # Apple: electronic computers, which no mapped committee covers.
            "AAPL": {"sic": "3571", "industry": "Electronic Computers", "name": "Apple"},
            "ADM": {"sic": "2070", "industry": "Fats & Oils", "name": "ADM"},
        },
    )
    return ConflictIndex(settings, store)


def test_a_remit_matches_only_the_industries_it_covers(settings, store):
    index = _prepare(settings, store)
    assert index.overlaps("senate-hep", "PFE"), "health committee covers a drug maker"
    assert index.overlaps("senate-hep", "AAPL") == [], "it does not cover a computer maker"
    assert index.overlaps("senate-farm", "ADM"), "agriculture covers an oils processor"
    assert index.overlaps("senate-farm", "PFE") == []


def test_a_chair_counts_for_more_than_a_junior_member(settings, store):
    """Both see the same briefings; only one sets the agenda."""
    index = _prepare(settings, store)
    assert index.strength("senate-chair", "PFE") > index.strength("senate-hep", "PFE")
    assert index.strength("senate-hep", "AAPL") == 0.0


def test_an_unknown_industry_never_matches(settings, store):
    """A ticker with no SIC on file must not silently match everything."""
    index = _prepare(settings, store)
    assert index.overlaps("senate-hep", "UNKNOWN") == []
    assert jurisdiction_for("") == set()


def test_broad_committees_are_deliberately_unmapped():
    """A rule that matches everything reports a conflict on every trade.

    Ways and Means touches every taxpayer and Judiciary reaches every merger,
    so neither is in the map. Their absence is the point, not an oversight.
    """
    for committee in ("HSWM", "HSJU", "HSRU", "SSJU", "SSFR"):
        assert committee not in JURISDICTIONS


def test_the_description_names_the_committee_and_refuses_to_accuse(settings, store):
    index = _prepare(settings, store)
    note = index.describe("senate-chair", "PFE", "Ann One")
    assert note is not None
    assert "Health, Education, Labor, and Pensions" in note
    assert "Pharmaceutical Preparations" in note
    assert "not evidence of wrongdoing" in note


def test_the_edge_is_measured_per_side_rather_than_assumed(settings, store, prices):
    """The measurement is what decides the weight, so it has to exist.

    Whether it comes out positive on this fixture is beside the point; the
    contract is that both directions are reported separately, because the
    real answer differs between them.
    """
    store.replace_committee_memberships(SEATS)
    save_sic_cache(
        settings,
        {"ACME": {"sic": "2834", "industry": "Pharmaceutical Preparations", "name": "Acme"}},
    )
    store.upsert_politicians(
        [
            Politician(
                politician_id="senate-hep",
                full_name="Ann One",
                last_name="One",
                chamber=Chamber.SENATE,
            )
        ]
    )
    store.add_trades(
        [
            DisclosedTrade(
                trade_id=f"t{index}",
                filing_id="senate:2026:1",
                politician_id="senate-hep",
                chamber=Chamber.SENATE,
                symbol="ACME",
                asset_name="Acme Common Stock",
                asset_class=AssetClass.STOCK,
                transaction_type=kind,
                transaction_date=date(2026, 1, 6),
                disclosure_date=date(2026, 2, 2),
                amount_low=1001.0,
                amount_high=15000.0,
            )
            for index, kind in enumerate((TransactionType.PURCHASE, TransactionType.SALE))
        ]
    )
    result = measure_committee_edge(settings, store, prices, horizon_days=30)
    assert set(result) >= {"buy", "sell", "horizon_days"}
    assert result["buy"]["conflicted"]["n"] == 1
    assert result["sell"]["conflicted"]["n"] == 1
