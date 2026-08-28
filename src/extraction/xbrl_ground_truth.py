"""
Phase 5 — XBRL ground-truth lookup.

Wraps EdgarClient.get_company_facts() (Phase 1) and pulls out the specific
line-item value SEC has on file for a given ticker + form + fiscal period,
to be used as the ground truth that Phase 5's LLM-extracted numbers get
checked against.

SEC's raw XBRL CompanyFacts JSON shape (relevant part):
    facts["us-gaap"][CONCEPT]["units"]["USD"] -> list of dicts like:
        {"end": "2026-06-27", "start": "2026-03-29", "val": 109417000000,
         "accn": "...", "fy": 2026, "fp": "Q3", "form": "10-Q", "filed": "..."}

    Instant concepts (e.g. Assets) have "end" but no "start".
    Duration concepts (e.g. Revenues, NetIncomeLoss) have both "start" and
    "end" — and a single fiscal period-end can appear MULTIPLE times with
    different "start" dates (a 10-Q's revenue is reported both for the
    current 3-month quarter AND the year-to-date cumulative period, sharing
    the same "end" date). Picking the wrong one silently gives a
    correct-looking but wrong ground truth, so duration matching below
    explicitly picks the shortest period ending on the target date (the
    single quarter, not YTD) unless prefer_ytd=True is passed.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from src.ingestion.edgar_client import EdgarClient

# Line items we score accuracy against. Each maps to a PRIMARY XBRL concept
# tag plus fallbacks — real filers are inconsistent about which GAAP tag
# they use for the "same" line item (e.g. post-ASC-606 filers often use
# RevenueFromContractWithCustomerExcludingAssessedTax instead of the older
# Revenues tag), so we try each in order and use the first with data for
# the target period.
LINE_ITEM_CONCEPTS: dict[str, list[str]] = {
    "total_revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
    ],
    "net_income": [
        "NetIncomeLoss",
        "ProfitLoss",
    ],
    "total_assets": [
        "Assets",
    ],
}

# Instant (balance-sheet, single date) vs. duration (income-statement,
# covers a period) — controls whether we match on "end" alone or need
# "start"+"end" period-length logic.
LINE_ITEM_IS_INSTANT: dict[str, bool] = {
    "total_revenue": False,
    "net_income": False,
    "total_assets": True,
}


def _days_between(start: str, end: str) -> int:
    return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).days


class XBRLGroundTruth:
    """Looks up SEC's official reported value for a line item, for a
    specific ticker/form/fiscal-period-end, to serve as validation ground
    truth. Reuses EdgarClient (Phase 1) for CIK resolution and the actual
    HTTP call — this class only does concept lookup + period matching on
    top of the raw JSON EdgarClient.get_company_facts() already returns."""

    def __init__(self, edgar_client: Optional[EdgarClient] = None):
        self.client = edgar_client or EdgarClient()
        self._facts_cache: dict[str, dict] = {}

    def _get_facts(self, ticker: str) -> dict:
        ticker = ticker.upper()
        if ticker not in self._facts_cache:
            cik10 = self.client.get_cik_for_ticker(ticker)
            self._facts_cache[ticker] = self.client.get_company_facts(cik10)
        return self._facts_cache[ticker]

    def lookup(
        self,
        ticker: str,
        line_item: str,
        form: str,
        report_date: str,
        prefer_ytd: bool = False,
    ) -> Optional[dict]:
        """
        Returns {"value": float, "concept": str, "start": str|None,
        "end": str, "unit": "USD"} for the best-matching XBRL fact, or None
        if SEC has no data for this line item / period / form combination
        (which is a data-availability gap, not necessarily an extraction
        failure — some smaller line items aren't always tagged).
        """
        if line_item not in LINE_ITEM_CONCEPTS:
            raise ValueError(f"Unknown line item '{line_item}'. "
                              f"Known: {list(LINE_ITEM_CONCEPTS)}")

        facts = self._get_facts(ticker)
        us_gaap = facts.get("facts", {}).get("us-gaap", {})
        is_instant = LINE_ITEM_IS_INSTANT[line_item]

        for concept in LINE_ITEM_CONCEPTS[line_item]:
            concept_data = us_gaap.get(concept)
            if not concept_data:
                continue
            usd_entries = concept_data.get("units", {}).get("USD", [])

            candidates = [
                e for e in usd_entries
                if e.get("form") == form and e.get("end") == report_date
            ]
            if not candidates:
                continue

            if is_instant:
                chosen = candidates[0]
                return {
                    "value": float(chosen["val"]),
                    "concept": concept,
                    "start": None,
                    "end": chosen["end"],
                    "unit": "USD",
                }

            # Duration concept: multiple candidates can share the same
            # "end" date (quarter vs. YTD). Sort by period length; pick
            # the shortest (single quarter) unless YTD was requested.
            dated = [c for c in candidates if c.get("start")]
            if not dated:
                continue
            dated.sort(key=lambda c: _days_between(c["start"], c["end"]))
            chosen = dated[-1] if prefer_ytd else dated[0]
            return {
                "value": float(chosen["val"]),
                "concept": concept,
                "start": chosen["start"],
                "end": chosen["end"],
                "unit": "USD",
            }

        return None