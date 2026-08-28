"""
Phase 5 — validate LLM-extracted line items against SEC's XBRL ground truth.

This is the file that produces the project's central portfolio metric:
"matched EDGAR's reported figures in X% of test cases."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.extraction.xbrl_ground_truth import XBRLGroundTruth, LINE_ITEM_CONCEPTS

# Relative tolerance for a "match" — accounts for LLM rounding/formatting
# noise (e.g. reading a table already rounded to millions), not genuine
# extraction errors. 0.5% is tight enough to catch real mistakes (wrong
# period, wrong line item, transposed digits) while forgiving harmless
# rounding.
DEFAULT_TOLERANCE = 0.005


@dataclass
class LineItemResult:
    line_item: str
    extracted_value: Optional[float]
    extracted_period: Optional[str]
    ground_truth_value: Optional[float]
    ground_truth_concept: Optional[str]
    match: Optional[bool]  # None = couldn't be scored (missing data either side)
    pct_diff: Optional[float]
    note: str = ""


@dataclass
class FilingValidationResult:
    ticker: str
    form: str
    report_date: str
    llm_source: str
    line_items: list[LineItemResult] = field(default_factory=list)

    @property
    def scored_count(self) -> int:
        return sum(1 for li in self.line_items if li.match is not None)

    @property
    def matched_count(self) -> int:
        return sum(1 for li in self.line_items if li.match is True)

    @property
    def accuracy(self) -> Optional[float]:
        if self.scored_count == 0:
            return None
        return self.matched_count / self.scored_count


class Validator:
    def __init__(self, xbrl: Optional[XBRLGroundTruth] = None,
                 tolerance: float = DEFAULT_TOLERANCE):
        self.xbrl = xbrl or XBRLGroundTruth()
        self.tolerance = tolerance

    def validate_filing(
        self,
        ticker: str,
        form: str,
        report_date: str,
        extracted: dict,
    ) -> FilingValidationResult:
        result = FilingValidationResult(
            ticker=ticker, form=form, report_date=report_date,
            llm_source=extracted.get("_llm_source", "unknown"),
        )

        for line_item in LINE_ITEM_CONCEPTS:
            ext_entry = extracted.get(line_item) or {}
            ext_value = ext_entry.get("value")
            ext_period = ext_entry.get("period")

            gt = self.xbrl.lookup(ticker, line_item, form, report_date,
                                   prefer_ytd=(ext_period == "ytd"))

            if ext_value is None and gt is None:
                result.line_items.append(LineItemResult(
                    line_item, None, ext_period, None, None, None, None,
                    note="Neither extracted nor found in XBRL — unscored."))
                continue
            if ext_value is None:
                result.line_items.append(LineItemResult(
                    line_item, None, ext_period, gt["value"], gt["concept"],
                    False, None, note="LLM returned null; SEC had data."))
                continue
            if gt is None:
                result.line_items.append(LineItemResult(
                    line_item, ext_value, ext_period, None, None, None, None,
                    note=f"No XBRL fact found for {form}/{report_date} under "
                         f"any known concept tag — unscored, not counted as "
                         f"an extraction failure."))
                continue

            pct_diff = abs(ext_value - gt["value"]) / abs(gt["value"]) if gt["value"] else None
            is_match = pct_diff is not None and pct_diff <= self.tolerance
            result.line_items.append(LineItemResult(
                line_item, ext_value, ext_period, gt["value"], gt["concept"],
                is_match, pct_diff,
                note="" if is_match else "Value mismatch beyond tolerance."))

        return result