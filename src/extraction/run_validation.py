"""
Phase 5 CLI entry point.

Usage:
    python -m src.extraction.run_validation
    python -m src.extraction.run_validation --ticker AAPL
    python -m src.extraction.run_validation --tolerance 0.01

Runs extraction + XBRL validation across every unique (ticker, form,
report_date) filing found in chunk_metadata.jsonl, and writes per-filing +
aggregate accuracy results to data/processed/validation_results.json.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from src.extraction.line_item_extractor import LineItemExtractor, ExtractionError
from src.extraction.validator import Validator
from src.retrieval.retriever import (
    Retriever,
    METADATA_PATH,
)  # ADJUST if this import fails — see note below

OUTPUT_PATH = Path("data/processed/validation_results.json")


def get_unique_filings(
    metadata_path: Path, ticker_filter: str | None = None
) -> list[tuple[str, str, str]]:
    seen = set()
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ticker, form, report_date = (
                row.get("ticker"),
                row.get("form"),
                row.get("report_date"),
            )
            if not (ticker and form and report_date):
                continue
            if ticker_filter and ticker.upper() != ticker_filter.upper():
                continue
            seen.add((ticker, form, report_date))
    return sorted(seen)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 5: extract + validate line items against XBRL"
    )
    parser.add_argument("--ticker", type=str, default=None, help="Only run this ticker")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.005,
        help="Relative match tolerance (default 0.5%%)",
    )
    args = parser.parse_args()

    filings = get_unique_filings(METADATA_PATH, args.ticker)
    if not filings:
        print("No filings found in chunk_metadata.jsonl matching filter.")
        return

    print(f"Found {len(filings)} unique filings to process.")

    extractor = LineItemExtractor(retriever=Retriever())
    validator = Validator(tolerance=args.tolerance)

    all_results = []
    for i, (ticker, form, report_date) in enumerate(filings, 1):
        print(
            f"[{i}/{len(filings)}] {ticker} {form} {report_date} ... ",
            end="",
            flush=True,
        )
        try:
            extracted = extractor.extract(ticker, form, report_date)
        except ExtractionError as e:
            print(f"SKIPPED (extraction failed): {e}")
            continue

        result = validator.validate_filing(ticker, form, report_date, extracted)
        acc_str = f"{result.accuracy:.0%}" if result.accuracy is not None else "n/a"
        print(
            f"accuracy {acc_str} ({result.matched_count}/{result.scored_count} scored) [{result.llm_source}]"
        )
        all_results.append(result)

    total_matched = sum(r.matched_count for r in all_results)
    total_scored = sum(r.scored_count for r in all_results)
    overall_accuracy = total_matched / total_scored if total_scored else None

    print("\n" + "=" * 60)
    if overall_accuracy is not None:
        print(
            f"OVERALL ACCURACY: {overall_accuracy:.1%} "
            f"({total_matched}/{total_scored} line items matched EDGAR's XBRL data)"
        )
    else:
        print("OVERALL ACCURACY: no scorable line items across all filings.")
    print("=" * 60)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "overall_accuracy": overall_accuracy,
                "total_matched": total_matched,
                "total_scored": total_scored,
                "tolerance": args.tolerance,
                "filings": [
                    {
                        "ticker": r.ticker,
                        "form": r.form,
                        "report_date": r.report_date,
                        "llm_source": r.llm_source,
                        "accuracy": r.accuracy,
                        "matched_count": r.matched_count,
                        "scored_count": r.scored_count,
                        "line_items": [asdict(li) for li in r.line_items],
                    }
                    for r in all_results
                ],
            },
            f,
            indent=2,
        )

    print(f"\nFull results written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
