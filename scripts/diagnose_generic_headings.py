"""
Investigate Fix B generic-heading detection on standard Item-structured filers.

Usage:
    python -m scripts.diagnose_generic_headings
    python -m scripts.diagnose_generic_headings --ticker MSFT --form 10-K --filing-date 2026-07-29

Walks one filing's HTML in document order, classifies each detected boundary
(Item / generic / post-Item), and flags generic headings that fire while a
numbered Item section is still open (likely sub-headings misclassified as
top-level sections).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from bs4 import BeautifulSoup, Tag

from src.config import settings
from src.parsing.html_parser import (
    HEADING_TAGS,
    FilingParser,
    SectionInfo,
)

DEFAULT_TICKER = "AAPL"
DEFAULT_FORM = "10-K"
DEFAULT_FILING_DATE = "2025-10-31"


def _safe_print(text: str) -> None:
    encoding = sys.stdout.encoding or "utf-8"
    print(text.encode(encoding, errors="replace").decode(encoding))


def _load_filing_path(ticker: str, form: str, filing_date: str) -> Path:
    manifest = pd.read_csv(Path(settings.raw_data_dir) / "manifest.csv")
    row = manifest[
        (manifest["ticker"] == ticker)
        & (manifest["form"] == form)
        & (manifest["filing_date"] == filing_date)
    ]
    if row.empty:
        raise FileNotFoundError(
            f"No manifest row for {ticker} {form} filed {filing_date}"
        )
    return Path(row.iloc[0]["local_path"])


def _classify_heading(
    parser: FilingParser, tag: Tag, normalized: str
) -> tuple[Optional[str], Optional[SectionInfo]]:
    """Return (kind, SectionInfo) where kind is item|generic|post_item."""
    if parser._looks_like_heading(tag, normalized):
        return "item", parser._parse_item_heading(normalized)
    if parser._looks_like_post_item_boundary(tag, normalized):
        return "post_item", parser._parse_post_item_boundary(normalized)
    if parser._looks_like_generic_heading(tag, normalized):
        return "generic", parser._parse_generic_heading(normalized)
    return None, None


def print_boundary_timeline(filepath: Path) -> None:
    html = filepath.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "lxml")
    ordered_tags = soup.find_all(True)
    tag_index = {tag: idx for idx, tag in enumerate(ordered_tags)}
    parser = FilingParser(filepath)

    events: list[dict] = []
    seen_indices: set[int] = set()

    for tag in soup.find_all(HEADING_TAGS):
        if tag.find_parent("table"):
            continue
        normalized = parser._normalize_text(tag.get_text(" ", strip=True))
        if not normalized:
            continue
        idx = tag_index[tag]
        if idx in seen_indices:
            continue

        kind, info = _classify_heading(parser, tag, normalized)
        if not kind or not info:
            continue
        seen_indices.add(idx)
        events.append(
            {
                "doc_idx": idx,
                "kind": kind,
                "info": info,
                "tag": tag.name,
                "text": normalized,
            }
        )

    events.sort(key=lambda e: e["doc_idx"])

    _safe_print(f"\n{'=' * 80}")
    _safe_print(f"BOUNDARY TIMELINE — {filepath.name}")
    _safe_print(f"Total detected boundaries: {len(events)}")
    _safe_print(f"{'=' * 80}\n")

    open_item: Optional[str] = None
    open_item_title: Optional[str] = None
    generic_inside_item: list[dict] = []
    kind_counts = {"item": 0, "generic": 0, "post_item": 0}

    for pos, event in enumerate(events, start=1):
        kind = event["kind"]
        info: SectionInfo = event["info"]
        kind_counts[kind] += 1

        inside_flag = ""
        if kind == "item":
            open_item = info.item_number
            open_item_title = info.section_title
        elif kind == "generic" and open_item:
            inside_flag = " *** INSIDE OPEN ITEM ***"
            generic_inside_item.append(
                {
                    "open_item": open_item,
                    "open_item_title": open_item_title,
                    **event,
                }
            )
        elif kind == "post_item":
            open_item = None
            open_item_title = None

        _safe_print(
            f"[{pos:3}] {kind:9} doc_idx={event['doc_idx']:6} "
            f"tag=<{event['tag']}> "
            f"item_number={info.item_number!r} "
            f"section_title={info.section_title[:70]!r}{inside_flag}"
        )
        _safe_print(f"       text: {event['text'][:120]}")

    _safe_print(f"\n{'-' * 80}")
    _safe_print(
        f"Summary: {kind_counts['item']} Item | "
        f"{kind_counts['generic']} generic | "
        f"{kind_counts['post_item']} post-Item"
    )
    _safe_print(
        f"Generic headings INSIDE an open numbered Item: {len(generic_inside_item)}"
    )
    _safe_print(f"{'-' * 80}\n")

    if generic_inside_item:
        _safe_print("FLAGGED — generic heading while numbered Item still open:\n")
        for flag in generic_inside_item[:40]:
            _safe_print(
                f"  while Item {flag['open_item']!r} ({flag['open_item_title'][:50]!r}) "
                f"was open @ doc_idx={flag['doc_idx']}:"
            )
            _safe_print(f"    generic -> {flag['text'][:100]!r}")
        if len(generic_inside_item) > 40:
            _safe_print(f"  ... and {len(generic_inside_item) - 40} more")
    else:
        _safe_print("No generic headings detected inside open numbered Items.")


def print_chunks_summary(ticker: str, form: str, filing_date: str) -> None:
    chunks_path = Path(settings.processed_data_dir) / "chunks.jsonl"
    total = empty = numbered = 0
    empty_titles: dict[str, int] = {}
    item_counts: dict[str, int] = {}

    with chunks_path.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if (
                record.get("ticker") != ticker
                or record.get("form") != form
                or record.get("filing_date") != filing_date
            ):
                continue
            total += 1
            inum = record.get("item_number") or ""
            if inum:
                numbered += 1
                item_counts[inum] = item_counts.get(inum, 0) + 1
            else:
                empty += 1
                title = record.get("section_title", "")
                empty_titles[title] = empty_titles.get(title, 0) + 1

    _safe_print(f"\n{'=' * 80}")
    _safe_print(f"CHUNKS SUMMARY — {ticker} {form} filed {filing_date}")
    _safe_print(f"Total chunks: {total} | numbered item_number: {numbered} | empty: {empty}")
    if total:
        _safe_print(f"Empty rate: {100 * empty / total:.1f}%")
    _safe_print("\nTop empty section_title values:")
    for title, count in sorted(empty_titles.items(), key=lambda x: -x[1])[:15]:
        _safe_print(f"  {count:4} {title!r}")
    _safe_print("\nNumbered item_number counts:")
    for inum, count in sorted(item_counts.items(), key=lambda x: (len(x[0]), x[0])):
        _safe_print(f"  {count:4} item {inum!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose generic heading detection")
    parser.add_argument("--ticker", default=DEFAULT_TICKER)
    parser.add_argument("--form", default=DEFAULT_FORM)
    parser.add_argument("--filing-date", default=DEFAULT_FILING_DATE)
    args = parser.parse_args()

    filing_path = _load_filing_path(args.ticker, args.form, args.filing_date)
    _safe_print(f"Target: {args.ticker} {args.form} filed {args.filing_date}")
    _safe_print(f"File:   {filing_path}")

    print_boundary_timeline(filing_path)
    print_chunks_summary(args.ticker, args.form, args.filing_date)


if __name__ == "__main__":
    main()
