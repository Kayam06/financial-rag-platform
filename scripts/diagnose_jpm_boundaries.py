"""
Investigate JPM section-boundary detection for one filing.

Usage:
    python -m scripts.diagnose_jpm_boundaries

Reads chunks.jsonl for a single JPM 10-Q and prints the detected section
sequence. Also walks the source .htm to show parser boundary events in true
document order (chunks.jsonl groups all text chunks before all table chunks,
so the HTML walk is the authoritative heading timeline).
"""
import json
import sys
from pathlib import Path

import pandas as pd

from src.config import settings
from src.parsing.html_parser import FilingParser

TICKER = "JPM"
FORM = "10-Q"
PREVIEW_CHARS = 80

# Most recent JPM 10-Q in the manifest at time of writing.
TARGET_FILING_DATE = "2026-05-01"


def _safe_print(text: str) -> None:
    encoding = sys.stdout.encoding or "utf-8"
    print(text.encode(encoding, errors="replace").decode(encoding))


def _preview(content: str) -> str:
    one_line = content.replace("\n", " ").strip()
    if len(one_line) > PREVIEW_CHARS:
        return one_line[:PREVIEW_CHARS] + "..."
    return one_line


def _load_target_filing_path() -> Path:
    manifest = pd.read_csv(Path(settings.raw_data_dir) / "manifest.csv")
    row = manifest[
        (manifest["ticker"] == TICKER)
        & (manifest["form"] == FORM)
        & (manifest["filing_date"] == TARGET_FILING_DATE)
    ]
    if row.empty:
        raise FileNotFoundError(
            f"No manifest row for {TICKER} {FORM} filed {TARGET_FILING_DATE}"
        )
    return Path(row.iloc[0]["local_path"])


def print_chunks_from_jsonl(filing_date: str) -> None:
    chunks_path = Path(settings.processed_data_dir) / "chunks.jsonl"
    chunks: list[dict] = []
    with chunks_path.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if (
                record.get("ticker") == TICKER
                and record.get("form") == FORM
                and record.get("filing_date") == filing_date
            ):
                chunks.append(record)

    _safe_print(f"\n{'=' * 72}")
    _safe_print(f"CHUNKS FROM chunks.jsonl — {TICKER} {FORM} filed {filing_date}")
    _safe_print(f"Total chunks: {len(chunks)}")
    _safe_print(
        "Note: parse_filings writes all narrative chunks first, then all table "
        "chunks — this is NOT strict document order. See HTML boundary walk below."
    )
    _safe_print(f"{'=' * 72}\n")

    last_section = object()
    for idx, record in enumerate(chunks, start=1):
        section_title = record.get("section_title", "")
        item_number = record.get("item_number", "")
        if section_title != last_section:
            _safe_print(f"--- section boundary: item_number={item_number!r} section_title={section_title!r} ---")
            last_section = section_title
        _safe_print(
            f"[{idx:4}] type={record.get('chunk_type'):5} "
            f"item={item_number!r:4} "
            f"chunk_index={record.get('chunk_index')} "
            f"| {_preview(record.get('content', ''))}"
        )


def print_html_boundary_walk(filepath: Path) -> None:
    from bs4 import BeautifulSoup

    html = filepath.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "lxml")
    ordered_tags = soup.find_all(True)
    tag_index = {tag: idx for idx, tag in enumerate(ordered_tags)}

    parser = FilingParser(filepath)
    starts = parser._find_section_starts(soup, tag_index)

    _safe_print(f"\n{'=' * 72}")
    _safe_print(f"PARSER BOUNDARY WALK (document order) — {filepath.name}")
    _safe_print(f"Detected {len(starts)} section-start events")
    _safe_print(f"{'=' * 72}\n")

    for pos, (doc_idx, info) in enumerate(starts, start=1):
        tag = ordered_tags[doc_idx]
        normalized = parser._normalize_text(tag.get_text(" ", strip=True))
        _safe_print(
            f"[{pos:3}] doc_idx={doc_idx:6} tag=<{tag.name}> "
            f"item_number={info.item_number!r} "
            f"section_title={info.section_title!r}"
        )
        _safe_print(f"       heading text: {normalized[:120]}")

    _safe_print(f"\n{'-' * 72}")
    _safe_print("SIGNATURE / post-Item candidate scan (all matches in document order)")
    _safe_print(f"{'-' * 72}\n")

    for doc_idx, tag in enumerate(ordered_tags):
        if tag.name not in {"div", "p", "span", "h1", "h2", "h3", "h4", "h5", "h6"}:
            continue
        if tag.find_parent("table"):
            continue
        normalized = parser._normalize_text(tag.get_text(" ", strip=True))
        if len(normalized) > 200:
            continue
        compact = parser._compact_boundary_text(normalized)
        if "SIGNATURE" in compact or "SIGNAT" in compact.upper():
            is_boundary = parser._looks_like_post_item_boundary(tag, normalized)
            parsed = parser._parse_post_item_boundary(normalized)
            _safe_print(
                f"doc_idx={doc_idx:6} tag=<{tag.name}> "
                f"boundary={is_boundary} "
                f"text={normalized!r}"
            )
            if parsed:
                _safe_print(f"         -> would set section_title={parsed.section_title!r}")


def print_item_heading_scan(filepath: Path) -> None:
    from bs4 import BeautifulSoup

    html = filepath.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "lxml")
    ordered_tags = soup.find_all(True)
    tag_index = {tag: idx for idx, tag in enumerate(ordered_tags)}
    parser = FilingParser(filepath)

    _safe_print(f"\n{'=' * 72}")
    _safe_print("ITEM HEADING CANDIDATES (matched vs rejected)")
    _safe_print(f"{'=' * 72}\n")

    for doc_idx, tag in enumerate(ordered_tags):
        if tag.name not in {"div", "p", "span", "h1", "h2", "h3", "h4", "h5", "h6"}:
            continue
        if tag.find_parent("table"):
            continue
        normalized = parser._normalize_text(tag.get_text(" ", strip=True))
        if not normalized.upper().startswith("ITEM"):
            continue
        if len(normalized) > 200:
            continue

        looks = parser._looks_like_heading(tag, normalized)
        parsed = parser._parse_item_heading(normalized)
        status = "ACCEPT" if looks else "reject"
        _safe_print(
            f"[{status:6}] doc_idx={doc_idx:6} tag=<{tag.name}> "
            f"text={normalized[:100]!r}"
        )
        if parsed and looks:
            _safe_print(
                f"         -> item_number={parsed.item_number!r} "
                f"section_title={parsed.section_title!r}"
            )


def main():
    filing_path = _load_target_filing_path()
    _safe_print(f"Target filing: {filing_path}")

    print_chunks_from_jsonl(TARGET_FILING_DATE)
    print_html_boundary_walk(filing_path)
    print_item_heading_scan(filing_path)


if __name__ == "__main__":
    main()
