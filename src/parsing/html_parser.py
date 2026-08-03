"""
Parse SEC EDGAR inline-XBRL HTML filings into narrative sections and tables.

SEC filings are not semantic HTML — companies use nested divs, layout tables,
and inconsistent heading markup. Item boundaries are detected with a regex plus
a few heuristics; expect to tune these as you add more tickers/sectors.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Optional

import pandas as pd
from bs4 import BeautifulSoup, Tag, XMLParsedAsHTMLWarning

# Inline-XBRL roots are XML-ish; lxml's HTML parser still works well enough.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# Matches SEC Item headings at the start of a line of text:
#   "Item 1.", "Item 1A.", "Item 7A.", "Item 12", "Item 7. MD&A", etc.
# Group 1 = item number + optional letter suffix (1, 1A, 7A, …)
# Group 2 = optional title text after the item label
ITEM_HEADING_RE = re.compile(
    r"^Item\s+(\d+[A-Za-z]?)\.?\s*(.*)$",
    re.IGNORECASE,
)

BLOCK_TAGS = frozenset({"div", "p", "h1", "h2", "h3", "h4", "h5", "h6", "li"})
HEADING_TAGS = frozenset({"div", "p", "h1", "h2", "h3", "h4", "h5", "h6", "span"})

# Headings are short; longer matches are usually in-sentence cross-references
# ("see Part II, Item 7 …") rather than section headers.
MAX_HEADING_CHARS = 200

PREAMBLE_ITEM_NUMBER = ""
PREAMBLE_SECTION_TITLE = "Preamble"


@dataclass(frozen=True)
class SectionInfo:
    """Canonical section identity for grouping; title is for display/citation."""

    item_number: str
    section_title: str


@dataclass
class NarrativeSection:
    """One SEC Item's narrative body text (tables excluded)."""

    item_number: str
    section_title: str
    text: str


@dataclass
class ExtractedTable:
    """One HTML table with document position and enclosing Item section."""

    item_number: str
    section_title: str
    position: int
    dataframe: pd.DataFrame


def serialize_table(df: pd.DataFrame) -> str:
    """Format a table as one line per row for RAG chunks.

    Colspan duplication: HTML tables with merged cells (colspan > 1) are
    flattened by pandas.read_html into repeated values — e.g. one cell
    spanning three columns becomes "California,California,California". We
    collapse consecutive identical values in each row before formatting.

    Rowspan merges, nested tables, and filers that encode layout tables as
    data tables may still need additional heuristics as the corpus grows.
    """
    if df.empty:
        return ""

    has_real_headers = not all(_is_placeholder_column(col) for col in df.columns)
    lines: list[str] = []

    for _, row in df.iterrows():
        collapsed = _collapse_row_duplicates(row.tolist())
        if not any(collapsed):
            continue

        if has_real_headers:
            label = collapsed[0]
            values = [v for v in collapsed[1:] if v]
        else:
            label, values = _first_nonempty_label(collapsed)

        if not label:
            continue
        if values:
            lines.append(f"{label}: {' | '.join(values)}")
        else:
            lines.append(label)

    return "\n".join(lines)


def _is_placeholder_column(name: object) -> bool:
    label = str(name).strip()
    return label.startswith("Unnamed") or label.isdigit()


def _collapse_row_duplicates(values: list[object]) -> list[str]:
    """Remove consecutive duplicate cells produced by colspan expansion."""
    normalized = ["" if pd.isna(v) else str(v).strip() for v in values]
    collapsed: list[str] = []
    for cell in normalized:
        if collapsed and cell == collapsed[-1]:
            continue
        collapsed.append(cell)
    return collapsed


def _first_nonempty_label(values: list[str]) -> tuple[str, list[str]]:
    """When read_html yields no header row, treat first non-empty cell as label."""
    for idx, value in enumerate(values):
        if value:
            rest = [v for i, v in enumerate(values) if i != idx and v]
            return value, rest
    return "", []


class FilingParser:
    """Parse a single downloaded .htm filing into sections and tables."""

    def __init__(self, filepath: str | Path):
        self.filepath = Path(filepath)

    def parse(self) -> tuple[list[NarrativeSection], list[ExtractedTable]]:
        """Return (narrative sections, extracted tables) for this filing."""
        html = self.filepath.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(html, "lxml")
        ordered_tags = soup.find_all(True)
        tag_index = {tag: idx for idx, tag in enumerate(ordered_tags)}

        section_starts = self._find_section_starts(soup, tag_index)
        section_for_index = self._build_section_lookup(section_starts, len(ordered_tags))

        sections = self._extract_narrative(soup, tag_index, section_for_index)
        tables = self._extract_tables(soup, tag_index, section_for_index)
        return sections, tables

    # ---- Item heading detection ---------------------------------------------
    @staticmethod
    def _normalize_text(text: str) -> str:
        """Collapse whitespace and non-breaking spaces for consistent matching."""
        return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()

    @classmethod
    def _parse_item_heading(cls, text: str) -> Optional[SectionInfo]:
        """Return canonical section info for a heading, or None if not a match.

        Detection logic (tune here as you see misparses in the wild):
        1. Normalized text must match ITEM_HEADING_RE from the very start.
        2. Total length must be <= MAX_HEADING_CHARS so we skip mid-paragraph
           references such as "Part II, Item 7 of this Form 10-K".
        3. Table-of-contents links (<a> with only "Item N." and no title) are
           skipped — real section headers almost always include a title or sit
           in a styled block element, not a bare anchor in the TOC grid.

        item_number (e.g. "8", "1A") is the canonical grouping key; section_title
        keeps the filer's original heading text for citations.
        """
        normalized = cls._normalize_text(text)
        match = ITEM_HEADING_RE.match(normalized)
        if not match or len(normalized) > MAX_HEADING_CHARS:
            return None

        item_number = match.group(1).upper()
        title = cls._normalize_text(match.group(2))
        if title:
            section_title = f"Item {item_number}. {title}"
        else:
            section_title = f"Item {item_number}."
        return SectionInfo(item_number=item_number, section_title=section_title)

    @classmethod
    def _is_toc_anchor(cls, tag: Tag, normalized_text: str) -> bool:
        """Bare 'Item N.' links in the TOC are not section boundaries."""
        if tag.name != "a":
            return False
        return bool(re.match(r"^Item\s+\d+[A-Za-z]?\.?$", normalized_text, re.I))

    @classmethod
    def _looks_like_heading(cls, tag: Tag, normalized_text: str) -> bool:
        section_info = cls._parse_item_heading(normalized_text)
        if not section_info:
            return False
        if cls._is_toc_anchor(tag, normalized_text):
            return False
        if tag.name not in HEADING_TAGS:
            return False

        # Prefer leaf-ish nodes so we don't double-count div + inner span.
        if tag.name == "span" and tag.find_parent(["div", "p"], recursive=False):
            return False

        # Title-less "Item N." outside anchors is usually a TOC row — require
        # bold styling or an explicit title when the label stands alone.
        title = ITEM_HEADING_RE.match(normalized_text).group(2).strip()
        if not title:
            style = (tag.get("style") or "") + (tag.find_parent(["div", "p", "span"]) or Tag(name="x")).get("style", "")
            if "font-weight:700" not in style and "font-weight:bold" not in style.lower():
                return False
        return True

    def _find_section_starts(
        self, soup: BeautifulSoup, tag_index: dict[Tag, int]
    ) -> list[tuple[int, SectionInfo]]:
        """Return (document position, section info) for each detected Item heading."""
        starts: list[tuple[int, SectionInfo]] = []
        seen_at_index: dict[int, SectionInfo] = {}

        for tag in soup.find_all(HEADING_TAGS):
            if tag.find_parent("table"):
                continue
            normalized = self._normalize_text(tag.get_text(" ", strip=True))
            if not self._looks_like_heading(tag, normalized):
                continue
            section_info = self._parse_item_heading(normalized)
            if not section_info:
                continue
            idx = tag_index[tag]
            # Keep the first heading at each index; skip duplicate span/div pairs.
            if idx not in seen_at_index:
                seen_at_index[idx] = section_info
                starts.append((idx, section_info))

        starts.sort(key=lambda pair: pair[0])
        return starts

    @staticmethod
    def _build_section_lookup(
        section_starts: list[tuple[int, SectionInfo]], num_tags: int
    ) -> list[SectionInfo]:
        """Map every tag index to the active Item section at that point."""
        preamble = SectionInfo(
            item_number=PREAMBLE_ITEM_NUMBER,
            section_title=PREAMBLE_SECTION_TITLE,
        )
        if not section_starts:
            return [preamble] * num_tags

        lookup: list[SectionInfo] = []
        start_ptr = 0
        current = preamble
        for idx in range(num_tags):
            while start_ptr < len(section_starts) and section_starts[start_ptr][0] <= idx:
                current = section_starts[start_ptr][1]
                start_ptr += 1
            lookup.append(current)
        return lookup

    # ---- narrative extraction ------------------------------------------------
    @classmethod
    def _is_leaf_block(cls, tag: Tag) -> bool:
        if tag.name not in BLOCK_TAGS:
            return False
        return tag.find(BLOCK_TAGS, recursive=False) is None

    def _extract_narrative(
        self,
        soup: BeautifulSoup,
        tag_index: dict[Tag, int],
        section_for_index: list[SectionInfo],
    ) -> list[NarrativeSection]:
        buffers: dict[str, list[str]] = {}
        titles: dict[str, str] = {}
        order: list[str] = []

        def ensure_section(item_number: str, section_title: str) -> None:
            if item_number not in buffers:
                buffers[item_number] = []
                titles[item_number] = section_title
                order.append(item_number)
            elif section_title and titles[item_number] == PREAMBLE_SECTION_TITLE:
                titles[item_number] = section_title

        ensure_section(PREAMBLE_ITEM_NUMBER, PREAMBLE_SECTION_TITLE)

        for tag in soup.find_all(BLOCK_TAGS):
            if tag.find_parent("table"):
                continue
            if not self._is_leaf_block(tag):
                continue

            normalized = self._normalize_text(tag.get_text(" ", strip=True))
            if not normalized:
                continue

            # Section headings themselves are keys, not body text.
            if self._looks_like_heading(tag, normalized):
                section_info = self._parse_item_heading(normalized)
                if section_info:
                    ensure_section(section_info.item_number, section_info.section_title)
                continue

            idx = tag_index[tag]
            section = section_for_index[idx]
            ensure_section(section.item_number, section.section_title)
            buffers[section.item_number].append(normalized)

        return [
            NarrativeSection(
                item_number=item_number,
                section_title=titles[item_number],
                text="\n\n".join(buffers[item_number]),
            )
            for item_number in order
            if buffers[item_number]
        ]

    # ---- table extraction ----------------------------------------------------
    def _extract_tables(
        self,
        soup: BeautifulSoup,
        tag_index: dict[Tag, int],
        section_for_index: list[SectionInfo],
    ) -> list[ExtractedTable]:
        extracted: list[ExtractedTable] = []

        for position, table_tag in enumerate(soup.find_all("table")):
            idx = tag_index[table_tag]
            section = section_for_index[idx]
            try:
                frames = pd.read_html(StringIO(str(table_tag)), flavor="lxml")
            except ValueError:
                continue
            if not frames:
                continue
            extracted.append(
                ExtractedTable(
                    item_number=section.item_number,
                    section_title=section.section_title,
                    position=position,
                    dataframe=frames[0],
                )
            )

        return extracted
