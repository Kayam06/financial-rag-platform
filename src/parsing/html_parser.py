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

# Matched against whitespace-stripped, uppercased text with all non-letters
# removed so "SIGNAT URES" (split across styled spans) still matches SIGNATURES.
POST_ITEM_BOUNDARY_COMPACT_RE = re.compile(
    r"^(?:"
    r"SIGNATURES?"
    r"|EXHIBIT(?:INDEX|LISTING)"
    r"|EXHIBITS"
    r"|INDEXTOEXHIBITS"
    r")$",
    re.IGNORECASE,
)

# pandas.read_html sometimes types numeric year headers as floats (2024.0).
INTEGER_FLOAT_SUFFIX_RE = re.compile(r"^-?\d+\.0$")

BLOCK_TAGS = frozenset({"div", "p", "h1", "h2", "h3", "h4", "h5", "h6", "li"})
HEADING_TAGS = frozenset({"div", "p", "h1", "h2", "h3", "h4", "h5", "h6", "span"})

# Headings are short; longer matches are usually in-sentence cross-references
# ("see Part II, Item 7 …") rather than section headers.
MAX_HEADING_CHARS = 200
MIN_GENERIC_HEADING_CHARS = 4
MAX_GENERIC_HEADING_CHARS = 120
MAX_GENERIC_HEADING_WORDS = 12

PREAMBLE_ITEM_NUMBER = ""
PREAMBLE_SECTION_TITLE = "Preamble"

# Internal sentinel: narrative buffers keyed by item_number merge on this title slot.
_ITEM_BUFFER_TITLE = "__item__"


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

    Integer float suffix: read_html may infer numeric dtypes for year-like
    header cells, rendering "2024" as "2024.0". We strip the ".0" suffix
    before deduplication so "2024" and "2024.0" collapse to one value.

    # TODO: multi-level header tables not yet handled, see project doc Issue 4 follow-up

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
    """Normalize cells, then remove consecutive duplicates from colspan expansion."""
    normalized = [_normalize_cell_value(v) for v in values]
    collapsed: list[str] = []
    for cell in normalized:
        if collapsed and cell == collapsed[-1]:
            continue
        collapsed.append(cell)
    return collapsed


def _normalize_cell_value(value: object) -> str:
    """Strip whitespace and render integer-valued floats as bare integers."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if INTEGER_FLOAT_SUFFIX_RE.match(text):
        return text[:-2]
    return text


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
    def _compact_boundary_text(cls, text: str) -> str:
        """Strip non-letters so split headings like 'SIGNAT URES' still match."""
        return re.sub(r"[^A-Za-z]", "", cls._normalize_text(text)).upper()

    @classmethod
    def _parse_post_item_boundary(cls, text: str) -> Optional[SectionInfo]:
        """Return section info for post-Item markers (signatures, exhibit index).

        These headings reset item_number to empty so trailing document content
        is not absorbed into the last numbered Item (typically Item 16).
        Compact matching handles filers that split words across styled spans.
        """
        normalized = cls._normalize_text(text)
        if len(normalized) > MAX_HEADING_CHARS:
            return None
        compact = cls._compact_boundary_text(normalized)
        if not POST_ITEM_BOUNDARY_COMPACT_RE.match(compact):
            return None
        return SectionInfo(item_number=PREAMBLE_ITEM_NUMBER, section_title=normalized)

    @classmethod
    def _is_styled_heading(cls, tag: Tag) -> bool:
        """True when tag markup suggests a section header rather than body text."""
        if tag.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            return True
        style = tag.get("style") or ""
        parent = tag.find_parent(["div", "p", "span"])
        if parent is not None:
            style += parent.get("style") or ""
        if "font-weight:700" in style or "font-weight:bold" in style.lower():
            return True
        letters = [c for c in tag.get_text(strip=True) if c.isalpha()]
        if letters and sum(c.isupper() for c in letters) / len(letters) >= 0.85:
            return True
        return False

    @classmethod
    def _heading_text_for_detection(cls, text: str) -> str:
        """Strip leading page numbers filers often prepend to section titles."""
        normalized = cls._normalize_text(text)
        return re.sub(r"^\d+\s+", "", normalized)

    @classmethod
    def _parse_generic_heading(cls, text: str) -> Optional[SectionInfo]:
        """Return section info for non-Item styled headings (e.g. JPM MD&A blocks).

        Some filers label MD&A subsections with business titles ("INTRODUCTION",
        "CAPITAL RISK MANAGEMENT") instead of "Item N." headings. We keep the
        filer's title as section_title and leave item_number empty — no attempt
        to map these to canonical Item numbers.
        """
        normalized = cls._heading_text_for_detection(text)
        if len(normalized) < MIN_GENERIC_HEADING_CHARS:
            return None
        if len(normalized) > MAX_GENERIC_HEADING_CHARS:
            return None
        if len(normalized.split()) > MAX_GENERIC_HEADING_WORDS:
            return None
        if cls._parse_item_heading(normalized) or cls._parse_post_item_boundary(normalized):
            return None
        if normalized.startswith("(") or normalized.lower().startswith("refer to"):
            return None
        return SectionInfo(item_number=PREAMBLE_ITEM_NUMBER, section_title=normalized)

    @classmethod
    def _looks_like_generic_heading(cls, tag: Tag, normalized_text: str) -> bool:
        if cls._parse_generic_heading(normalized_text) is None:
            return False
        if tag.find_parent("table"):
            return False
        if tag.name not in HEADING_TAGS:
            return False
        if tag.name == "span" and tag.find_parent(["div", "p"], recursive=False):
            return False
        return cls._is_styled_heading(tag)

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

    @classmethod
    def _looks_like_post_item_boundary(cls, tag: Tag, normalized_text: str) -> bool:
        if cls._parse_post_item_boundary(normalized_text) is None:
            return False
        if tag.find_parent("table"):
            return False
        if tag.name not in HEADING_TAGS:
            return False
        if tag.name == "span" and tag.find_parent(["div", "p"], recursive=False):
            return False
        return True

    def _find_section_starts(
        self, soup: BeautifulSoup, tag_index: dict[Tag, int]
    ) -> list[tuple[int, SectionInfo]]:
        """Return (document position, section info) for Item and post-Item headings."""
        item_starts: list[tuple[int, SectionInfo]] = []
        post_item_candidates: list[tuple[int, SectionInfo]] = []
        seen_at_index: dict[int, SectionInfo] = {}

        for tag in soup.find_all(HEADING_TAGS):
            if tag.find_parent("table"):
                continue
            normalized = self._normalize_text(tag.get_text(" ", strip=True))

            section_info: Optional[SectionInfo] = None
            is_post_item = False
            if self._looks_like_heading(tag, normalized):
                section_info = self._parse_item_heading(normalized)
            elif self._looks_like_post_item_boundary(tag, normalized):
                section_info = self._parse_post_item_boundary(normalized)
                is_post_item = True
            elif self._looks_like_generic_heading(tag, normalized):
                section_info = self._parse_generic_heading(normalized)

            if not section_info:
                continue
            idx = tag_index[tag]
            if idx in seen_at_index:
                continue
            seen_at_index[idx] = section_info
            if is_post_item:
                post_item_candidates.append((idx, section_info))
            else:
                item_starts.append((idx, section_info))

        starts = list(item_starts)
        if item_starts and post_item_candidates:
            last_item_idx = max(idx for idx, _ in item_starts)
            starts.extend(
                (idx, info)
                for idx, info in post_item_candidates
                if idx > last_item_idx
            )

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

    @staticmethod
    def _buffer_key(info: SectionInfo) -> tuple[str, str]:
        """Bucket key for narrative text; item sections merge, unnumbered do not."""
        if info.item_number:
            return (info.item_number, _ITEM_BUFFER_TITLE)
        return (PREAMBLE_ITEM_NUMBER, info.section_title)

    def _extract_narrative(
        self,
        soup: BeautifulSoup,
        tag_index: dict[Tag, int],
        section_for_index: list[SectionInfo],
    ) -> list[NarrativeSection]:
        buffers: dict[tuple[str, str], list[str]] = {}
        titles: dict[tuple[str, str], str] = {}
        order: list[tuple[str, str]] = []

        def activate(section_info: SectionInfo) -> None:
            key = self._buffer_key(section_info)
            if key not in buffers:
                buffers[key] = []
                titles[key] = section_info.section_title
                order.append(key)

        activate(
            SectionInfo(
                item_number=PREAMBLE_ITEM_NUMBER,
                section_title=PREAMBLE_SECTION_TITLE,
            )
        )

        for tag in soup.find_all(BLOCK_TAGS):
            if tag.find_parent("table"):
                continue
            if not self._is_leaf_block(tag):
                continue

            normalized = self._normalize_text(tag.get_text(" ", strip=True))
            if not normalized:
                continue

            if self._looks_like_heading(tag, normalized):
                section_info = self._parse_item_heading(normalized)
                if section_info:
                    activate(section_info)
                continue

            if self._looks_like_post_item_boundary(tag, normalized):
                boundary = self._parse_post_item_boundary(normalized)
                if boundary:
                    activate(boundary)
                continue

            if self._looks_like_generic_heading(tag, normalized):
                generic = self._parse_generic_heading(normalized)
                if generic:
                    activate(generic)
                continue

            idx = tag_index[tag]
            section = section_for_index[idx]
            activate(section)
            buffers[self._buffer_key(section)].append(normalized)

        return [
            NarrativeSection(
                item_number=key[0],
                section_title=titles[key],
                text="\n\n".join(buffers[key]),
            )
            for key in order
            if buffers[key]
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
