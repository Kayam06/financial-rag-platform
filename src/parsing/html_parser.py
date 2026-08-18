"""
Parse SEC EDGAR inline-XBRL HTML filings into narrative sections and tables.

SEC filings are not semantic HTML — companies use nested divs, layout tables,
and inconsistent heading markup. Item boundaries are detected with a regex plus
a few heuristics; expect to tune these as you add more tickers/sectors.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, replace
from io import StringIO
from pathlib import Path
from typing import Optional

import pandas as pd
from bs4 import BeautifulSoup, Tag, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

ITEM_HEADING_RE = re.compile(
    r"^Item\s+(\d+[A-Za-z]?)\.?\s*(.*)$",
    re.IGNORECASE,
)

# Matches "PART I", "PART II", etc. — 10-Qs restart Item numbering in Part II
# (Item 1/2 mean different things in Part I vs Part II), so we track the
# active Part alongside item_number to keep those sections from colliding.
PART_HEADING_RE = re.compile(
    r"^PART\s+([IVXLC]+)\b",
    re.IGNORECASE,
)
MAX_PART_HEADING_CHARS = 60
DEFAULT_PART = ""  # unknown/not-yet-seen a Part marker (e.g. most 10-Ks)

POST_ITEM_BOUNDARY_COMPACT_RE = re.compile(
    r"^(?:"
    r"SIGNATURES?"
    r"|EXHIBIT(?:INDEX|LISTING)"
    r"|EXHIBITS"
    r"|INDEXTOEXHIBITS"
    r"|POWEROFATTORNEY"
    r")$",
    re.IGNORECASE,
)

INTEGER_FLOAT_SUFFIX_RE = re.compile(r"^-?\d+\.0$")

BLOCK_TAGS = frozenset({"div", "p", "h1", "h2", "h3", "h4", "h5", "h6", "li"})
HEADING_TAGS = frozenset({"div", "p", "h1", "h2", "h3", "h4", "h5", "h6", "span"})

MAX_HEADING_CHARS = 200
MIN_GENERIC_HEADING_CHARS = 4
MAX_GENERIC_HEADING_CHARS = 120
MAX_GENERIC_HEADING_WORDS = 12

PREAMBLE_ITEM_NUMBER = ""
PREAMBLE_SECTION_TITLE = "Preamble"

_ITEM_BUFFER_TITLE = "__item__"


@dataclass(frozen=True)
class SectionInfo:
    """Canonical section identity for grouping; title is for display/citation.

    `part` disambiguates 10-Q Item numbers that repeat across Part I and
    Part II (e.g. Part I Item 1 = FINANCIAL STATEMENTS, Part II Item 1 =
    LEGAL PROCEEDINGS) — without it they'd collide under the same key.
    """

    item_number: str
    section_title: str
    part: str = DEFAULT_PART


@dataclass
class NarrativeSection:
    """One SEC Item's narrative body text (tables excluded)."""

    item_number: str
    section_title: str
    text: str
    part: str = DEFAULT_PART


@dataclass
class ExtractedTable:
    """One HTML table with document position and enclosing Item section."""

    item_number: str
    section_title: str
    position: int
    dataframe: pd.DataFrame
    part: str = DEFAULT_PART


def serialize_table(df: pd.DataFrame) -> str:
    """Format a table as one line per row for RAG chunks.

    Colspan duplication: HTML tables with merged cells (colspan > 1) are
    flattened by pandas.read_html into repeated values. Collapsed here.

    Integer float suffix: read_html may infer numeric dtypes for year-like
    header cells ("2024" -> "2024.0"). Stripped before dedup.

    # TODO: multi-level header tables not yet handled, see project doc Issue 4 follow-up
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
    normalized = [_normalize_cell_value(v) for v in values]
    collapsed: list[str] = []
    for cell in normalized:
        if collapsed and cell == collapsed[-1]:
            continue
        collapsed.append(cell)
    return collapsed


def _normalize_cell_value(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if INTEGER_FLOAT_SUFFIX_RE.match(text):
        return text[:-2]
    return text


def _first_nonempty_label(values: list[str]) -> tuple[str, list[str]]:
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
        html = self.filepath.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(html, "lxml")
        ordered_tags = soup.find_all(True)
        tag_index = {tag: idx for idx, tag in enumerate(ordered_tags)}

        section_starts = self._find_section_starts(soup, tag_index)
        section_for_index = self._build_section_lookup(section_starts, len(ordered_tags))

        sections = self._extract_narrative(soup, tag_index, section_for_index)
        tables = self._extract_tables(soup, tag_index, section_for_index)
        return sections, tables

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()

    @classmethod
    def _parse_item_heading(cls, text: str) -> Optional[SectionInfo]:
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
        return re.sub(r"[^A-Za-z]", "", cls._normalize_text(text)).upper()

    @classmethod
    def _parse_post_item_boundary(cls, text: str) -> Optional[SectionInfo]:
        normalized = cls._normalize_text(text)
        if len(normalized) > MAX_HEADING_CHARS:
            return None
        compact = cls._compact_boundary_text(normalized)
        if not POST_ITEM_BOUNDARY_COMPACT_RE.match(compact):
            return None
        return SectionInfo(item_number=PREAMBLE_ITEM_NUMBER, section_title=normalized)

    @classmethod
    def _parse_part_heading(cls, text: str) -> Optional[str]:
        """Return the roman-numeral Part ('I', 'II', ...) or None."""
        normalized = cls._normalize_text(text)
        if len(normalized) > MAX_PART_HEADING_CHARS:
            return None
        match = PART_HEADING_RE.match(normalized)
        if not match:
            return None
        return match.group(1).upper()

    @classmethod
    def _is_styled_heading(cls, tag: Tag) -> bool:
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
        normalized = cls._normalize_text(text)
        return re.sub(r"^\d+\s+", "", normalized)

    @classmethod
    def _parse_generic_heading(cls, text: str) -> Optional[SectionInfo]:
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
        if tag.name != "a":
            return False
        return bool(re.match(r"^Item\s+\d+[A-Za-z]?\.?$", normalized_text, re.I))

    @classmethod
    def _looks_like_heading(cls, tag: Tag, normalized_text: str) -> bool:
        """True when this tag is a real 'Item N.' section heading.

        FIX: styling is now required unconditionally (bold, h1-h6, or
        all-caps per `_is_styled_heading`) — not just when the Item number
        has no title text after it. Previously a titled false-positive like
        "Item 7A. of the registrant's Annual Report on Form 10-K for 2025."
        (a plain-text cross-reference, not a heading) slipped through
        because the style check was skipped whenever *any* title text
        followed the item number.
        """
        section_info = cls._parse_item_heading(normalized_text)
        if not section_info:
            return False
        if cls._is_toc_anchor(tag, normalized_text):
            return False
        if tag.name not in HEADING_TAGS:
            return False
        if tag.name == "span" and tag.find_parent(["div", "p"], recursive=False):
            return False
        if not cls._is_styled_heading(tag):
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

    @classmethod
    def _looks_like_part_heading(cls, tag: Tag, normalized_text: str) -> bool:
        if cls._parse_part_heading(normalized_text) is None:
            return False
        if tag.find_parent("table"):
            return False
        if tag.name not in HEADING_TAGS:
            return False
        if tag.name == "span" and tag.find_parent(["div", "p"], recursive=False):
            return False
        return cls._is_styled_heading(tag)

    def _find_section_starts(self, soup: BeautifulSoup, tag_index: dict[Tag, int]) -> list[tuple[int, SectionInfo]]:
        """Return (document position, section info) for Item and post-Item headings.

        Part headings (PART I / PART II) update current_part but are never
        themselves added as a section — they just tag every subsequent
        section with the right part so Part I Item 1 and Part II Item 1
        (10-Q) don't collide.
        """
        item_starts: list[tuple[int, SectionInfo]] = []
        post_item_starts: list[tuple[int, SectionInfo]] = []
        seen_at_index: dict[int, SectionInfo] = {}
        current_item_number = PREAMBLE_ITEM_NUMBER
        current_part = DEFAULT_PART

        for tag in soup.find_all(HEADING_TAGS):
            if tag.find_parent("table"):
                continue
            normalized = self._normalize_text(tag.get_text(" ", strip=True))

            if self._looks_like_part_heading(tag, normalized):
                part = self._parse_part_heading(normalized)
                if part:
                    current_part = part
                continue

            section_info: Optional[SectionInfo] = None
            is_post_item = False
            if self._looks_like_heading(tag, normalized):
                section_info = self._parse_item_heading(normalized)
                if section_info:
                    current_item_number = section_info.item_number
            elif self._looks_like_post_item_boundary(tag, normalized):
                section_info = self._parse_post_item_boundary(normalized)
                is_post_item = True
                if section_info:
                    current_item_number = PREAMBLE_ITEM_NUMBER
            elif current_item_number == PREAMBLE_ITEM_NUMBER and self._looks_like_generic_heading(
                tag, normalized
            ):
                section_info = self._parse_generic_heading(normalized)

            if not section_info:
                continue
            section_info = replace(section_info, part=current_part)
            idx = tag_index[tag]
            if idx in seen_at_index:
                continue
            seen_at_index[idx] = section_info
            if is_post_item:
                post_item_starts.append((idx, section_info))
            else:
                item_starts.append((idx, section_info))

        starts = item_starts + post_item_starts
        starts.sort(key=lambda pair: pair[0])
        return starts

    @staticmethod
    def _build_section_lookup(
        section_starts: list[tuple[int, SectionInfo]], num_tags: int
    ) -> list[SectionInfo]:
        preamble = SectionInfo(
            item_number=PREAMBLE_ITEM_NUMBER,
            section_title=PREAMBLE_SECTION_TITLE,
            part=DEFAULT_PART,
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

    @classmethod
    def _is_leaf_block(cls, tag: Tag) -> bool:
        if tag.name not in BLOCK_TAGS:
            return False
        return tag.find(BLOCK_TAGS, recursive=False) is None

    @staticmethod
    def _buffer_key(info: SectionInfo) -> tuple[str, str, str]:
        """Bucket key for narrative text; item sections merge, unnumbered do not.

        FIX: `part` is now the leading component. Without it, a 10-Q's
        Part I "Item 1. FINANCIAL STATEMENTS" and Part II "Item 1. LEGAL
        PROCEEDINGS" both reduced to the same key and their text was
        silently merged into one section under one title.
        """
        if info.item_number:
            return (info.part, info.item_number, _ITEM_BUFFER_TITLE)
        return (info.part, PREAMBLE_ITEM_NUMBER, info.section_title)

    def _extract_narrative(
        self,
        soup: BeautifulSoup,
        tag_index: dict[Tag, int],
        section_for_index: list[SectionInfo],
    ) -> list[NarrativeSection]:
        buffers: dict[tuple[str, str, str], list[str]] = {}
        titles: dict[tuple[str, str, str], str] = {}
        parts: dict[tuple[str, str, str], str] = {}
        order: list[tuple[str, str, str]] = []
        current_item_number = PREAMBLE_ITEM_NUMBER
        current_part = DEFAULT_PART

        def activate(section_info: SectionInfo) -> None:
            key = self._buffer_key(section_info)
            if key not in buffers:
                buffers[key] = []
                titles[key] = section_info.section_title
                parts[key] = section_info.part
                order.append(key)

        activate(
            SectionInfo(
                item_number=PREAMBLE_ITEM_NUMBER,
                section_title=PREAMBLE_SECTION_TITLE,
                part=DEFAULT_PART,
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

            if self._looks_like_part_heading(tag, normalized):
                part = self._parse_part_heading(normalized)
                if part:
                    current_part = part
                continue

            if self._looks_like_heading(tag, normalized):
                section_info = self._parse_item_heading(normalized)
                if section_info:
                    section_info = replace(section_info, part=current_part)
                    activate(section_info)
                    current_item_number = section_info.item_number
                continue

            if self._looks_like_post_item_boundary(tag, normalized):
                boundary = self._parse_post_item_boundary(normalized)
                if boundary:
                    boundary = replace(boundary, part=current_part)
                    activate(boundary)
                    current_item_number = PREAMBLE_ITEM_NUMBER
                continue

            if current_item_number == PREAMBLE_ITEM_NUMBER and self._looks_like_generic_heading(
                tag, normalized
            ):
                generic = self._parse_generic_heading(normalized)
                if generic:
                    generic = replace(generic, part=current_part)
                    activate(generic)
                continue

            idx = tag_index[tag]
            section = section_for_index[idx]
            activate(section)
            buffers[self._buffer_key(section)].append(normalized)

        return [
            NarrativeSection(
                item_number=key[1],
                section_title=titles[key],
                text="\n\n".join(buffers[key]),
                part=parts[key],
            )
            for key in order
            if buffers[key]
        ]

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
                    part=section.part,
                )
            )

        return extracted