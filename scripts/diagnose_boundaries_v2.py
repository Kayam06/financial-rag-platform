"""Diagnose XOM Item 16 boundary runaway and repeated boilerplate headings.

Run with: python -m scripts.diagnose_boundaries_v2
Requires no network. Reads already-downloaded filings from data/raw/<TICKER>/*.htm.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from bs4 import BeautifulSoup

from src.parsing.html_parser import FilingParser, HEADING_TAGS


def diagnose_item16_runaway(filepath: Path) -> None:
    print(f"\n=== ITEM 16 CHECK: {filepath.name} ===")
    html = filepath.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "lxml")
    ordered_tags = soup.find_all(True)
    tag_index = {tag: idx for idx, tag in enumerate(ordered_tags)}

    parser = FilingParser(filepath)
    section_starts = parser._find_section_starts(soup, tag_index)

    item16_idx = None
    for idx, info in section_starts:
        if info.item_number == "16":
            item16_idx = idx
            break

    if item16_idx is None:
        print("No Item 16 boundary detected at all in this filing.")
        return

    print(f"Item 16 starts at tag idx {item16_idx} of {len(ordered_tags)} total tags")
    print("\nBoundaries from Item 16 onward:")
    for idx, info in section_starts:
        if idx >= item16_idx:
            print(f"  idx={idx:>7}  item_number={info.item_number!r:6}  section_title={info.section_title!r}")

    print("\nCandidate post-Item marker text after Item 16 (checking why it may not match):")
    count = 0
    for tag in ordered_tags[item16_idx:]:
        if tag.name not in HEADING_TAGS:
            continue
        if tag.find_parent("table"):
            continue
        text = FilingParser._normalize_text(tag.get_text(" ", strip=True))
        if not text or len(text) > 150:
            continue
        compact = FilingParser._compact_boundary_text(text)
        if any(k in compact for k in ("SIGNATURE", "EXHIBIT", "POWEROFATTORNEY", "INDEXTOEXHIBITS")):
            matched = FilingParser._parse_post_item_boundary(text) is not None
            print(f"  idx={tag_index[tag]:>7}  matched={matched}  text={text[:100]!r}")
            count += 1
        if count > 30:
            break


def diagnose_repeated_headings(filepath: Path, ticker: str) -> None:
    print(f"\n=== REPEATED HEADING CHECK: {filepath.name} ({ticker}) ===")
    html = filepath.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "lxml")
    ordered_tags = soup.find_all(True)
    tag_index = {tag: idx for idx, tag in enumerate(ordered_tags)}

    parser = FilingParser(filepath)
    section_starts = parser._find_section_starts(soup, tag_index)

    title_counts = Counter(info.section_title for _, info in section_starts if info.item_number == "")
    repeated = [(t, c) for t, c in title_counts.items() if c >= 3]
    repeated.sort(key=lambda x: -x[1])

    print("Repeated generic section_titles (>=3 occurrences):")
    for title, count in repeated[:10]:
        print(f"  {count:>4}x  {title!r}")

    if repeated:
        target_title = repeated[0][0]
        print(f"\nTag context for first 5 occurrences of {target_title!r}:")
        shown = 0
        for tag in soup.find_all(HEADING_TAGS):
            if tag.find_parent("table"):
                continue
            text = FilingParser._normalize_text(tag.get_text(" ", strip=True))
            if text == target_title:
                idx = tag_index[tag]
                print(f"  idx={idx:>7}  tag=<{tag.name} {dict(tag.attrs)}>")
                shown += 1
            if shown >= 5:
                break

def diagnose_full_item_sequence(filepath: Path) -> None:
    print(f"\n=== FULL ITEM SEQUENCE: {filepath.name} ===")
    html = filepath.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "lxml")
    ordered_tags = soup.find_all(True)
    tag_index = {tag: idx for idx, tag in enumerate(ordered_tags)}

    parser = FilingParser(filepath)
    section_starts = parser._find_section_starts(soup, tag_index)

    for idx, info in section_starts:
        if info.item_number:
            print(f"  idx={idx:>7}  Item {info.item_number:<4} section_title={info.section_title!r}")

def diagnose_toc_vs_real_heading(filepath: Path) -> None:
    """Dump raw tag markup around the first 'Item 1' match, and search for a
    second later occurrence of Item 1-style text, to distinguish a TOC entry
    from a real section heading.
    """
    print(f"\n=== TOC vs REAL HEADING CHECK: {filepath.name} ===")
    html = filepath.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "lxml")
    ordered_tags = soup.find_all(True)
    tag_index = {tag: idx for idx, tag in enumerate(ordered_tags)}

    parser = FilingParser(filepath)
    section_starts = parser._find_section_starts(soup, tag_index)

    first_item1 = next((idx for idx, info in section_starts if info.item_number == "1"), None)
    if first_item1 is None:
        print("No Item 1 detected at all.")
        return

    tag = ordered_tags[first_item1]
    print(f"First detected 'Item 1' at idx={first_item1}")
    print(f"Tag: <{tag.name} {dict(tag.attrs)}>")
    print(f"Surrounding markup:\n{tag.parent}"[:800])

    print("\nSearching for ANY other 'Item 1.' or 'Item 1 ' text later in the document (not just detected headings):")
    count = 0
    for t in ordered_tags[first_item1 + 50:]:
        text = FilingParser._normalize_text(t.get_text(" ", strip=True))
        if text.upper().startswith("ITEM 1.") or text.upper().startswith("ITEM 1 "):
            idx = tag_index[t]
            is_heading = parser._looks_like_heading(t, text)
            styled = parser._is_styled_heading(t)
            print(f"  idx={idx:>7}  looks_like_heading={is_heading}  is_styled={styled}  tag=<{t.name}>  text={text[:90]!r}")
            count += 1
        if count >= 10:
            break



if __name__ == "__main__":
    xom_dir = Path("data/raw/XOM")
    xom_filings = sorted(xom_dir.glob("*.htm")) if xom_dir.exists() else []
    if not xom_filings:
        print(f"No .htm files found under {xom_dir} — check the folder name/path.")
    for fp in xom_filings:
        diagnose_item16_runaway(fp)

    for fp in xom_filings:
        diagnose_full_item_sequence(fp)

    for fp in xom_filings:
        diagnose_toc_vs_real_heading(fp)

    for ticker in ("KO", "BA", "MSFT", "JPM"):
        tdir = Path(f"data/raw/{ticker}")
        matches = sorted(tdir.glob("*.htm")) if tdir.exists() else []
        if matches:
            diagnose_repeated_headings(matches[0], ticker)
