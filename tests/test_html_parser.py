"""
Mocked / fixture-based — no network access and no dependency on real filing files.
"""
import pandas as pd

from src.parsing.html_parser import FilingParser, ITEM_HEADING_RE, serialize_table


SAMPLE_FILING_HTML = """
<html><body>
  <div>
    <table><tr><td><a href="#item1">Item 1.</a></td><td>Financial Statements</td></tr></table>
  </div>
  <div style="font-weight:700">Item 1.&#160;&#160;Financial Statements</div>
  <p>Consolidated revenue increased 5% year over year in the first quarter.</p>
  <p>Operating margins expanded due to cost discipline across all segments.</p>
  <table>
    <tr><th>Line Item</th><th>Q1 2026</th><th>Q1 2025</th></tr>
    <tr><td>Revenue</td><td>100</td><td>95</td></tr>
    <tr><td>Net income</td><td>20</td><td>18</td></tr>
  </table>
  <div style="font-weight:700">Item 2.&#160;&#160;Management's Discussion and Analysis</div>
  <p>Results reflect strong demand in our core markets.</p>
  <p>We expect continued growth through the remainder of the fiscal year.</p>
  <div style="font-weight:700">Item 1A.&#160;&#160;Risk Factors</div>
  <p>Market volatility may adversely affect our stock price and liquidity.</p>
</body></html>
"""

COLSPAN_TABLE_HTML = """
<html><body>
  <table>
    <tr><td colspan="3">California</td></tr>
    <tr><td>Revenue</td><td>100</td><td>200</td></tr>
  </table>
</body></html>
"""

ITEM8_VARIANTS_HTML = """
<html><body>
  <div style="font-weight:700">Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA</div>
  <p>Audited statements for fiscal 2025.</p>
  <div style="font-weight:700">Item 8. Financial statements and supplementary data</div>
  <p>Notes to consolidated financial statements.</p>
  <div style="font-weight:700">Item 8. FINANCIAL STATE MENTS AND SUPPLEMENTARY DATA</div>
  <p>Schedule II valuation accounts.</p>
</body></html>
"""

ITEM16_BOUNDARY_HTML = """
<html><body>
  <div style="font-weight:700">Item 16. Form 10-K Summary</div>
  <p>None.</p>
  <p>SIGNAT URES</p>
  <p>/s/ Jane Doe</p>
  <p>Chief Executive Officer</p>
  <div style="font-weight:700">EXHIBIT INDEX</div>
  <p>Exhibit 31.1 — Certification of CEO</p>
</body></html>
"""

GENERIC_HEADING_HTML = """
<html><body>
  <p>Cover page boilerplate before any heading.</p>
  <div style="font-weight:700">INTRODUCTION</div>
  <p>MD&amp;A opening paragraph.</p>
  <div style="font-weight:700">CONSOLIDATED RESULTS OF OPERATIONS</div>
  <p>Revenue increased 5% year over year.</p>
  <div style="font-weight:700">Item 16. Form 10-K Summary</div>
  <p>None.</p>
  <p>SIGNAT URES</p>
  <p>/s/ Jane Doe</p>
</body></html>
"""


def test_item_heading_regex_matches_common_formats():
    assert ITEM_HEADING_RE.match("Item 1. Business")
    assert ITEM_HEADING_RE.match("Item 1A. Risk Factors")
    assert ITEM_HEADING_RE.match("Item 7A. Quantitative Disclosures")
    assert ITEM_HEADING_RE.match("Item 12")
    assert not ITEM_HEADING_RE.match("See Part II, Item 7 for details")


def test_parse_extracts_sections_and_tables(tmp_path):
    filing = tmp_path / "sample.htm"
    filing.write_text(SAMPLE_FILING_HTML, encoding="utf-8")

    sections, tables = FilingParser(filing).parse()

    section_titles = [s.section_title for s in sections]
    assert "Item 1. Financial Statements" in section_titles
    assert "Item 2. Management's Discussion and Analysis" in section_titles
    assert "Item 1A. Risk Factors" in section_titles

    mdna = next(s for s in sections if s.item_number == "2")
    assert mdna.section_title.startswith("Item 2.")
    assert "strong demand" in mdna.text
    assert "Consolidated revenue" not in mdna.text

    assert len(tables) == 2  # layout TOC table + financial table
    financial = [t for t in tables if "Revenue" in t.dataframe.to_string()]
    assert len(financial) == 1
    assert financial[0].item_number == "1"
    assert financial[0].section_title == "Item 1. Financial Statements"
    assert list(financial[0].dataframe.columns) == ["Line Item", "Q1 2026", "Q1 2025"]


def test_item8_heading_variants_share_item_number(tmp_path):
    filing = tmp_path / "item8.htm"
    filing.write_text(ITEM8_VARIANTS_HTML, encoding="utf-8")

    sections, _ = FilingParser(filing).parse()
    item8_sections = [s for s in sections if s.item_number == "8"]

    assert len(item8_sections) == 1
    assert "Audited statements" in item8_sections[0].text
    assert "Notes to consolidated" in item8_sections[0].text
    assert "Schedule II valuation" in item8_sections[0].text


def test_item_number_canonicalization_from_heading_text():
    variants = [
        "Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA",
        "Item 8. Financial statements and supplementary data",
        "Item 8. FINANCIAL STATE MENTS AND SUPPLEMENTARY DATA",
    ]
    numbers = {FilingParser._parse_item_heading(v).item_number for v in variants}
    assert numbers == {"8"}


def test_serialize_table_collapses_colspan_duplicates(tmp_path):
    filing = tmp_path / "colspan.htm"
    filing.write_text(COLSPAN_TABLE_HTML, encoding="utf-8")

    _, tables = FilingParser(filing).parse()
    assert len(tables) == 1

    content = serialize_table(tables[0].dataframe)
    assert "California,California" not in content
    assert content.splitlines()[0] == "California"
    assert "Revenue: 100 | 200" in content


def test_serialize_table_row_format_with_headers():
    df = pd.DataFrame(
        [["Revenue", "100", "95"], ["Net income", "20", "18"]],
        columns=["Line Item", "Q1 2026", "Q1 2025"],
    )
    content = serialize_table(df)
    assert content == "Revenue: 100 | 95\nNet income: 20 | 18"


def test_item16_content_stops_at_signatures_boundary(tmp_path):
    filing = tmp_path / "item16.htm"
    filing.write_text(ITEM16_BOUNDARY_HTML, encoding="utf-8")

    sections, _ = FilingParser(filing).parse()

    item16 = next(s for s in sections if s.item_number == "16")
    assert item16.text.strip() == "None."
    assert "/s/ Jane Doe" not in item16.text

    signatures = next(s for s in sections if s.item_number == "" and "SIGNAT" in s.section_title.upper())
    assert "/s/ Jane Doe" in signatures.text


def test_preamble_not_renamed_to_signature_when_signature_appears_later(tmp_path):
    filing = tmp_path / "generic.htm"
    filing.write_text(GENERIC_HEADING_HTML, encoding="utf-8")

    sections, _ = FilingParser(filing).parse()
    by_title = {s.section_title: s for s in sections}

    assert "Cover page boilerplate" in by_title["Preamble"].text
    assert "MD&A opening" in by_title["INTRODUCTION"].text
    assert "Revenue increased" in by_title["CONSOLIDATED RESULTS OF OPERATIONS"].text
    assert "MD&A opening" not in by_title["SIGNAT URES"].text
    assert by_title["Item 16. Form 10-K Summary"].text.strip() == "None."
    assert "/s/ Jane Doe" in by_title["SIGNAT URES"].text


def test_serialize_table_strips_integer_float_suffix_duplicates():
    df = pd.DataFrame(
        [["Years ended December 31,", "2024", 2024.0, "2023", 2023.0]],
        columns=["Label", "c1", "c2", "c3", "c4"],
    )
    content = serialize_table(df)
    assert "2024.0" not in content
    assert "2023.0" not in content
    assert content == "Years ended December 31,: 2024 | 2023"
