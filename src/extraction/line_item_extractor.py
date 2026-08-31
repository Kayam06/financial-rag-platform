"""
Phase 5 — LLM structured line-item extraction.

Unlike Phase 4 (open-ended Q&A over the whole corpus, where embedding
search is the right tool because we don't know which chunk has the
answer), Phase 5 always starts already knowing the exact filing
(ticker + form + report_date). So instead of embedding search, this
pulls chunks directly out of chunk_metadata.jsonl by exact filing match.

Chunk selection strategy (row-label content matching, not section_title):
Direct inspection during the Phase 5 session showed section_title is
unreliable across filers — AAPL/KO have clean per-statement titles
("CONSOLIDATED STATEMENTS OF OPERATIONS"), but MSFT's Item 8 financial
statements are split into many small table fragments that ALL share one
generic section_title ("Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY
DATA"), with no per-table title to match on. What IS reliable across every
filer checked (AAPL, MSFT) is that the actual row labels ("Net income",
"Total assets", "Revenue") appear verbatim somewhere in the table's own
content, regardless of chunk title. So instead of matching titles, this
scans each table chunk's full content for those row-label phrases and
gathers every matching fragment for the filing — works for both "one big
table" filers (AAPL) and "many small fragments" filers (MSFT) the same way.

BA/JNJ, and JPM's true audited statements, still fail this check entirely
— confirmed via direct inspection to be a Phase 2 parsing gap (garbage/
boilerplate table content, or MD&A-only tables under a misleading section
title), not something this matching strategy can work around. Documented
in PROJECT_CONTEXT.md as a known limitation for a future Phase 2 session.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from src.retrieval.llm_clients import FallbackLLM
from src.retrieval.retriever import METADATA_PATH

# Row-label phrases that reliably appear as literal text inside the real
# financial-statement tables, regardless of what section_title says.
LINE_ITEM_ROW_LABELS = ["NET INCOME", "TOTAL ASSETS", "REVENUE", "NET EARNINGS"]

# Some filers (JNJ) use "Sales to customers" instead of "Revenue" — but that
# phrase alone also appears in dozens of unrelated per-product/segment
# breakdown tables. Only treat it as a real income-statement match when it
# co-occurs with "Cost of products sold" and "Gross profit" in the same
# chunk, which narrows it down to the actual statement.
REVENUE_STATEMENT_COOCCURRENCE = [
    "SALES TO CUSTOMERS",
    "COST OF PRODUCTS SOLD",
    "GROSS PROFIT",
]

EXTRACTION_PROMPT_TEMPLATE = """You are extracting exact financial figures from SEC filing excerpts below.
Extract ONLY the following three line items for {ticker}'s {form} filed for the period ending {report_date}:

1. total_revenue — total net sales / total revenue for the period (NOT a year-ago comparison figure)
2. net_income — net income (or net loss, as a negative number) for the period
3. total_assets — total assets as of the balance sheet date

Report each figure as a plain number in actual dollars, NOT in millions or thousands. If the source
table is captioned "(in millions)", multiply by 1,000,000 before reporting. For example, if a table
shows "109417" under a "(in millions)" caption, report 109417000000, not 109417.
If a value cannot be determined from the excerpts below, use null for that field.
Also report which period each duration figure (total_revenue, net_income) covers: "quarter" (single
3-month period) or "ytd" (year-to-date cumulative), as a "period" field. total_assets is a point-in-time
balance and has no period field.

Tables may show multiple fiscal years side by side — use only the column matching {report_date}'s
fiscal year, not prior-year comparatives. Tables may also repeat the same figure in different contexts
(e.g. a summary recap and the full statement) — if values agree, use that value; if they disagree,
prefer the most detailed/complete table.

Respond with ONLY valid JSON, no other text, in exactly this shape:
{{"total_revenue": {{"value": <number or null>, "period": "quarter"|"ytd"|null}},
  "net_income": {{"value": <number or null>, "period": "quarter"|"ytd"|null}},
  "total_assets": {{"value": <number or null>}}}}

FILING EXCERPTS:
{context}
"""

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


class ExtractionError(Exception):
    pass


class LineItemExtractor:
    def __init__(
        self, metadata_path: Path = METADATA_PATH, llm: Optional[FallbackLLM] = None
    ):
        self.metadata_path = metadata_path
        self.llm = llm or FallbackLLM()
        self._metadata: Optional[list[dict]] = None  # lazy-loaded, cached

    def _load_metadata(self) -> list[dict]:
        if self._metadata is None:
            rows = []
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            self._metadata = rows
        return self._metadata

    def _gather_context(self, ticker: str, form: str, report_date: str) -> str:
        filing_tables = [
            row
            for row in self._load_metadata()
            if row.get("ticker") == ticker
            and row.get("form") == form
            and row.get("report_date") == report_date
            and row.get("chunk_type") == "table"
        ]

        matches = []
        for row in filing_tables:
            content_upper = row.get("content", "").upper()
            if any(kw in content_upper for kw in LINE_ITEM_ROW_LABELS):
                matches.append(row)
            elif all(kw in content_upper for kw in REVENUE_STATEMENT_COOCCURRENCE):
                matches.append(row)

        if not matches:
            raise ExtractionError(
                f"No table chunks containing net income/total assets/revenue "
                f"row labels found for {ticker} {form} {report_date}. This "
                f"filer's financial statements likely weren't extracted "
                f"correctly in Phase 2 (see PROJECT_CONTEXT.md known limitations)."
            )

        return "\n\n---\n\n".join(
            f"[{r.get('section_title')} | Part {r.get('part')} | Item {r.get('item_number')}]\n"
            f"{r.get('content', '')}"
            for r in matches
        )

    def extract(self, ticker: str, form: str, report_date: str) -> dict:
        context = self._gather_context(ticker, form, report_date)
        prompt = EXTRACTION_PROMPT_TEMPLATE.format(
            ticker=ticker, form=form, report_date=report_date, context=context
        )
        raw_response, source = self.llm.generate(prompt)

        match = _JSON_BLOCK_RE.search(raw_response)
        if not match:
            raise ExtractionError(
                f"LLM ({source}) did not return parseable JSON for "
                f"{ticker} {form} {report_date}. Raw response: {raw_response[:300]}"
            )
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as e:
            raise ExtractionError(
                f"LLM ({source}) returned malformed JSON for "
                f"{ticker} {form} {report_date}: {e}. Raw: {match.group(0)[:300]}"
            )

        parsed["_llm_source"] = source
        return parsed
