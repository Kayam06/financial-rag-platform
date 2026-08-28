"""
Phase 5 — LLM structured line-item extraction.

Retrieves the relevant financial-statement chunks for one filing (via
Phase 4's Retriever, ticker-filtered) and asks the LLM to extract a fixed
set of line items into strict JSON. Calls FallbackLLM directly (not
AnswerEngine) — extraction needs raw JSON out, not a citation-formatted
chat answer.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from src.retrieval.retriever import Retriever
from src.retrieval.llm_clients import FallbackLLM

EXTRACTION_QUERIES = {
    "total_revenue": "total net sales revenue for the period",
    "net_income": "net income for the period",
    "total_assets": "total assets balance sheet",
}

EXTRACTION_PROMPT_TEMPLATE = """You are extracting exact financial figures from SEC filing excerpts below.
Extract ONLY the following three line items for {ticker}'s {form} filed for the period ending {report_date}:

1. total_revenue — total net sales / total revenue for the period (NOT a year-ago comparison figure)
2. net_income — net income (or net loss, as a negative number) for the period
3. total_assets — total assets as of the balance sheet date

Report each figure as a plain number in actual dollars (e.g. 109417000000, not "109417" or "$109.4B").
If a value cannot be determined from the excerpts below, use null for that field.
Also report which period each duration figure (total_revenue, net_income) covers: "quarter" (single
3-month period) or "ytd" (year-to-date cumulative), as a "period" field. total_assets is a point-in-time
balance and has no period field.

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
    def __init__(self, retriever: Optional[Retriever] = None,
                 llm: Optional[FallbackLLM] = None):
        self.retriever = retriever or Retriever()
        self.llm = llm or FallbackLLM()

    def _gather_context(self, ticker: str, form: str, report_date: str,
                         top_k: int = 6) -> str:
        seen = set()
        chunks = []
        for query_text in EXTRACTION_QUERIES.values():
            results = self.retriever.search(query_text, top_k=top_k, ticker_filter=ticker)
            for r in results:
                # Only keep chunks from this exact filing (form + report_date) —
                # a ticker has multiple filings indexed, ticker_filter alone
                # isn't enough to scope to one.
                if r.form != form or r.report_date != report_date:
                    continue
                key = (r.section_title, r.item_number, r.part, r.text[:50])
                if key in seen:
                    continue
                seen.add(key)
                chunks.append(r)

        if not chunks:
            raise ExtractionError(
                f"No chunks retrieved for {ticker} {form} {report_date} — "
                "check that ticker/form/report_date exactly match values in "
                "chunk_metadata.jsonl."
            )

        return "\n\n---\n\n".join(
            f"[{c.section_title} | Part {c.part} | Item {c.item_number}]\n{c.text}"
            for c in chunks
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