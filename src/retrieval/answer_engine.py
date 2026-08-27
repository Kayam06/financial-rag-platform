"""
Phase 4 — Answer engine.

Retrieves relevant chunks, builds a citation-forcing prompt, and generates
an answer that:
  - only uses retrieved context (no outside knowledge)
  - cites ticker / form / item / part for every claim
  - explicitly refuses if the retrieved context doesn't contain the answer
"""

from src.retrieval.retriever import Retriever, RetrievedChunk
from src.retrieval.llm_clients import FallbackLLM

SYSTEM_PROMPT = """You are a financial research assistant. You answer questions
ONLY using the excerpts provided below, which come from real SEC filings
(10-K/10-Q). Follow these rules strictly:

1. Only use information present in the excerpts. Do not use outside knowledge.
2. For every factual claim, cite the source in this format:
   [TICKER, FORM, Item ITEM_NUMBER, Part PART]
3. If the excerpts do not contain enough information to answer the question,
   say exactly: "I don't have enough information in the retrieved filings to
   answer this." Do not guess or fill in gaps.
4. Be concise and precise, especially with numbers — do not round or alter
   any figures from the source text.
"""


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    context_blocks = []
    for c in chunks:
        header = f"[{c.ticker}, {c.form}, Item {c.item_number}, Part {c.part}, {c.section_title}]"
        context_blocks.append(f"{header}\n{c.text}")
    context = "\n\n---\n\n".join(context_blocks)

    return f"""{SYSTEM_PROMPT}

RETRIEVED EXCERPTS:
{context}

QUESTION: {question}

ANSWER:"""


class AnswerEngine:
    def __init__(self):
        self.retriever = Retriever()
        self.llm = FallbackLLM()

    def ask(
        self,
        question: str,
        top_k: int = 8,
        ticker_filter: str | None = None,
    ) -> dict:
        chunks = self.retriever.search(question, top_k=top_k, ticker_filter=ticker_filter)

        if not chunks:
            return {
                "answer": "I don't have enough information in the retrieved filings to answer this.",
                "sources_used": "none",
                "citations": [],
            }

        prompt = build_prompt(question, chunks)
        answer_text, source_used = self.llm.generate(prompt)

        citations = [
            {
                "ticker": c.ticker,
                "form": c.form,
                "item_number": c.item_number,
                "part": c.part,
                "section_title": c.section_title,
                "filing_date": c.filing_date,
                "score": round(c.score, 4),
            }
            for c in chunks
        ]

        return {
            "answer": answer_text.strip(),
            "sources_used": source_used,
            "citations": citations,
        }