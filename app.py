"""
Phase 7 — Gradio demo app for Hugging Face Spaces.

Two tabs:
  1. Ask a Question — live RAG Q&A against the indexed SEC filings,
     via AnswerEngine (Phase 4).
  2. Extraction Accuracy — reads the saved Phase 5 validation results
     and shows per-filing / overall XBRL-matched accuracy.

Secrets (GEMINI_API_KEY, SEC_USER_AGENT) are read from environment
variables, which HF Spaces injects from its Secrets UI — nothing
sensitive is hardcoded or committed here.
"""

import json
from pathlib import Path

import gradio as gr

from src.retrieval.answer_engine import AnswerEngine

TICKERS = ["All", "AAPL", "MSFT", "JPM", "WMT", "JNJ", "XOM", "KO", "BA"]
VALIDATION_RESULTS_PATH = Path("data/processed/validation_results.json")

# AnswerEngine loads the FAISS index + embedder once at import time —
# expensive, so it's built a single time at module load, not per-request.
_engine = AnswerEngine()


def answer_question(question: str, ticker: str, top_k: int):
    if not question or not question.strip():
        return "Please enter a question.", ""

    ticker_filter = None if ticker == "All" else ticker
    result = _engine.ask(question, top_k=top_k, ticker_filter=ticker_filter)

    answer = result["answer"]
    citations = result["citations"]

    if not citations:
        citations_md = "_No sources retrieved._"
    else:
        lines = [
            f"- **{c['ticker']}** {c['form']} — Item {c['item_number']}, "
            f"Part {c['part']} — *{c['section_title']}* "
            f"(filed {c['filing_date']}, relevance {c['score']})"
            for c in citations
        ]
        citations_md = "\n".join(lines)

    footer = f"\n\n---\n*Answered via: {result['sources_used']}*"
    return answer + footer, citations_md


def load_accuracy_view():
    if not VALIDATION_RESULTS_PATH.exists():
        return (
            "No validation results found yet. Run "
            "`python -m src.extraction.run_validation` to generate them.",
            [],
        )

    with open(VALIDATION_RESULTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    overall = data.get("overall_accuracy")
    overall_str = f"{overall:.1%}" if overall is not None else "n/a"
    total_matched = data.get("total_matched", 0)
    total_scored = data.get("total_scored", 0)

    summary = (
        f"### Overall accuracy: **{overall_str}** "
        f"({total_matched}/{total_scored} line items matched SEC's official XBRL data)"
    )

    rows = []
    for filing in data.get("filings", []):
        acc = filing.get("accuracy")
        acc_str = f"{acc:.0%}" if acc is not None else "n/a"
        rows.append([
            filing.get("ticker"),
            filing.get("form"),
            filing.get("report_date"),
            acc_str,
            f"{filing.get('matched_count', 0)}/{filing.get('scored_count', 0)}",
            filing.get("llm_source"),
        ])

    return summary, rows


with gr.Blocks(title="Financial Document Intelligence Platform") as demo:
    gr.Markdown(
        "# Financial Document Intelligence Platform\n"
        "Ask plain-English questions about SEC filings (10-K/10-Q) for "
        "AAPL, MSFT, JPM, WMT, JNJ, XOM, KO, and BA — answers are sourced "
        "and cited directly from the actual filings, with extracted "
        "figures cross-validated against SEC's own official XBRL data."
    )

    with gr.Tab("Ask a Question"):
        with gr.Row():
            question_box = gr.Textbox(
                label="Your question",
                placeholder="e.g. What was Apple's total revenue in their most recent 10-Q?",
                lines=2,
            )
        with gr.Row():
            ticker_dropdown = gr.Dropdown(TICKERS, value="All", label="Filter by ticker")
            top_k_slider = gr.Slider(3, 15, value=8, step=1, label="Chunks to retrieve")
        ask_button = gr.Button("Ask", variant="primary")
        answer_output = gr.Markdown(label="Answer")
        citations_output = gr.Markdown(label="Sources")

        ask_button.click(
            fn=answer_question,
            inputs=[question_box, ticker_dropdown, top_k_slider],
            outputs=[answer_output, citations_output],
        )

    with gr.Tab("Extraction Accuracy"):
        gr.Markdown(
            "Cross-validation results: line items (revenue, net income, "
            "total assets) extracted by the LLM from each filing, checked "
            "against SEC's official XBRL CompanyFacts data."
        )
        refresh_button = gr.Button("Load results")
        accuracy_summary = gr.Markdown()
        accuracy_table = gr.Dataframe(
            headers=["Ticker", "Form", "Report Date", "Accuracy", "Matched/Scored", "LLM Source"],
            label="Per-filing results",
        )
        refresh_button.click(fn=load_accuracy_view, outputs=[accuracy_summary, accuracy_table])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)