"""
Phase 7 — Streamlit demo app for Streamlit Community Cloud.

Two tabs:
  1. Ask a Question — live RAG Q&A against the indexed SEC filings,
     via AnswerEngine (Phase 4).
  2. Extraction Accuracy — reads the saved Phase 5 validation results
     and shows per-filing / overall XBRL-matched accuracy.

Secrets (GEMINI_API_KEY, SEC_USER_AGENT) are read from Streamlit's
secrets manager (st.secrets) when deployed on Streamlit Community
Cloud, and copied into os.environ so the existing os.getenv()-based
config code (src/config.py) works unchanged. For local development,
a normal .env file (loaded via python-dotenv, as before) still works
fine — st.secrets is simply empty/unavailable locally unless you also
create a local .streamlit/secrets.toml, which is optional.
"""

import json
import os
from pathlib import Path

import streamlit as st

# Copy Streamlit secrets into the environment BEFORE importing anything
# that reads os.getenv() at import time (AnswerEngine -> Retriever ->
# Embedder, and config.py's validation). Wrapped in try/except because
# st.secrets raises if no secrets.toml exists at all (e.g. fresh local
# clone with only a .env file) -- that's fine, just skip in that case.
try:
    for key, value in st.secrets.items():
        os.environ.setdefault(key, str(value))
except Exception:
    pass

from src.retrieval.answer_engine import AnswerEngine

TICKERS = ["All", "AAPL", "MSFT", "JPM", "WMT", "JNJ", "XOM", "KO", "BA"]
VALIDATION_RESULTS_PATH = Path("data/processed/validation_results.json")

st.set_page_config(
    page_title="Financial Document Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling — a restrained, professional dark theme with a gradient header,
# bordered "panel" treatment for controls, and card-style answer/metrics.
# Pure CSS injected via st.markdown; no new dependency.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main .block-container {
        padding-top: 1.5rem;
        max-width: 1100px;
    }

    /* Header banner */
    .hero-banner {
        background: linear-gradient(135deg, #1a1d24 0%, #23181b 100%);
        border: 1px solid #2d323c;
        border-radius: 12px;
        padding: 1.75rem 2rem;
        margin-bottom: 1.75rem;
    }
    .hero-banner h1 {
        font-weight: 800 !important;
        letter-spacing: -0.03em;
        font-size: 2rem !important;
        margin: 0 0 0.5rem 0 !important;
        background: linear-gradient(90deg, #ffffff, #ff8a8a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-banner .subtitle {
        color: #9CA3AF;
        font-size: 0.98rem;
        line-height: 1.55;
        margin: 0;
    }
    .hero-banner .subtitle b {
        color: #E5E7EB;
    }

    /* Controls panel */
    .controls-panel {
        background: #15171c;
        border: 1px solid #262a33;
        border-radius: 10px;
        padding: 1rem 1.25rem 0.5rem 1.25rem;
        margin-bottom: 1rem;
    }

    /* Answer card */
    .answer-card {
        background: #1a1d24;
        border: 1px solid #2d323c;
        border-left: 4px solid #FF4B4B;
        border-radius: 8px;
        padding: 1.25rem 1.5rem;
        margin-top: 0.5rem;
    }
    .answer-card p {
        font-size: 1.05rem;
        line-height: 1.6;
        margin-bottom: 0;
        color: #E5E7EB;
    }

    .source-badge {
        display: block;
        background: #191c22;
        border: 1px solid #2a2f38;
        border-radius: 6px;
        padding: 0.6rem 0.9rem;
        margin-bottom: 0.5rem;
        font-size: 0.86rem;
    }
    .ticker-tag {
        color: #FF6B6B;
        font-weight: 700;
    }

    .section-header {
        font-size: 1.05rem;
        font-weight: 700;
        margin-top: 1.75rem;
        margin-bottom: 0.6rem;
        color: #F3F4F6;
        letter-spacing: -0.01em;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.75rem !important;
    }

    /* Metric cards on Accuracy tab */
    [data-testid="stMetric"] {
        background: #15171c;
        border: 1px solid #262a33;
        border-radius: 10px;
        padding: 1rem 1.25rem;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        font-size: 0.95rem;
    }

    .footer-note {
        color: #6B7280;
        font-size: 0.85rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #262a33;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_engine():
    # Loads the FAISS index + embedder once per app instance, not per
    # request/rerun -- this is the expensive part (model + index in memory).
    return AnswerEngine()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="hero-banner">'
    "<h1>📊 Financial Document Intelligence Platform</h1>"
    '<div class="subtitle">Ask plain-English questions about SEC filings '
    "(10-K / 10-Q) for <b>AAPL, MSFT, JPM, WMT, JNJ, XOM, KO, BA</b> — answers "
    "are sourced and cited directly from the filings, with extracted figures "
    "cross-validated against SEC's own official XBRL data.</div>"
    "</div>",
    unsafe_allow_html=True,
)

tab_ask, tab_accuracy = st.tabs(["💬  Ask a Question", "✅  Extraction Accuracy"])

# ---------------------------------------------------------------------------
# Tab 1 — Ask a Question
# ---------------------------------------------------------------------------
with tab_ask:
    st.markdown('<div class="controls-panel">', unsafe_allow_html=True)
    left, right = st.columns([2.2, 1], gap="large")

    with left:
        question = st.text_area(
            "Your question",
            placeholder="e.g. What was Apple's total revenue in their most recent 10-Q?",
            height=90,
            label_visibility="collapsed",
        )
        ask_clicked = st.button("Ask", type="primary", use_container_width=False)

    with right:
        st.markdown("**Filters**")
        ticker = st.selectbox("Ticker", TICKERS, index=0, label_visibility="collapsed")
        top_k = st.slider("Chunks to retrieve", min_value=3, max_value=15, value=8)
    st.markdown("</div>", unsafe_allow_html=True)

    if ask_clicked:
        if not question or not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving relevant filings and generating an answer..."):
                engine = load_engine()
                ticker_filter = None if ticker == "All" else ticker
                try:
                    result = engine.ask(
                        question, top_k=top_k, ticker_filter=ticker_filter
                    )
                except Exception:
                    st.error(
                        "The Gemini API is temporarily unavailable (rate limit or "
                        "outage) and this deployed demo has no local Ollama fallback "
                        "to run on — that fallback only works when running the app "
                        "on a machine with Ollama installed. Please try again in a "
                        "minute or two."
                    )
                    st.stop()

            st.markdown(
                '<div class="section-header">Answer</div>', unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="answer-card"><p>{result["answer"]}</p></div>',
                unsafe_allow_html=True,
            )
            st.caption(f"Answered via **{result['sources_used']}**")

            citations = result["citations"]
            if citations:
                with st.expander(f"📎 View {len(citations)} sources used"):
                    for c in citations:
                        st.markdown(
                            f'<div class="source-badge">'
                            f'<span class="ticker-tag">{c["ticker"]}</span> · {c["form"]} · '
                            f'Item {c["item_number"] or "—"}, Part {c["part"] or "—"} · '
                            f'<i>{c["section_title"]}</i><br>'
                            f'<span style="color:#6B7280;">filed {c["filing_date"]} · '
                            f'relevance {c["score"]}</span>'
                            f"</div>",
                            unsafe_allow_html=True,
                        )
            else:
                st.caption("No sources retrieved.")

# ---------------------------------------------------------------------------
# Tab 2 — Extraction Accuracy
# ---------------------------------------------------------------------------
with tab_accuracy:
    st.markdown(
        "Line items (revenue, net income, total assets) extracted by the LLM "
        "from each filing, automatically checked against SEC's official XBRL "
        "CompanyFacts data — the government's own reported numbers."
    )

    if not VALIDATION_RESULTS_PATH.exists():
        st.info(
            "No validation results found yet. Run "
            "`python -m src.extraction.run_validation` to generate them."
        )
    else:
        with open(VALIDATION_RESULTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        overall = data.get("overall_accuracy")
        total_matched = data.get("total_matched", 0)
        total_scored = data.get("total_scored", 0)
        filings = data.get("filings", [])
        overall_str = f"{overall:.1%}" if overall is not None else "n/a"

        m1, m2, m3 = st.columns(3)
        m1.metric("Overall accuracy", overall_str)
        m2.metric("Line items matched", f"{total_matched} / {total_scored}")
        m3.metric("Filings evaluated", len(filings))

        st.markdown(
            '<div class="section-header">Per-filing results</div>',
            unsafe_allow_html=True,
        )

        rows = []
        for filing in filings:
            acc = filing.get("accuracy")
            acc_str = f"{acc:.0%}" if acc is not None else "n/a"
            rows.append(
                {
                    "Ticker": filing.get("ticker"),
                    "Form": filing.get("form"),
                    "Report Date": filing.get("report_date"),
                    "Accuracy": acc_str,
                    "Matched/Scored": f"{filing.get('matched_count', 0)}/{filing.get('scored_count', 0)}",
                    "LLM Source": filing.get("llm_source"),
                }
            )

        st.dataframe(rows, use_container_width=True, hide_index=True)

st.markdown(
    '<div class="footer-note">Built entirely on free-tier tools — SEC EDGAR, '
    "self-hosted embeddings, FAISS, and Gemini/Ollama — with zero paid "
    "infrastructure.</div>",
    unsafe_allow_html=True,
)
