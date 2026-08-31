---
title: Financial Document Intelligence Platform
emoji: 📊
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Financial Document Intelligence Platform

Ask a plain-English question about a public company's financials and get an
answer sourced directly from that company's actual SEC filings (10-K/10-Q) —
with the specific numbers cross-checked against SEC's own official
structured data (XBRL), not just a plausible-sounding LLM guess.

Built entirely on free tools: no paid APIs, no cloud bill, no credit card
required to run any part of this.

## What makes this different

Most "RAG over documents" projects stop at "it answers questions about a
PDF." This one goes further: SEC EDGAR publishes every company's official
reported numbers as structured XBRL data, for free. That means every number
this pipeline extracts from a messy 10-K can be automatically checked
against SEC's own ground truth — producing a real, measured accuracy
percentage instead of just a demo that "seems to work."

**Current measured accuracy: 72.9%** of extracted line items (revenue, net
income, total assets) exactly matched SEC's official XBRL data, across all
32 filings from 8 companies (AAPL, MSFT, JPM, WMT, JNJ, XOM, KO, BA).

## Architecture

```
SEC EDGAR filings (10-K / 10-Q, free API)
        |
Parse & extract (custom Item-boundary-aware HTML parser —
                  BeautifulSoup + lxml + pandas.read_html;
                  section-aware chunking by Item number and Part)
        |
Embed & index (BGE-base-en-v1.5 embeddings, self-hosted via
                sentence-transformers; FAISS IndexFlatIP vector store)
        |
   /----------------------------\
  |                              |
Answer engine                Extract & validate
(retrieval + Gemini/Ollama,   (LLM extracts line items to JSON,
 cites source filing/         checked against SEC's XBRL
 section/part, refuses        CompanyFacts API — the ground
 if not in context)            truth)
  |                              |
Cited answers                Accuracy score
(Gradio demo / CLI)          (per-filing + overall)
```


## Tech stack — every choice here is because it's genuinely $0

| Layer        | Tool                                                                     | Why                                                                                                               |
| ------------ | ------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| Data source  | SEC EDGAR REST API                                                       | Free, no key. Also gives free ground-truth data (XBRL CompanyFacts)                                               |
| Parsing      | Custom HTML parser (BeautifulSoup + lxml + pandas)                       | SEC filings aren't semantic HTML; a custom Item-boundary-aware parser handles this better than PDF-oriented tools |
| Embeddings   | BGE-base-en-v1.5, self-hosted (768-dim, MIT licensed)                    | Zero API cost, zero rate limits, runs on CPU                                                                      |
| Vector store | FAISS (IndexFlatIP)                                                      | Free, local, no hosted vector DB fees                                                                             |
| LLM          | Google Gemini API free tier (primary) + Ollama/llama3.1 (local fallback) | No credit card, no expiry; fully local fallback for when quota is hit                                             |
| Demo UI      | Gradio, deployed via Docker to Hugging Face Spaces                       | Free, no card, no time limit                                                                                      |

## What's built so far

- [x] **Phase 1 — Ingestion**: downloads 10-K/10-Q filings for 8 tickers from SEC EDGAR
- [x] **Phase 2 — Parsing**: custom Item/Part-boundary-aware chunking of narrative text and tables
- [x] **Phase 3 — Embedding + indexing**: BGE-base embeddings, FAISS vector index
- [x] **Phase 4 — RAG answer engine**: cited, refusal-capable Q&A over the indexed filings
- [x] **Phase 5 — Structured extraction + XBRL validation**: LLM-extracted line items checked against SEC's official data — **72.9% accuracy**
- [ ] **Phase 6 — SQL storage + lightweight dashboard**
- [x] **Phase 7 — Containerized + deployed** (this Space)
- [ ] **Phase 8 — Full writeup, evaluation, polish**

## Try it

Use the **Ask a Question** tab to query any of the 8 indexed companies
(AAPL, MSFT, JPM, WMT, JNJ, XOM, KO, BA), or the **Extraction Accuracy**
tab to see the measured validation results against SEC's own data.

## Running locally

```bash
git clone <repo-url>
cd financial-rag-platform
python -m venv venv
source venv/Scripts/activate  # Windows Git Bash
pip install -r requirements.txt
cp .env.example .env  # fill in SEC_USER_AGENT and GEMINI_API_KEY
python app.py
```

Then open `http://localhost:7860`.
