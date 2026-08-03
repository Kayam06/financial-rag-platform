# Financial Document Intelligence Platform (RAG)

Ask a plain-English question about a public company's financials and get an
answer sourced directly from its actual SEC filings — with the specific
numbers checked against SEC's own structured data, stored in a queryable
database, and visualized on a dashboard.

Built entirely on free tools: no paid APIs, no cloud bill, no credit card
required to run any part of this.

## Architecture

```
SEC EDGAR filings
      |
Parse & extract (pdfplumber + PyMuPDF)
      |
Embed & index (BGE-M3, self-hosted + FAISS)
      |
   /-----\
  |       |
Answer   Extract & validate
engine   (vs EDGAR ground truth)
  |       |
Cited    SQL database
answers      |
         Power BI dashboard
```

## Why SEC EDGAR

SEC EDGAR is free, public, requires no API key, and — critically — it also
exposes every company's officially reported financial numbers as structured
XBRL data (the "Company Facts" API). That means we're not just guessing
whether our extraction pipeline works: we can automatically check every
number our pipeline pulls out of a messy 10-K against SEC's own machine-
readable ground truth, and report a real accuracy percentage.

## Free-tier tech stack

| Layer | Tool | Cost |
|---|---|---|
| Data source | SEC EDGAR REST API | Free, no key |
| Parsing | pdfplumber, PyMuPDF | Free, open source |
| Embeddings | BGE-M3 (self-hosted via `sentence-transformers`) | Free, runs on your machine |
| Vector store | FAISS | Free, open source |
| Orchestration | LangChain | Free, open source |
| LLM | Gemini API (free tier) primary, Ollama (local) fallback | Free, no card |
| Storage | Postgres (local Docker) or SQLite | Free |
| Dashboard | Power BI Desktop | Free (Desktop only, not the paid Service) |
| Deployment | Hugging Face Spaces | Free, no card |

## Setup

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# then edit .env:
#   - SEC_USER_AGENT: your name + email (SEC just wants a contact, no signup)
#   - GEMINI_API_KEY: free key from https://aistudio.google.com/app/apikey
```

Optional local Postgres instead of SQLite:
```bash
docker compose -f docker/docker-compose.yml up -d
```

## Phase 1: download filings

```bash
python -m src.ingestion.download_filings
```

Edit the `TICKERS` list in `src/ingestion/download_filings.py` first — it
ships with 8 companies across different sectors (tech, banking, retail,
healthcare, energy, consumer staples, industrials) since filing structure
varies a lot by industry, which is exactly what makes Phase 2 (parsing) a
real problem worth solving rather than a toy demo.

This writes raw filings to `data/raw/<TICKER>/` and a `data/raw/manifest.csv`
index of everything downloaded.

Run the tests (fully mocked, no network needed):
```bash
pytest tests/
```

## Roadmap

- [x] Phase 1 — Ingestion: SEC EDGAR downloader + manifest
- [ ] Phase 2 — Parsing: section-aware chunking of tables + narrative text
- [ ] Phase 3 — Embed + index: BGE-M3 into FAISS with per-chunk metadata
- [ ] Phase 4 — RAG answer engine: retrieval + cited, hallucination-guarded answers
- [ ] Phase 5 — Structured extraction + validation against EDGAR ground truth
- [ ] Phase 6 — SQL storage + Power BI dashboard
- [ ] Phase 7 — Containerize + deploy demo to Hugging Face Spaces
- [ ] Phase 8 — Evaluation write-up (accuracy %, retrieval precision, failure modes)
