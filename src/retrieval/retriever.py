"""
Phase 4 — Retrieval layer.

Loads the FAISS index + aligned metadata built in Phase 3, embeds an
incoming question with the same Embedder used at index-build time, and
returns the top-k most similar chunks with their full metadata attached.

Confirmed against real Phase 3 code:
    FAISS index:   data/processed/faiss_index.bin   (IndexFlatIP, 768-dim)
    Metadata:      data/processed/chunk_metadata.jsonl  (row i <-> index i)
    Embedder:      src/embeddings/embedder.py -> Embedder(device=...).encode(texts, ...)
                   returns L2-normalized float32 vectors (cosine similarity
                   via inner product) -- so higher score = more similar,
                   no extra normalization needed here.
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import faiss
import numpy as np

from src.embeddings.embedder import Embedder

FAISS_INDEX_PATH = Path("data/processed/faiss_index.bin")
METADATA_PATH = Path("data/processed/chunk_metadata.jsonl")


@dataclass
class RetrievedChunk:
    text: str
    ticker: str
    form: str
    item_number: str
    part: str
    section_title: str
    chunk_type: str
    filing_date: str
    report_date: str
    score: float


class Retriever:
    def __init__(
        self,
        index_path: Path = FAISS_INDEX_PATH,
        metadata_path: Path = METADATA_PATH,
        device: Optional[str] = None,
    ):
        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {index_path}. "
                "Did Phase 3's build_index.py run and save here?"
            )
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}.")

        self.index = faiss.read_index(str(index_path))

        self.metadata: list[dict] = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.metadata.append(json.loads(line))

        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                f"Mismatch: FAISS index has {self.index.ntotal} vectors but "
                f"metadata has {len(self.metadata)} rows. They must be aligned "
                "1:1 in the same order for retrieval to return correct chunks."
            )

        # device=None lets Embedder auto-detect (matches build_index.py's own
        # default behavior when no --device flag is passed at index-build time)
        self.embedder = Embedder(device=device)

    def search(
        self,
        query: str,
        top_k: int = 8,
        ticker_filter: Optional[str] = None,
    ) -> list[RetrievedChunk]:
        """
        Embed the query, run FAISS similarity search, return top_k chunks
        (optionally restricted to a single ticker) with metadata attached.
        """
        query_vec = self.embedder.encode([query], batch_size=1, show_progress_bar=False)
        query_vec = np.asarray(query_vec, dtype="float32")

        # Over-fetch when filtering by ticker so we still end up with top_k
        # relevant results after filtering.
        fetch_k = top_k * 5 if ticker_filter else top_k
        fetch_k = min(fetch_k, self.index.ntotal)

        scores, indices = self.index.search(query_vec, fetch_k)

        results: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            row = self.metadata[idx]
            if ticker_filter and row.get("ticker", "").upper() != ticker_filter.upper():
                continue
            results.append(
                RetrievedChunk(
                    text=row.get("text", ""),
                    ticker=row.get("ticker", ""),
                    form=row.get("form", ""),
                    item_number=row.get("item_number", ""),
                    part=row.get("part", ""),
                    section_title=row.get("section_title", ""),
                    chunk_type=row.get("chunk_type", ""),
                    filing_date=row.get("filing_date", ""),
                    report_date=row.get("report_date", ""),
                    score=float(score),
                )
            )
            if len(results) >= top_k:
                break

        return results
