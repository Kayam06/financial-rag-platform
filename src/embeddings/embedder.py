"""
BGE embedding wrapper for the Financial Document Intelligence Platform.

Model choice: BAAI/bge-base-en-v1.5 (109M params, English-specialized,
MIT licensed) — chosen over the larger multilingual BGE-M3 (560M params)
after profiling actual embedding speed on CPU-only hardware. BGE-M3's
extra capability (multilingual support, native 8192-token context,
multi-vector retrieval) isn't used by this project — the corpus is 100%
English SEC filings chunked to 500-800 tokens — so its ~5x larger compute
cost bought nothing here. bge-base-en-v1.5 is a well-regarded, purpose-fit
model for English financial text and runs in a fraction of the time.

This module knows nothing about chunks.jsonl or FAISS — it's a thin,
reusable wrapper around "text in -> vector out". Phase 4 (retrieval) will
reuse this same class to embed user questions at query time, so keeping it
separate from the indexing script matters.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MODEL_NAME = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIM = 768  # bge-base-en-v1.5's native output dimension


class Embedder:
    """
    Loads the BGE embedding model once and exposes a simple `.encode()` method.

    Usage:
        embedder = Embedder()
        vectors = embedder.encode(["some text", "more text"])
        # vectors.shape == (2, 768), L2-normalized float32
    """

    def __init__(self, model_name: str = MODEL_NAME, device: Optional[str] = None):
        """
        device: "cuda", "cpu", or None (auto-detect: uses GPU if available,
        otherwise falls back to CPU with no error).
        """
        logger.info("Loading %s (device=%s)...", model_name, device or "auto")
        self.model = SentenceTransformer(model_name, device=device)
        logger.info("Model loaded on device: %s", self.model.device)

    def encode(
        self,
        texts: List[str],
        batch_size: int = 16,
        show_progress_bar: bool = True,
    ) -> np.ndarray:
        """
        Encode a list of strings into L2-normalized float32 vectors.

        Normalizing here (rather than at search time) means we can use a
        plain FAISS IndexFlatIP (inner product) and get cosine similarity
        for free — inner product of two unit vectors equals cosine
        similarity, so we don't need a separate normalization step later.
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype("float32")