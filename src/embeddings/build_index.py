"""
Phase 3 CLI entry point: embed every chunk in data/processed/chunks.jsonl
and build a FAISS index for similarity search.

Run from the repo root with the venv activated:

    python -m src.embeddings.build_index

Outputs (both written to data/processed/):
  - faiss_index.bin       the FAISS vector index (just the vectors)
  - chunk_metadata.jsonl  one JSON object per row, SAME ORDER as the index,
                           so FAISS position i <-> chunk_metadata.jsonl line i
                           (0-indexed). Includes ticker/form/section_title/
                           etc AND the original chunk text, so Phase 4 can
                           retrieve and cite without a second lookup into
                           chunks.jsonl.

Why a separate metadata file instead of a database: FAISS only stores
vectors + an integer position, nothing else. A plain .jsonl file aligned by
row order is the simplest possible approach at this scale and avoids
pulling in a DB dependency in Phase 3 — Phase 6 adds Postgres/SQLite later,
for validated extraction results, which is a different concern.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

import faiss

from src.embeddings.embedder import Embedder, EMBEDDING_DIM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CHUNKS_PATH = Path("data/processed/chunks.jsonl")
DEFAULT_INDEX_PATH = Path("data/processed/faiss_index.bin")
DEFAULT_METADATA_PATH = Path("data/processed/chunk_metadata.jsonl")


def load_chunks(path: Path) -> list[dict]:
    """Read chunks.jsonl into a list of dicts, preserving file order."""
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError as e:
                # Fail loud rather than silently skipping a bad row — a
                # malformed line usually means something upstream broke.
                raise ValueError(f"Malformed JSON at line {line_num} of {path}: {e}") from e
    return chunks


def build_index(
    chunks_path: Path = DEFAULT_CHUNKS_PATH,
    index_path: Path = DEFAULT_INDEX_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
    batch_size: int = 16,
    device: Optional[str] = None,
) -> None:
    logger.info("Loading chunks from %s", chunks_path)
    chunks = load_chunks(chunks_path)
    logger.info("Loaded %d chunks", len(chunks))

    if not chunks:
        raise ValueError(f"No chunks found in {chunks_path} — nothing to embed.")

    # Guard against empty/whitespace-only content, which would otherwise
    # produce a meaningless embedding and silently corrupt retrieval later.
    texts = []
    for i, c in enumerate(chunks):
        content = c.get("content", "")
        if not content or not content.strip():
            raise ValueError(
                f"Chunk at index {i} (ticker={c.get('ticker')}, "
                f"item={c.get('item_number')}) has empty content — "
                f"check chunks.jsonl for corruption before embedding."
            )
        texts.append(content)

    embedder = Embedder(device=device)

    start = time.time()
    vectors = embedder.encode(texts, batch_size=batch_size, show_progress_bar=True)
    elapsed = time.time() - start
    logger.info(
        "Embedded %d chunks in %.1fs (%.3fs/chunk)",
        len(texts), elapsed, elapsed / len(texts),
    )

    if vectors.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"Unexpected embedding dimension: got {vectors.shape[1]}, "
            f"expected {EMBEDDING_DIM}. Did the model change?"
        )

    # IndexFlatIP = exact (brute-force) inner-product search. With ~8,000
    # vectors this is fast enough (milliseconds per query) and simplest to
    # reason about — no need for an approximate index (IVF/HNSW) at this
    # scale. Revisit only if the corpus grows into the hundreds of
    # thousands of chunks.
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(vectors)
    logger.info("FAISS index built: %d vectors, dim=%d", index.ntotal, EMBEDDING_DIM)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    logger.info("Saved FAISS index to %s", index_path)

    with open(metadata_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    logger.info("Saved aligned metadata to %s", metadata_path)

    logger.info("Done. FAISS index row i <-> chunk_metadata.jsonl line i (0-indexed).")


def main():
    parser = argparse.ArgumentParser(description="Phase 3: embed chunks and build FAISS index.")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--index-out", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--metadata-out", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Lower this (e.g. 4-8) if you hit memory errors on CPU.",
    )
    parser.add_argument(
        "--device", type=str, default=None, choices=[None, "cpu", "cuda"],
        help="Force a device; default auto-detects.",
    )
    args = parser.parse_args()

    build_index(
        chunks_path=args.chunks,
        index_path=args.index_out,
        metadata_path=args.metadata_out,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()