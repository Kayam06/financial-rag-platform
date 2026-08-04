"""
One-off diagnostic: sample chunks with empty item_number for manual review.

Usage:
    python -m scripts.sample_empty_item_chunks

Windows note: all file reads use encoding='utf-8' — the default codepage will
crash on filing text containing non-ASCII characters.
"""
import json
import random
import sys
from pathlib import Path

from src.config import settings

SAMPLE_SIZE = 15
PREVIEW_CHARS = 150


def _safe_print(text: str) -> None:
    """Print UTF-8 safely on Windows consoles that default to cp1252."""
    encoding = sys.stdout.encoding or "utf-8"
    print(text.encode(encoding, errors="replace").decode(encoding))


def main():
    chunks_path = Path(settings.processed_data_dir) / "chunks.jsonl"
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"{chunks_path} not found. Run Phase 2 first:\n"
            "  python -m src.parsing.parse_filings"
        )

    empty_item_chunks: list[dict] = []
    with chunks_path.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            item_number = record.get("item_number")
            if item_number is None or item_number == "":
                empty_item_chunks.append(record)

    print(f"Total empty/null item_number chunks: {len(empty_item_chunks)}")
    if not empty_item_chunks:
        return

    sample = random.sample(empty_item_chunks, min(SAMPLE_SIZE, len(empty_item_chunks)))
    print(f"\nRandom sample of {len(sample)}:\n")
    for idx, record in enumerate(sample, start=1):
        content = record.get("content", "")
        preview = content[:PREVIEW_CHARS].replace("\n", " ")
        if len(content) > PREVIEW_CHARS:
            preview += "..."
        _safe_print(f"--- Sample {idx} ---")
        _safe_print(f"ticker:        {record.get('ticker')}")
        _safe_print(f"form:          {record.get('form')}")
        _safe_print(f"chunk_type:    {record.get('chunk_type')}")
        _safe_print(f"section_title: {record.get('section_title')}")
        _safe_print(f"content:       {preview}")
        print()


if __name__ == "__main__":
    main()
