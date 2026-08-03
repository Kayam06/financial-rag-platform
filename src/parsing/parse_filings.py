"""
Phase 2 entry point.

Usage:
    python -m src.parsing.parse_filings

Reads the Phase 1 manifest, parses every downloaded .htm filing into narrative
sections and tables, chunks the narrative text, and writes one JSON object per
line to data/processed/chunks.jsonl for embedding in Phase 3.
"""
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.config import settings
from src.parsing.chunker import chunk_text
from src.parsing.html_parser import FilingParser, serialize_table


def main():
    manifest_path = Path(settings.raw_data_dir) / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. Run Phase 1 first:\n"
            "  python -m src.ingestion.download_filings"
        )

    manifest = pd.read_csv(manifest_path)
    out_dir = Path(settings.processed_data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "chunks.jsonl"

    chunk_rows = []
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Filings"):
        filing_path = Path(row["local_path"])
        if not filing_path.exists():
            print(f"  [skip] missing file: {filing_path}")
            continue

        parser = FilingParser(filing_path)
        sections, tables = parser.parse()

        base = {
            "ticker": row["ticker"],
            "cik10": row["cik10"],
            "form": row["form"],
            "filing_date": row["filing_date"],
            "report_date": row["report_date"],
        }

        for section in sections:
            if not section.text.strip():
                continue
            for chunk_index, content in enumerate(chunk_text(section.text)):
                chunk_rows.append(
                    {
                        **base,
                        "item_number": section.item_number,
                        "section_title": section.section_title,
                        "chunk_index": chunk_index,
                        "chunk_type": "text",
                        "content": content,
                    }
                )

        for table in tables:
            chunk_rows.append(
                {
                    **base,
                    "item_number": table.item_number,
                    "section_title": table.section_title,
                    "chunk_index": 0,
                    "chunk_type": "table",
                    "content": serialize_table(table.dataframe),
                }
            )

    with out_path.open("w", encoding="utf-8") as f:
        for record in chunk_rows:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    text_chunks = sum(1 for r in chunk_rows if r["chunk_type"] == "text")
    table_chunks = sum(1 for r in chunk_rows if r["chunk_type"] == "table")
    print(
        f"\nWrote {len(chunk_rows)} chunks ({text_chunks} text, {table_chunks} table) "
        f"from {manifest['ticker'].nunique()} companies."
    )
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
