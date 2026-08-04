"""Quick post-parse verification of item_number and JPM section titles."""
import json
from collections import Counter, defaultdict

from src.config import settings

chunks_path = f"{settings.processed_data_dir}/chunks.jsonl"

jpm_text_titles: Counter = Counter()
jpm_sig_mislabeled = 0
jpm_mdna_titles: Counter = Counter()

by_ticker: dict[str, Counter] = defaultdict(Counter)
empty_by_ticker: dict[str, Counter] = defaultdict(Counter)

with open(chunks_path, encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        ticker = r.get("ticker")
        inum = r.get("item_number") or ""
        by_ticker[ticker][inum or "(empty)"] += 1
        if not inum:
            empty_by_ticker[ticker][r.get("section_title", "")] += 1

        if (
            ticker == "JPM"
            and r.get("form") == "10-Q"
            and r.get("filing_date") == "2026-05-01"
            and r.get("chunk_type") == "text"
        ):
            title = r.get("section_title", "")
            jpm_text_titles[title] += 1
            content = r.get("content", "")
            if title.upper() in ("SIGNATURE", "SIGNATURES", "SIGNAT URES"):
                if any(
                    k in content
                    for k in (
                        "CONSOLIDATED RESULTS",
                        "credit exposure",
                        "Capital allocation",
                        "MD&A",
                    )
                ):
                    jpm_sig_mislabeled += 1
            if any(
                k in title.upper()
                for k in ("INTRODUCTION", "EXECUTIVE", "CONSOLIDATED", "CAPITAL RISK", "CREDIT")
            ):
                jpm_mdna_titles[title] += 1

total = sum(sum(c.values()) for c in by_ticker.values())
print(f"Total chunks: {total}\n")

print("=== item_number by ticker ===")
for ticker in sorted(by_ticker):
    c = by_ticker[ticker]
    print(
        f"  {ticker}: {sum(c.values())} chunks | "
        f"empty={c['(empty)']} | item 16={c.get('16', 0)}"
    )

print("\n=== JPM 10-Q 2026-05-01 text section_title (top 15) ===")
for title, count in jpm_text_titles.most_common(15):
    print(f"  {count:4} {title!r}")

print(f"\nMislabeled SIGNATURE text chunks: {jpm_sig_mislabeled}")
print(f"Descriptive MD&A-style titles: {sum(jpm_mdna_titles.values())} chunks")
for title, count in jpm_mdna_titles.most_common(8):
    print(f"  {count:4} {title!r}")

print("\n=== empty item_number top section_titles per ticker ===")
for ticker in sorted(empty_by_ticker):
    top = empty_by_ticker[ticker].most_common(5)
    print(f"  {ticker}: " + ", ".join(f"{t!r}({n})" for t, n in top))
