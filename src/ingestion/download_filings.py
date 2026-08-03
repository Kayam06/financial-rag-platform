"""
Phase 1 entry point.

Usage:
    python -m src.ingestion.download_filings

Edit TICKERS below to pick your companies. A good starting mix is one from
each of a few different sectors, since filing structure varies a lot between
e.g. a bank, a retailer, and a software company — that variety is what makes
the parsing phase (Phase 2) an actual engineering problem instead of a demo.
"""
import pandas as pd
from tqdm import tqdm

from src.config import settings
from src.ingestion.edgar_client import EdgarClient

# Pick 8-10 tickers across a few sectors. Feel free to change these.
TICKERS = [
    "AAPL",  # tech
    "MSFT",  # tech
    "JPM",   # banking
    "WMT",   # retail
    "JNJ",   # healthcare
    "XOM",   # energy
    "KO",    # consumer staples
    "BA",    # industrials
]

FORM_TYPES = ("10-K", "10-Q")
FILINGS_PER_FORM = 2  # 2 x 10-K and 2 x 10-Q per company to start


def main():
    client = EdgarClient()
    manifest_rows = []

    for ticker in tqdm(TICKERS, desc="Companies"):
        try:
            cik10 = client.get_cik_for_ticker(ticker)
        except ValueError as e:
            print(f"  [skip] {e}")
            continue

        filings = client.get_recent_filings(
            cik10, form_types=FORM_TYPES, limit_per_form=FILINGS_PER_FORM
        )

        for filing in filings:
            local_path = client.download_filing_document(
                filing, dest_dir=f"{settings.raw_data_dir}/{ticker}"
            )
            manifest_rows.append(
                {
                    "ticker": ticker,
                    "cik10": filing["cik10"],
                    "company_name": filing["company_name"],
                    "form": filing["form"],
                    "filing_date": filing["filingDate"],
                    "report_date": filing["reportDate"],
                    "accession_number": filing["accessionNumber"],
                    "local_path": str(local_path),
                }
            )

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = f"{settings.raw_data_dir}/manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"\nDownloaded {len(manifest)} filings across {manifest['ticker'].nunique()} companies.")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
