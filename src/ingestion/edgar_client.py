"""
Thin client around the free, public SEC EDGAR APIs.

Endpoints used (all free, no API key, per SEC's official developer docs
https://www.sec.gov/search-filings/edgar-application-programming-interfaces):

  - https://www.sec.gov/files/company_tickers.json
      Bulk file mapping ticker -> CIK for every registered company.
  - https://data.sec.gov/submissions/CIK{cik10}.json
      Filing history + metadata for one company.
  - https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_nodash}/{filename}
      The actual filing document.
  - https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json
      Every structured XBRL fact ever filed by a company — this is the
      ground-truth source we validate extracted numbers against in Phase 5.

SEC's only requirement is a descriptive User-Agent with a contact email, and a
self-imposed rate limit (SEC's stated ceiling is 10 requests/second; we stay
well under that to be a polite citizen of a free public resource).
"""
import json
import time
from pathlib import Path
from typing import Optional

import requests

from src.config import settings, require_sec_user_agent

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/{filename}"

MIN_INTERVAL_SECONDS = 0.15  # keeps us well under SEC's 10 req/sec ceiling


class EdgarClient:
    def __init__(self, user_agent: Optional[str] = None, cache_dir: str = "data/raw/.cache"):
        self.user_agent = user_agent or require_sec_user_agent()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})
        self._last_request_time = 0.0
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._ticker_map: Optional[dict] = None

    # ---- low-level request helper -----------------------------------------
    def _get(self, url: str, **kwargs) -> requests.Response:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < MIN_INTERVAL_SECONDS:
            time.sleep(MIN_INTERVAL_SECONDS - elapsed)
        resp = self.session.get(url, timeout=30, **kwargs)
        self._last_request_time = time.monotonic()
        resp.raise_for_status()
        return resp

    # ---- ticker -> CIK ------------------------------------------------------
    def _load_ticker_map(self) -> dict:
        """Downloads (and caches locally) the bulk ticker->CIK mapping file.
        This is a few MB and rarely changes, so we cache it instead of
        re-fetching on every run."""
        if self._ticker_map is not None:
            return self._ticker_map

        cache_path = self.cache_dir / "company_tickers.json"
        if cache_path.exists():
            self._ticker_map = json.loads(cache_path.read_text())
            return self._ticker_map

        resp = self._get(TICKERS_URL)
        data = resp.json()
        cache_path.write_text(json.dumps(data))
        self._ticker_map = data
        return data

    def get_cik_for_ticker(self, ticker: str) -> str:
        """Returns a zero-padded 10-digit CIK string for a given ticker, e.g. 'AAPL' -> '0000320193'."""
        ticker = ticker.upper().strip()
        data = self._load_ticker_map()
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker:
                return str(entry["cik_str"]).zfill(10)
        raise ValueError(f"Ticker '{ticker}' not found in SEC's company_tickers.json")

    # ---- filings listing ----------------------------------------------------
    def get_recent_filings(
        self,
        cik10: str,
        form_types: tuple = ("10-K", "10-Q"),
        limit_per_form: int = 4,
    ) -> list[dict]:
        """Returns recent filings of the given form types, most recent first.
        Each item includes accessionNumber, form, filingDate, reportDate, and
        primaryDocument (the filename of the main filing document)."""
        resp = self._get(SUBMISSIONS_URL.format(cik10=cik10))
        payload = resp.json()
        recent = payload["filings"]["recent"]

        filings = []
        n = len(recent["form"])
        counts = {f: 0 for f in form_types}
        for i in range(n):
            form = recent["form"][i]
            if form in form_types and counts[form] < limit_per_form:
                filings.append(
                    {
                        "cik10": cik10,
                        "company_name": payload.get("name"),
                        "form": form,
                        "accessionNumber": recent["accessionNumber"][i],
                        "filingDate": recent["filingDate"][i],
                        "reportDate": recent["reportDate"][i],
                        "primaryDocument": recent["primaryDocument"][i],
                    }
                )
                counts[form] += 1
        return filings

    # ---- downloading a filing document --------------------------------------
    def download_filing_document(self, filing: dict, dest_dir: str) -> Path:
        """Downloads the primary document for one filing to dest_dir and returns the local path."""
        cik_int = int(filing["cik10"])  # archive URLs want the CIK without leading zeros
        accession_nodash = filing["accessionNumber"].replace("-", "")
        url = ARCHIVE_URL.format(
            cik_int=cik_int,
            accession_nodash=accession_nodash,
            filename=filing["primaryDocument"],
        )
        resp = self._get(url)

        dest_path = Path(dest_dir)
        dest_path.mkdir(parents=True, exist_ok=True)
        out_file = dest_path / f"{filing['cik10']}_{filing['form']}_{filing['filingDate']}_{filing['primaryDocument']}"
        out_file.write_bytes(resp.content)
        return out_file

    # ---- ground truth for validation (Phase 5) ------------------------------
    def get_company_facts(self, cik10: str) -> dict:
        """Returns every structured XBRL fact SEC has on file for this company —
        this is the free ground-truth source used later to score extraction accuracy."""
        resp = self._get(COMPANY_FACTS_URL.format(cik10=cik10))
        return resp.json()
