"""
Central config loader. Everything here is free-tier / self-hosted by design:
- SEC EDGAR: free, public, no key, but requires a descriptive User-Agent.
- Gemini: free tier via Google AI Studio (no credit card).
- Ollama: fully local, zero external dependency, used as fallback.
"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    # SEC requires a descriptive User-Agent identifying you + a contact email.
    # Format recommended by SEC: "Sample Company Name AdminContact@sample.com"
    sec_user_agent: str = os.getenv("SEC_USER_AGENT", "")

    # Google AI Studio free tier — https://aistudio.google.com/app/apikey
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # Local Ollama fallback (no key needed, must be running locally: `ollama serve`)
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1")

    # Local paths
    raw_data_dir: str = os.getenv("RAW_DATA_DIR", "data/raw")
    processed_data_dir: str = os.getenv("PROCESSED_DATA_DIR", "data/processed")

    # Local Postgres (via docker-compose) — falls back to SQLite if not set
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///data/processed/financial_rag.db")


settings = Settings()


def require_sec_user_agent() -> str:
    """SEC will block requests without a descriptive User-Agent. Fail loudly and early
    rather than letting a script silently get 403'd mid-run."""
    if not settings.sec_user_agent or "@" not in settings.sec_user_agent:
        raise RuntimeError(
            "SEC_USER_AGENT is not set (or looks invalid) in your .env file.\n"
            "SEC requires a descriptive User-Agent with a contact email, e.g.:\n"
            '  SEC_USER_AGENT="Kayam Pathan kayampathan06@gmail.com"\n'
            "This is free and requires no registration — SEC just wants a way to "
            "reach you if your script misbehaves."
        )
    return settings.sec_user_agent
