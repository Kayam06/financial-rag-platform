"""
Phase 4 — LLM clients: Gemini (primary, free tier, via the new google-genai
SDK) with Ollama (local fallback) if Gemini's rate limit or an error is hit.
"""

import os
import requests
from google import genai
from dotenv import load_dotenv

load_dotenv()


class LLMError(Exception):
    pass


class GeminiClient:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise LLMError("GEMINI_API_KEY not set in .env")
        self.client = genai.Client(api_key=api_key)
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        if not response.text:
            raise LLMError("Gemini returned an empty response")
        return response.text


class OllamaClient:
    def __init__(self, base_url: str = None):
        self.base_url = base_url or os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        self.model_name = os.getenv("OLLAMA_MODEL", "llama3.1")

    def generate(self, prompt: str) -> str:
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model_name, "prompt": prompt, "stream": False},
                timeout=120,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            raise LLMError(
                f"Ollama request failed ({e}). Is `ollama serve` running locally "
                f"and have you pulled `{self.model_name}`?"
            )
        data = resp.json()
        text = data.get("response", "")
        if not text:
            raise LLMError("Ollama returned an empty response")
        return text


class FallbackLLM:
    """Tries Gemini first; falls back to Ollama on any failure."""

    def __init__(self):
        self.primary = GeminiClient()
        self.fallback = OllamaClient()

    def generate(self, prompt: str) -> tuple[str, str]:
        """Returns (answer_text, source_used)."""
        try:
            return self.primary.generate(prompt), "gemini"
        except Exception as e:
            print(f"[warn] Gemini failed ({e}), falling back to Ollama...")
            return self.fallback.generate(prompt), "ollama"
