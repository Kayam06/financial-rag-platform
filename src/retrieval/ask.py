"""
Phase 4 CLI entry point.

Usage:
    python -m src.retrieval.ask "What was AAPL's revenue in the most recent 10-K?"
    python -m src.retrieval.ask "What are JPM's risk factors?" --ticker JPM --top-k 5
"""

import argparse
import json

from src.retrieval.answer_engine import AnswerEngine


def main():
    parser = argparse.ArgumentParser(description="Ask a question against the filings RAG system.")
    parser.add_argument("question", type=str, help="Your question in plain English")
    parser.add_argument("--ticker", type=str, default=None, help="Restrict retrieval to one ticker, e.g. AAPL")
    parser.add_argument("--top-k", type=int, default=8, help="Number of chunks to retrieve")
    args = parser.parse_args()

    engine = AnswerEngine()
    result = engine.ask(args.question, top_k=args.top_k, ticker_filter=args.ticker)

    print("\n=== ANSWER ===")
    print(result["answer"])
    print(f"\n(LLM used: {result['sources_used']})")
    print("\n=== CITATIONS ===")
    print(json.dumps(result["citations"], indent=2))


if __name__ == "__main__":
    main()