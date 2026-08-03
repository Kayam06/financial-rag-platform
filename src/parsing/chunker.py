"""
Split long narrative sections into retrieval-sized chunks.

Uses a word-count proxy for tokens (len(text.split())) — good enough for Phase 2
before we wire in a real tokenizer for embedding budget control.
"""


def estimate_tokens(text: str) -> int:
    """Rough token count; 1 word ≈ 1 token for English financial prose."""
    return len(text.split())


def chunk_text(
    text: str,
    min_tokens: int = 500,
    max_tokens: int = 800,
) -> list[str]:
    """Split plain text into ~500–800 token chunks on paragraph boundaries.

    Paragraphs are separated by blank lines. We never split mid-paragraph, so
    short paragraphs may be merged until min_tokens is reached; a single long
    paragraph becomes its own chunk even if it exceeds max_tokens.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for paragraph in paragraphs:
        para_tokens = estimate_tokens(paragraph)

        if current and current_tokens + para_tokens > max_tokens:
            chunks.append("\n\n".join(current))
            current = [paragraph]
            current_tokens = para_tokens
            continue

        current.append(paragraph)
        current_tokens += para_tokens

        if current_tokens >= min_tokens:
            chunks.append("\n\n".join(current))
            current = []
            current_tokens = 0

    if current:
        if chunks and current_tokens < min_tokens:
            chunks[-1] = chunks[-1] + "\n\n" + "\n\n".join(current)
        else:
            chunks.append("\n\n".join(current))

    return chunks
