"""estimate_tokens: rough token counts and costs without a tokenizer.

Budgeting an LLM call usually needs an answer *before* the text is sent,
in environments where the model's real tokenizer is not installed. This
tool blends the two standard rules of thumb for English text — about 4
characters per token, and about 0.75 words per token — into one estimate.
It is a heuristic: expect it to land within roughly 10-20% of a real BPE
tokenizer on prose, and further off on code or non-English text.
"""
from __future__ import annotations

from ..registry import registry

CHARS_PER_TOKEN = 4.0
WORDS_PER_TOKEN = 0.75


def estimate_token_count(text: str) -> int:
    """Blend the character and word rules of thumb into one estimate."""
    if not text.strip():
        return 0
    char_estimate = len(text) / CHARS_PER_TOKEN
    word_estimate = len(text.split()) / WORDS_PER_TOKEN
    return int(round((char_estimate + word_estimate) / 2))


@registry.tool(
    name="estimate_tokens",
    description=("Estimate how many tokens a text will cost an LLM, using a "
                 "character/word heuristic (no tokenizer needed). Optionally "
                 "estimate the monetary cost given a price per 1K tokens."),
    parameters={
        "type": "object",
        "properties": {
            "text": {"type": "string",
                     "description": "The text to estimate."},
            "price_per_1k_tokens": {
                "type": "number",
                "description": "Optional price per 1,000 tokens in your "
                               "currency unit; when given, an estimated "
                               "cost is included in the result.",
            },
        },
        "required": ["text"],
    },
    tags=["text", "cost"],
)
def estimate_tokens(text: str, price_per_1k_tokens: float | None = None) -> dict:
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    if price_per_1k_tokens is not None and price_per_1k_tokens < 0:
        raise ValueError("price_per_1k_tokens must be >= 0")

    tokens = estimate_token_count(text)
    cost = None
    if price_per_1k_tokens is not None:
        cost = round(tokens / 1000 * price_per_1k_tokens, 6)
    return {"estimated_tokens": tokens,
            "characters": len(text),
            "words": len(text.split()),
            "price_per_1k_tokens": price_per_1k_tokens,
            "estimated_cost": cost}
