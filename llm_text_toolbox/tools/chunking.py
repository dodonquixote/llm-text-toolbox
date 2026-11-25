"""chunk_text: split long documents into LLM-sized pieces.

Splits text into chunks of at most `chunk_size` words with `overlap`
words carried over from the previous chunk — the standard shape for
retrieval pipelines and map-reduce summarisation. Sentences are kept
whole whenever they fit, so chunks break at sentence (and paragraph)
boundaries instead of mid-thought; a single sentence longer than the
chunk size is hard-split by words as a last resort.
"""
from __future__ import annotations

import re

from ..registry import registry

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, keeping paragraph boundaries as splits."""
    sentences: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        for piece in _SENTENCE_SPLIT.split(paragraph.strip()):
            piece = " ".join(piece.split())
            if piece:
                sentences.append(piece)
    return sentences


def _units(text: str, chunk_size: int, overlap: int) -> list[list[str]]:
    """Word-list units: whole sentences, hard-split when oversized.

    Oversized sentences are cut into pieces of chunk_size - overlap so
    the next chunk's overlap seed still fits alongside a piece.
    """
    piece = max(1, chunk_size - overlap)
    units: list[list[str]] = []
    for sentence in split_sentences(text):
        words = sentence.split()
        if len(words) <= chunk_size:
            units.append(words)
        else:
            for i in range(0, len(words), piece):
                units.append(words[i:i + piece])
    return units


@registry.tool(
    name="chunk_text",
    description=("Split long text into chunks of at most chunk_size words, "
                 "carrying overlap words over from the previous chunk. "
                 "Sentence and paragraph boundaries are preferred, so chunks "
                 "rarely break mid-sentence."),
    parameters={
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "The text to split."},
            "chunk_size": {"type": "integer",
                           "description": "Maximum words per chunk (default 200).",
                           "default": 200},
            "overlap": {"type": "integer",
                        "description": "Words repeated from the previous chunk "
                                       "at the start of the next (default 20). "
                                       "Must be smaller than chunk_size.",
                        "default": 20},
        },
        "required": ["text"],
    },
    tags=["text"],
)
def chunk_text(text: str, chunk_size: int = 200, overlap: int = 20) -> dict:
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    total_words = len(text.split())
    if total_words == 0:
        return {"chunks": [], "chunk_count": 0, "chunk_size": chunk_size,
                "overlap": overlap, "total_words": 0}

    chunks: list[list[str]] = []
    current: list[str] = []
    for unit in _units(text, chunk_size, overlap):
        if len(current) + len(unit) <= chunk_size:
            current = current + unit
            continue
        # Flush, then seed the next chunk with the tail of this one.
        chunks.append(current)
        seed = current[-overlap:] if overlap else []
        if len(seed) + len(unit) <= chunk_size:
            current = seed + unit
        else:
            current = list(unit)
    if current:
        chunks.append(current)

    texts = [" ".join(chunk) for chunk in chunks]
    return {"chunks": texts, "chunk_count": len(texts),
            "chunk_size": chunk_size, "overlap": overlap,
            "total_words": total_words}
