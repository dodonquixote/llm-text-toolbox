"""dedup_texts: find near-duplicate texts in a batch.

Agents that collect drafts, search results or scraped pages end up with
the same content reworded a dozen ways. Exact-match dedupe misses those;
this tool compares texts by Jaccard similarity over word shingles
(consecutive word n-grams), which catches copies with small edits, and
clusters everything at or above a similarity threshold with union-find.
"""
from __future__ import annotations

import re

from ..registry import registry

_WORD = re.compile(r"\w+")


def word_shingles(text: str, shingle_size: int) -> set[tuple[str, ...]]:
    """The set of consecutive word n-grams in a text.

    A text shorter than the shingle size contributes its whole word
    sequence as a single shingle; a text with no words contributes none.
    """
    words = _WORD.findall(text.lower())
    if not words:
        return set()
    if len(words) <= shingle_size:
        return {tuple(words)}
    return {tuple(words[i:i + shingle_size])
            for i in range(len(words) - shingle_size + 1)}


def jaccard(a: set, b: set) -> float:
    """Jaccard similarity of two sets; two empty sets count as identical."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@registry.tool(
    name="dedup_texts",
    description=("Detect near-duplicate texts in a list using Jaccard "
                 "similarity over word shingles. Returns clusters (groups of "
                 "indices) whose members are at or above the similarity "
                 "threshold, so an agent can keep one copy of each."),
    parameters={
        "type": "object",
        "properties": {
            "texts": {"type": "array", "items": {"type": "string"},
                      "description": "The texts to compare with each other."},
            "threshold": {"type": "number",
                          "description": "Minimum Jaccard similarity to call "
                                         "two texts duplicates (default 0.6, "
                                         "must be in (0, 1]).",
                          "default": 0.6},
            "shingle_size": {"type": "integer",
                             "description": "Words per shingle (default 3). "
                                            "Smaller is more forgiving of edits.",
                             "default": 3},
        },
        "required": ["texts"],
    },
    tags=["text"],
)
def dedup_texts(texts: list[str], threshold: float = 0.6,
                shingle_size: int = 3) -> dict:
    if not isinstance(texts, list) or not all(
            isinstance(t, str) for t in texts):
        raise ValueError("texts must be a list of strings")
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    if shingle_size < 1:
        raise ValueError("shingle_size must be at least 1")

    shingles = [word_shingles(t, shingle_size) for t in texts]

    # Union-find over indices.
    parent = list(range(len(texts)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if jaccard(shingles[i], shingles[j]) >= threshold:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri

    clusters: dict[int, list[int]] = {}
    for i in range(len(texts)):
        clusters.setdefault(find(i), []).append(i)

    groups = [{"indices": sorted(members), "size": len(members)}
              for members in clusters.values() if len(members) >= 2]
    groups.sort(key=lambda g: g["indices"][0])

    return {"groups": groups,
            "group_count": len(groups),
            "duplicate_count": sum(g["size"] for g in groups),
            "text_count": len(texts),
            "threshold": threshold,
            "shingle_size": shingle_size}
