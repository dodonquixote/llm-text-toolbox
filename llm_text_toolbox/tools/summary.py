"""extractive_summary: shrink a document without calling an LLM.

Frequency-based extractive summarisation: score every sentence by the
normalised frequency of its content words (words that recur across the
document are its topic words), take the top N, and return them in their
original order. No model call, no cost, deterministic — the right tool
for a first-pass digest, a preview, or trimming context before the
expensive model sees the text.
"""
from __future__ import annotations

import math
import re
from collections import Counter

from ..registry import registry
from .chunking import split_sentences

_WORD = re.compile(r"\w+")

_STOPWORDS = frozenset("""
a an the and or but if while of at by for with about against between into
through during before after above below to from up down in out on off over
under again further then once here there when where why how all any both
each few more most other some such no nor not only own same so than too
very can will just is am are was were be been being have has had having do
does did doing would should could ought i you he she it we they them his
her its our their this that these those as
""".split())


def _content_words(sentence: str) -> list[str]:
    return [w for w in _WORD.findall(sentence.lower()) if w not in _STOPWORDS]


def score_sentences(sentences: list[str]) -> list[float]:
    """Score each sentence by its share of the document's topic words."""
    frequencies: Counter[str] = Counter()
    for sentence in sentences:
        frequencies.update(_content_words(sentence))
    if not frequencies:
        return [0.0] * len(sentences)
    peak = max(frequencies.values())
    weights = {word: count / peak for word, count in frequencies.items()}

    scores = []
    for sentence in sentences:
        words = _content_words(sentence)
        if not words:
            scores.append(0.0)
            continue
        # Sum of topic-word weights, damped by sqrt(length) so long
        # sentences do not win just by containing more words.
        scores.append(sum(weights[w] for w in words) / math.sqrt(len(words)))
    return scores


@registry.tool(
    name="extractive_summary",
    description=("Summarise a text extractively: pick the top-N most "
                 "informative sentences by word-frequency scoring and return "
                 "them in their original order. Deterministic, no LLM call."),
    parameters={
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "The text to summarise."},
            "sentence_count": {"type": "integer",
                               "description": "How many sentences to keep "
                                              "(default 3).",
                               "default": 3},
        },
        "required": ["text"],
    },
    tags=["text"],
)
def extractive_summary(text: str, sentence_count: int = 3) -> dict:
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    if sentence_count < 1:
        raise ValueError("sentence_count must be at least 1")

    sentences = split_sentences(text)
    if not sentences:
        return {"summary": "", "sentences": [], "sentence_count": 0,
                "total_sentences": 0}
    if len(sentences) <= sentence_count:
        return {"summary": " ".join(sentences), "sentences": sentences,
                "sentence_count": len(sentences),
                "total_sentences": len(sentences)}

    scores = score_sentences(sentences)
    ranked = sorted(range(len(sentences)), key=lambda i: (-scores[i], i))
    chosen = sorted(ranked[:sentence_count])
    selected = [sentences[i] for i in chosen]
    return {"summary": " ".join(selected), "sentences": selected,
            "sentence_count": len(selected),
            "total_sentences": len(sentences)}
