"""Chained example: clean up after an LLM, with no API key needed.

A realistic agent workflow where the tools feed each other through the
registry:

1. The model returned "JSON" wrapped in prose, in a code fence, with
   single quotes and a trailing comma -> repair_json recovers it.
2. The recovered record points at a long source document -> chunk_text
   splits it for retrieval, and estimate_tokens prices the chunks.
3. The model drafted the same answer three ways -> dedup_texts finds
   the near-copies so only one is kept.
4. extractive_summary produces a cheap preview of the document before
   anyone pays for a model-written one.

Run:  python examples/agent_example.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_text_toolbox import registry

RAW_MODEL_OUTPUT = """\
Of course! Here is the extracted record:

```json
{
  'title': 'Quarterly infrastructure review',
  'topics': ['latency', 'cost', 'reliability',],
}
```

Let me know if you need anything else."""

DOCUMENT = (
    "Latency dominated the review for the third quarter in a row. "
    "The checkout path still waits on two synchronous services, and "
    "latency there directly costs completed orders. "
    "Costs rose mainly because of idle capacity kept for peak traffic. "
    "The finance team asked for a capacity plan before the next quarter. "
    "Reliability improved after the retry budget was introduced, with "
    "fewer cascading failures during deploys. "
    "Latency work is now the top priority: the team will attack the "
    "checkout path first, then re-measure latency across all services."
)

DRAFT_ANSWERS = [
    "Latency is the top priority and the checkout path comes first.",
    "Latency is the top priority, and the checkout path comes first!",
    "The canteen menu changes next week to include more vegetarian meals.",
]


def main() -> None:
    print("Registered tools:", ", ".join(registry.names()))

    # Step 1: recover the structured record from the model's raw text.
    recovered = registry.call("repair_json", {"text": RAW_MODEL_OUTPUT})
    print(f"\n1) repair_json: data={json.dumps(recovered['data'])}")
    print(f"   repairs applied: {recovered['repairs']}")

    # Step 2: split the document the record points at, and price it.
    chunks = registry.call("chunk_text", {
        "text": DOCUMENT, "chunk_size": 40, "overlap": 8})
    print(f"\n2) chunk_text: {chunks['chunk_count']} chunks "
          f"from {chunks['total_words']} words")
    budget = registry.call("estimate_tokens", {
        "text": DOCUMENT, "price_per_1k_tokens": 0.50})
    print(f"   estimate_tokens: ~{budget['estimated_tokens']} tokens, "
          f"est. cost {budget['estimated_cost']} at "
          f"{budget['price_per_1k_tokens']}/1K tokens")

    # Step 3: the model paraphrased itself — keep one copy.
    deduped = registry.call("dedup_texts", {"texts": DRAFT_ANSWERS})
    print(f"\n3) dedup_texts: {deduped['group_count']} duplicate group(s), "
          f"indices {deduped['groups'][0]['indices'] if deduped['groups'] else []}")

    # Step 4: a free preview summary before spending model tokens.
    summary = registry.call("extractive_summary", {
        "text": DOCUMENT, "sentence_count": 2})
    print(f"\n4) extractive_summary: {summary['summary']}")

    print("\nChained workflow done — messy model output became a clean "
          "record, a priced set of chunks, one deduplicated answer and "
          "a preview summary, all offline.")


if __name__ == "__main__":
    main()
