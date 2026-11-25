"""Demo: list the registered tools and run each one once, offline.

Run:  python examples/demo.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_text_toolbox import registry

print("Registered tools:")
for definition in registry.function_definitions():
    fn = definition["function"]
    print(f"  - {fn['name']}: {fn['description']}")

print("\n1) repair_json on messy LLM output:")
messy = "Sure! Here is the result:\n```json\n{'name': 'Ada', 'skills': ['python', 'sql',],}\n```"
print(json.dumps(registry.call("repair_json", {"text": messy}),
                 indent=2, ensure_ascii=False))

print("\n2) estimate_tokens with a price:")
print(registry.call("estimate_tokens",
                    {"text": "Budgeting a prompt starts with counting it.",
                     "price_per_1k_tokens": 0.50}))

print("\n3) chunk_text on a paragraph:")
text = ("Retrieval pipelines split documents before embedding them. "
        "Chunks that break mid-sentence embed badly. "
        "Sentence-aware splitting keeps each chunk coherent. "
        "Overlap preserves context across the seams.")
print(registry.call("chunk_text",
                    {"text": text, "chunk_size": 12, "overlap": 3}))

print("\n4) dedup_texts on near-copies:")
drafts = [
    "The report shows revenue grew twelve percent this quarter overall.",
    "The report shows revenue grew twelve percent this quarter overall!",
    "An unrelated note about office plants and watering schedules.",
]
print(registry.call("dedup_texts", {"texts": drafts}))

print("\n5) extractive_summary:")
article = ("Solar power is getting cheaper every year. "
           "Panels now cost a fraction of what they did a decade ago. "
           "The city council met on Tuesday to discuss parking. "
           "Because solar is cheap, rooftop solar installations keep "
           "growing across the country.")
print(registry.call("extractive_summary",
                    {"text": article, "sentence_count": 2}))
