---
name: llm-text-toolbox
description: Text utilities for LLM and AI-agent workflows. Use this skill when you need to (1) recover JSON from messy LLM output — code fences, surrounding prose, trailing commas, single-quoted strings, (2) split long text into chunks with overlap for retrieval or map-reduce, preferring sentence boundaries, (3) estimate token counts and cost without installing a tokenizer, (4) find near-duplicate texts in a batch using word-shingle Jaccard similarity, or (5) summarise a text extractively by sentence frequency scoring, with no model call. The tools run locally in Python (standard library only) through a registry, as LLM function definitions, or over an MCP server.
---

# llm-text-toolbox

A dependency-free Python library of text tools for LLM/agent workflows: cleaning up after model output, preparing documents for models, and checking costs before spending them. Source of truth and updates: https://github.com/dodonquixote/llm-text-toolbox

## Installation

Requires Python 3.10+. No third-party dependencies.

```bash
git clone https://github.com/dodonquixote/llm-text-toolbox
cd llm-text-toolbox
```

Or copy the `llm_text_toolbox/` package from this zip into your project. Then:

```python
from llm_text_toolbox import registry
```

## Calling tools from Python

```python
# Every tool, as OpenAI-style function definitions for your LLM:
definitions = registry.function_definitions()

# Call a tool by name (arguments as a dict or a JSON string):
result = registry.call("repair_json", {"text": "Here you go: {'a': 1,}"})
```

## Tools

### repair_json
Extract and repair the first JSON object or array from messy LLM output text: strips markdown code fences and surrounding prose, removes trailing commas and converts single-quoted strings to double-quoted ones.
- Parameters: `text` (string, required).
- Returns: `{data, repaired, repairs}` — the parsed value, whether anything had to change, and the list of repairs applied (e.g. `"removed markdown code fence"`, `"removed trailing commas"`).
- Raises `ValueError` when no JSON value is present or the value cannot be recovered; it never guesses at missing data.

### chunk_text
Split long text into chunks of at most `chunk_size` words, carrying `overlap` words over from the previous chunk. Sentence and paragraph boundaries are preferred; a single sentence longer than the chunk size is hard-split by words.
- Parameters: `text` (string, required), `chunk_size` (integer, default 200), `overlap` (integer, default 20, must be smaller than `chunk_size`).
- Returns: `{chunks, chunk_count, chunk_size, overlap, total_words}`.

### estimate_tokens
Estimate how many tokens a text will cost an LLM, blending the ~4 characters/token and ~0.75 words/token rules of thumb — no tokenizer needed. Optionally estimate monetary cost.
- Parameters: `text` (string, required), `price_per_1k_tokens` (number, optional).
- Returns: `{estimated_tokens, characters, words, price_per_1k_tokens, estimated_cost}` where `estimated_cost` is null unless a price was given.
- This is a heuristic: expect roughly 10-20% accuracy on English prose, less on code or other languages.

### dedup_texts
Detect near-duplicate texts in a list using Jaccard similarity over word shingles, clustered with union-find. Catches copies with small edits that exact matching misses.
- Parameters: `texts` (array of strings, required), `threshold` (number, default 0.6, in (0, 1]), `shingle_size` (integer, default 3).
- Returns: `{groups, group_count, duplicate_count, text_count, threshold, shingle_size}` where each group is `{indices, size}` and only groups of 2+ are returned.

### extractive_summary
Summarise a text extractively: score sentences by normalised word frequency (stopwords excluded), keep the top N, return them in their original order. Deterministic, no LLM call.
- Parameters: `text` (string, required), `sentence_count` (integer, default 3).
- Returns: `{summary, sentences, sentence_count, total_sentences}`.

## MCP server

This package includes a complete MCP server (`llm_text_toolbox/server.py`, standard library only) that exposes all five tools over stdio to any MCP client (Claude Code, Cursor, Copilot, and others). Setup:

1. Unzip this package on the machine where your agent runs (or clone the repo above).
2. Point your MCP client at it using the included `mcp_config_template.json`, adjusting the path:

```json
{
  "mcpServers": {
    "llm-text-toolbox": {
      "command": "python",
      "args": ["-m", "llm_text_toolbox.server"],
      "cwd": "/path/to/llm-text-toolbox"
    }
  }
}
```

3. Restart the client; the llm-text-toolbox tools appear alongside its built-in tools.

## Notes for the agent

- Reach for `repair_json` before giving up on a model's structured output or hand-writing a fallback parser; check its `repairs` list to know how dirty the output was.
- Use `estimate_tokens` for budgeting and routing decisions, not for billing — it is an estimate by design.
- `dedup_texts` threshold 0.6 with 3-word shingles is a good default for drafts and scraped pages; raise the threshold toward 0.9 when only near-identical copies should merge.
- All tools are pure and safe to call repeatedly; invalid arguments raise `ValueError` with a specific message.
