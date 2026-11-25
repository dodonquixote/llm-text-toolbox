# llm-text-toolbox

**The text utilities every LLM workflow ends up needing** — a small, dependency-free Python library for the unglamorous text work around language models: salvaging JSON from messy model output, splitting documents for retrieval, estimating token costs before you spend them, catching near-duplicate drafts, and summarising without a model call.

Every tool is a plain Python function registered with a JSON-schema description, so the same tool can be:

- passed to an LLM as an **OpenAI-style function definition**,
- called directly from Python, and
- served over **MCP (Model Context Protocol)** — see the
  [MCP server](#mcp-server) section below.

## Quickstart

```bash
git clone https://github.com/dodonquixote/llm-text-toolbox
cd llm-text-toolbox
python examples/demo.py
```

```python
from llm_text_toolbox import registry

# What tools exist, in LLM-ready format:
print(registry.function_definitions())

# Call one:
fixed = registry.call("repair_json", {"text": "Here you go: {'a': 1,}"})
print(fixed["data"], fixed["repairs"])
```

## Tools

| Tool | What it does |
| --- | --- |
| `repair_json` | Extract the first JSON object/array from messy LLM output — strips markdown code fences and surrounding prose, removes trailing commas, converts single-quoted strings — and report exactly what was repaired. Raises a clear error when the JSON is unrecoverable instead of guessing. |
| `chunk_text` | Split long text into chunks of at most N words with configurable overlap, preferring sentence and paragraph boundaries (oversized sentences are hard-split as a last resort). |
| `estimate_tokens` | Heuristic token-count estimate (character + word rules of thumb, no tokenizer install needed), with an optional cost estimate given a price per 1K tokens. |
| `dedup_texts` | Near-duplicate detection over a list of texts using Jaccard similarity on word shingles; returns clusters of duplicates above a threshold. |
| `extractive_summary` | Frequency-based extractive summarisation: score sentences by topic-word frequency and return the top N in their original order. Deterministic, no LLM call. |
| MCP server | Expose every tool over the Model Context Protocol (stdio). |


## Design notes

- **Standard library only**: no dependency surprises when an agent environment installs it — and no tokenizer download just to estimate a prompt's cost.
- **One registry, many front doors**: function calling today, MCP tomorrow, same tool code.
- **Honest failure**: `repair_json` raises instead of inventing data; `estimate_tokens` documents itself as a heuristic (expect ~10–20% accuracy on prose), not a tokenizer replacement.
- **Deterministic**: the same input always produces the same chunks, clusters and summary — safe to cache and to test against.

## MCP server

Every registered tool is also available over the Model Context
Protocol, served over stdio by `llm_text_toolbox/server.py` — standard
library only, no MCP SDK required. It speaks JSON-RPC 2.0 and
implements `initialize`, `tools/list`, `tools/call` and `ping`.

Run it directly:

```bash
python -m llm_text_toolbox.server
```

Or point an MCP client (Claude Code, Cursor, Copilot, …) at it, using
the included `mcp_config_template.json`:

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

Restart the client and the llm-text-toolbox tools appear alongside its
built-in ones. Any tool added to the registry is automatically exposed
— nothing extra to wire up.

## Tests

```bash
python -m unittest discover -s tests
```

Try the chained example (repair a messy model output, then chunk,
price, dedupe and summarise a document — no API key needed):

```bash
python examples/agent_example.py
```

## License

MIT
