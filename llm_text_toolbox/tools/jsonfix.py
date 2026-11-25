"""repair_json: salvage structured data from messy LLM output.

LLMs asked for JSON routinely return something *almost* JSON: wrapped in
markdown code fences, padded with prose ("Sure! Here is the data: ..."),
trailing commas, or single-quoted strings copied from Python reprs.
repair_json extracts the first JSON object or array from the text and
fixes the common breakage, reporting exactly what it had to repair. If
the value cannot be recovered it raises ValueError instead of guessing.
"""
from __future__ import annotations

import json

from ..registry import registry

_FENCE = "```"


def _strip_code_fence(text: str) -> tuple[str, bool]:
    """Remove a markdown code fence around the payload, if present."""
    start = text.find(_FENCE)
    if start == -1:
        return text, False
    # Skip the fence line itself (``` or ```json).
    body_start = text.find("\n", start)
    if body_start == -1:
        return text, False
    end = text.find(_FENCE, body_start)
    if end == -1:
        return text, False
    return text[body_start + 1:end], True


def _extract_first_value(text: str) -> tuple[str | None, bool]:
    """Return (candidate, had_surrounding_text) for the first { or [ value.

    The scan tracks both quote styles so brackets inside strings do not
    unbalance the match.
    """
    start = None
    for i, ch in enumerate(text):
        if ch in "{[":
            start = i
            break
    if start is None:
        return None, False

    stack: list[str] = []
    pairs = {"{": "}", "[": "]"}
    in_single = in_double = escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if escaped:
            escaped = False
            continue
        if ch == "\\" and (in_single or in_double):
            escaped = True
            continue
        if in_double:
            if ch == '"':
                in_double = False
            continue
        if in_single:
            if ch == "'":
                in_single = False
            continue
        if ch == '"':
            in_double = True
        elif ch == "'":
            in_single = True
        elif ch in pairs:
            stack.append(pairs[ch])
        elif ch in "}]":
            if not stack or stack[-1] != ch:
                # Unbalanced: hand the rest to the repair/parse stages,
                # which will either fix it or raise a clear error.
                return text[start:], True
            stack.pop()
            if not stack:
                surrounding = bool(text[:start].strip() or text[i + 1:].strip())
                return text[start:i + 1], surrounding
    # Ran off the end (truncated output): take what is there.
    return text[start:], bool(text[:start].strip())


def _remove_trailing_commas(text: str) -> tuple[str, bool]:
    """Drop commas that sit immediately before a } or ] (outside strings)."""
    out: list[str] = []
    in_single = in_double = escaped = False
    changed = False
    for i, ch in enumerate(text):
        if escaped:
            out.append(ch)
            escaped = False
            continue
        if ch == "\\" and (in_single or in_double):
            out.append(ch)
            escaped = True
            continue
        if in_double:
            out.append(ch)
            if ch == '"':
                in_double = False
            continue
        if in_single:
            out.append(ch)
            if ch == "'":
                in_single = False
            continue
        if ch == '"':
            in_double = True
            out.append(ch)
        elif ch == "'":
            in_single = True
            out.append(ch)
        elif ch == ",":
            j = i + 1
            while j < len(text) and text[j] in " \t\r\n":
                j += 1
            if j < len(text) and text[j] in "}]":
                changed = True
                continue  # drop the comma
            out.append(ch)
        else:
            out.append(ch)
    return "".join(out), changed


def _convert_single_quotes(text: str) -> tuple[str, bool]:
    """Convert '...' strings to "..." strings where it is safe to do so.

    A single-quoted string is converted only when it terminates cleanly;
    apostrophes inside double-quoted strings and inside words are left
    alone because the scan only starts a string at a quote that follows
    a structural character ({ [ , : or whitespace).
    """
    out: list[str] = []
    i = 0
    changed = False
    in_double = escaped = False
    prev_significant = ""
    while i < len(text):
        ch = text[i]
        if escaped:
            out.append(ch)
            escaped = False
            i += 1
            continue
        if in_double:
            out.append(ch)
            if ch == "\\":
                escaped = True
            elif ch == '"':
                in_double = False
            i += 1
            continue
        if ch == '"':
            in_double = True
            out.append(ch)
            prev_significant = ch
            i += 1
            continue
        if ch == "'" and prev_significant in "{[,:":
            # Find the closing quote, honouring \' escapes.
            j = i + 1
            buf: list[str] = []
            closed = False
            while j < len(text):
                c = text[j]
                if c == "\\" and j + 1 < len(text) and text[j + 1] == "'":
                    buf.append("'")
                    j += 2
                    continue
                if c == "'":
                    closed = True
                    break
                buf.append(c)
                j += 1
            if not closed:
                out.append(ch)
                prev_significant = ch
                i += 1
                continue
            content = "".join(buf)
            content = (content.replace("\\", "\\\\")
                              .replace('"', '\\"')
                              .replace("\n", "\\n")
                              .replace("\r", "\\r")
                              .replace("\t", "\\t"))
            out.append(f'"{content}"')
            changed = True
            prev_significant = '"'
            i = j + 1
            continue
        out.append(ch)
        if not ch.isspace():
            prev_significant = ch
        i += 1
    return "".join(out), changed


@registry.tool(
    name="repair_json",
    description=("Extract and repair the first JSON object or array from messy "
                 "LLM output text: strips markdown code fences and surrounding "
                 "prose, removes trailing commas and converts single-quoted "
                 "strings. Returns the parsed value plus the list of repairs "
                 "applied; raises ValueError if the JSON is unrecoverable."),
    parameters={
        "type": "object",
        "properties": {
            "text": {"type": "string",
                     "description": "The raw text that should contain JSON."},
        },
        "required": ["text"],
    },
    tags=["text", "json"],
)
def repair_json(text: str) -> dict:
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    repairs: list[str] = []

    candidate, fenced = _strip_code_fence(text)
    if fenced:
        repairs.append("removed markdown code fence")

    candidate, surrounding = _extract_first_value(candidate)
    if candidate is None:
        raise ValueError("no JSON object or array found in text")
    if surrounding:
        repairs.append("extracted JSON from surrounding text")

    last_error: Exception | None = None
    try:
        return {"data": json.loads(candidate), "repaired": bool(repairs),
                "repairs": repairs}
    except json.JSONDecodeError as exc:
        last_error = exc

    candidate, changed = _remove_trailing_commas(candidate)
    if changed:
        repairs.append("removed trailing commas")
        try:
            return {"data": json.loads(candidate), "repaired": True,
                    "repairs": repairs}
        except json.JSONDecodeError as exc:
            last_error = exc

    candidate, changed = _convert_single_quotes(candidate)
    if changed:
        repairs.append("converted single-quoted strings to double-quoted")
        try:
            return {"data": json.loads(candidate), "repaired": True,
                    "repairs": repairs}
        except json.JSONDecodeError as exc:
            last_error = exc

    raise ValueError(
        f"could not repair JSON: {last_error}. "
        f"Repairs attempted: {repairs or 'none'}")
