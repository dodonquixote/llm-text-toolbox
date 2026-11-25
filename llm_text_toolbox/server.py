"""MCP server for llm-text-toolbox.

Exposes every registered tool over the Model Context Protocol using
JSON-RPC 2.0 on stdio — standard library only, no MCP SDK needed.

Protocol surface (MCP 2024-11-05):
- ``initialize`` — handshake; returns protocol version, capabilities
  and server info.
- ``tools/list`` — every registered tool as an MCP tool definition
  (name, description, inputSchema).
- ``tools/call`` — run one tool by name; the result is returned as a
  text content block holding the tool's JSON result.
- ``ping`` — liveness check.
- Notifications (requests without an ``id``, e.g.
  ``notifications/initialized``) are accepted and get no reply.

Run it:

    python -m llm_text_toolbox.server

Point an MCP client at it with ``mcp_config_template.json`` (see the
README's "MCP server" section). The server reads one JSON-RPC message
per line on stdin and writes one response per line on stdout.
"""
from __future__ import annotations

import json
import sys
from typing import Any, TextIO

from . import registry

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "llm-text-toolbox"
SERVER_VERSION = "0.1.0"

# JSON-RPC error codes.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


def _error(code: int, message: str, request_id: Any) -> dict[str, Any]:
    """Build a JSON-RPC error response object."""
    return {"jsonrpc": "2.0", "id": request_id,
            "error": {"code": code, "message": message}}


def _result(result: Any, request_id: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _mcp_tool_definitions() -> list[dict[str, Any]]:
    """Registry tools in MCP ``tools/list`` shape."""
    definitions = []
    for definition in registry.function_definitions():
        fn = definition["function"]
        definitions.append({
            "name": fn["name"],
            "description": fn["description"],
            "inputSchema": fn["parameters"],
        })
    return definitions


def _handle_initialize(params: Any) -> dict[str, Any]:
    """Handshake: declare the protocol version and capabilities."""
    return {
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {"tools": {}},
        "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
    }


def _handle_tools_list(params: Any) -> dict[str, Any]:
    return {"tools": _mcp_tool_definitions()}


def _handle_tools_call(params: Any) -> dict[str, Any]:
    """Run one tool and return its result as a text content block."""
    if not isinstance(params, dict):
        raise ValueError("tools/call params must be an object")
    name = params.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("tools/call params.name must be a non-empty string")
    arguments = params.get("arguments", {})
    if not isinstance(arguments, dict):
        raise ValueError("tools/call params.arguments must be an object")
    try:
        result = registry.call(name, arguments)
    except Exception as exc:  # noqa: BLE001 - report tool failures, don't crash the server
        return {"content": [{"type": "text",
                             "text": f"{type(exc).__name__}: {exc}"}],
                "isError": True}
    return {"content": [{"type": "text",
                         "text": json.dumps(result, ensure_ascii=False)}],
            "isError": False}


_METHODS = {
    "initialize": _handle_initialize,
    "tools/list": _handle_tools_list,
    "tools/call": _handle_tools_call,
    "ping": lambda params: {},
}


def handle_request(request: Any) -> dict[str, Any] | None:
    """Handle one decoded JSON-RPC message.

    Returns the response object, or None when no reply is due (a
    notification, i.e. a request without an ``id``).
    """
    if not isinstance(request, dict) or request.get("jsonrpc") != "2.0":
        return _error(INVALID_REQUEST, "request must be a JSON-RPC 2.0 object",
                      request.get("id") if isinstance(request, dict) else None)
    request_id = request.get("id")
    is_notification = "id" not in request
    method = request.get("method")
    params = request.get("params", {})

    if not isinstance(method, str):
        if is_notification:
            return None
        return _error(INVALID_REQUEST, "method must be a string", request_id)

    handler = _METHODS.get(method)
    if handler is None:
        # Unknown methods (including notifications like
        # notifications/initialized) get no reply when they are
        # notifications; a request gets METHOD_NOT_FOUND.
        if is_notification:
            return None
        return _error(METHOD_NOT_FOUND, f"method not found: {method}", request_id)

    try:
        result = handler(params)
    except ValueError as exc:
        if is_notification:
            return None
        return _error(INVALID_PARAMS, str(exc), request_id)
    except Exception as exc:  # noqa: BLE001 - a handler bug must not kill the server
        if is_notification:
            return None
        return _error(INTERNAL_ERROR, f"{type(exc).__name__}: {exc}", request_id)

    if is_notification:
        return None
    return _result(result, request_id)


def run_server(input_stream: TextIO, output_stream: TextIO) -> None:
    """Serve JSON-RPC over stdio: one message per line, until EOF."""
    for line in input_stream:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            response = _error(PARSE_ERROR, f"invalid JSON: {exc}", None)
            output_stream.write(json.dumps(response) + "\n")
            output_stream.flush()
            continue
        response = handle_request(request)
        if response is not None:
            output_stream.write(json.dumps(response) + "\n")
            output_stream.flush()


def main() -> None:
    run_server(sys.stdin, sys.stdout)


if __name__ == "__main__":
    main()
