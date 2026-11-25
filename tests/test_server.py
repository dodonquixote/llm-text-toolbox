"""Tests for the MCP server (llm_text_toolbox/server.py).

All tests are in-process and offline: requests are fed to
handle_request() / run_server() directly, so no network and no MCP
client are needed.
"""
import io
import json
import unittest

from llm_text_toolbox import registry
from llm_text_toolbox.server import (
    PROTOCOL_VERSION,
    SERVER_NAME,
    handle_request,
    run_server,
)


def _request(method, params=None, request_id=1):
    req = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        req["params"] = params
    if request_id is not None:
        req["id"] = request_id
    return req


class InitializeTestCase(unittest.TestCase):
    def test_initialize_handshake(self):
        response = handle_request(_request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {}, "clientInfo": {"name": "test", "version": "0"}}))
        self.assertEqual(response["id"], 1)
        result = response["result"]
        self.assertEqual(result["protocolVersion"], PROTOCOL_VERSION)
        self.assertEqual(result["serverInfo"]["name"], SERVER_NAME)
        self.assertIn("tools", result["capabilities"])

    def test_ping(self):
        response = handle_request(_request("ping"))
        self.assertEqual(response["result"], {})


class ToolsListTestCase(unittest.TestCase):
    def test_lists_every_registered_tool(self):
        response = handle_request(_request("tools/list"))
        tools = response["result"]["tools"]
        listed = {tool["name"] for tool in tools}
        self.assertEqual(listed, set(registry.names()))
        self.assertTrue(listed >= {"repair_json", "chunk_text",
                                   "estimate_tokens", "dedup_texts",
                                   "extractive_summary"})
        for tool in tools:
            self.assertIn("description", tool)
            self.assertEqual(tool["inputSchema"]["type"], "object")


class ToolsCallTestCase(unittest.TestCase):
    def test_call_repair_json(self):
        response = handle_request(_request("tools/call", {
            "name": "repair_json",
            "arguments": {"text": "Here you go: {'a': 1,} — done"}}))
        self.assertFalse(response["result"]["isError"])
        result = json.loads(response["result"]["content"][0]["text"])
        self.assertEqual(result["data"], {"a": 1})
        self.assertTrue(result["repaired"])

    def test_call_unknown_tool_returns_error_not_crash(self):
        response = handle_request(_request("tools/call", {
            "name": "no_such_tool", "arguments": {}}))
        self.assertTrue(response["result"]["isError"])
        self.assertIn("Unknown tool", response["result"]["content"][0]["text"])

    def test_call_missing_name_is_invalid_params(self):
        response = handle_request(_request("tools/call", {"arguments": {}}))
        self.assertEqual(response["error"]["code"], -32602)


class ProtocolErrorTestCase(unittest.TestCase):
    def test_unknown_method(self):
        response = handle_request(_request("resources/list"))
        self.assertEqual(response["error"]["code"], -32601)

    def test_notification_gets_no_reply(self):
        # A request without "id" (e.g. notifications/initialized) is a
        # notification: the server must stay silent.
        response = handle_request(_request("notifications/initialized",
                                           request_id=None))
        self.assertIsNone(response)

    def test_not_an_object_is_invalid_request(self):
        response = handle_request(["not", "an", "object"])
        self.assertEqual(response["error"]["code"], -32600)


class StdioLoopTestCase(unittest.TestCase):
    def test_run_server_round_trip_and_bad_json(self):
        lines = [
            json.dumps(_request("initialize", {}, request_id=1)),
            json.dumps(_request("notifications/initialized", request_id=None)),
            "{not valid json",
            json.dumps(_request("tools/list", request_id=2)),
        ]
        stdin = io.StringIO("\n".join(lines) + "\n")
        stdout = io.StringIO()
        run_server(stdin, stdout)
        responses = [json.loads(line) for line in stdout.getvalue().splitlines()]
        # initialize reply, parse-error reply, tools/list reply;
        # the notification produced no line.
        self.assertEqual(len(responses), 3)
        self.assertEqual(responses[0]["id"], 1)
        self.assertEqual(responses[1]["error"]["code"], -32700)
        self.assertEqual(responses[2]["id"], 2)
        self.assertIn("tools", responses[2]["result"])


if __name__ == "__main__":
    unittest.main()
