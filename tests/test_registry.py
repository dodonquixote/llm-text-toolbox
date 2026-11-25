import unittest

from llm_text_toolbox import registry

ALL_TOOLS = {"repair_json", "chunk_text", "estimate_tokens",
             "dedup_texts", "extractive_summary"}


class TestRegistry(unittest.TestCase):
    def test_all_bundled_tools_registered(self):
        self.assertEqual(set(registry.names()), ALL_TOOLS)

    def test_function_definition_shape(self):
        defs = {d["function"]["name"]: d
                for d in registry.function_definitions()}
        self.assertEqual(set(defs), ALL_TOOLS)
        for definition in defs.values():
            self.assertEqual(definition["type"], "function")
            params = definition["function"]["parameters"]
            self.assertEqual(params["type"], "object")
            self.assertTrue(params["required"])

    def test_repair_json_definition(self):
        defs = {d["function"]["name"]: d
                for d in registry.function_definitions()}
        params = defs["repair_json"]["function"]["parameters"]
        self.assertIn("text", params["properties"])
        self.assertEqual(params["required"], ["text"])

    def test_call_unknown_tool(self):
        with self.assertRaises(KeyError):
            registry.call("does_not_exist", {})

    def test_call_accepts_json_string(self):
        result = registry.call("estimate_tokens", '{"text": "Hello world"}')
        self.assertGreaterEqual(result["estimated_tokens"], 1)

    def test_call_dispatches(self):
        result = registry.call("chunk_text",
                               {"text": "One two three.",
                                "chunk_size": 10, "overlap": 2})
        self.assertEqual(result["chunk_count"], 1)


if __name__ == "__main__":
    unittest.main()
