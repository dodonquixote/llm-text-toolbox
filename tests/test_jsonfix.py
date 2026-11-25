import unittest

from llm_text_toolbox.tools.jsonfix import repair_json


class CleanJsonTestCase(unittest.TestCase):
    def test_clean_object_needs_no_repairs(self):
        result = repair_json('{"name": "Ada", "age": 36}')
        self.assertEqual(result["data"], {"name": "Ada", "age": 36})
        self.assertFalse(result["repaired"])
        self.assertEqual(result["repairs"], [])

    def test_clean_array(self):
        result = repair_json('[1, 2, 3]')
        self.assertEqual(result["data"], [1, 2, 3])
        self.assertFalse(result["repaired"])


class FenceAndProseTestCase(unittest.TestCase):
    def test_markdown_fence_stripped(self):
        text = '```json\n{"a": 1}\n```'
        result = repair_json(text)
        self.assertEqual(result["data"], {"a": 1})
        self.assertIn("removed markdown code fence", result["repairs"])

    def test_bare_fence_stripped(self):
        result = repair_json('```\n[true, false]\n```')
        self.assertEqual(result["data"], [True, False])

    def test_surrounding_prose_extracted(self):
        text = 'Sure! Here is the data you asked for: {"ok": true} Hope it helps.'
        result = repair_json(text)
        self.assertEqual(result["data"], {"ok": True})
        self.assertIn("extracted JSON from surrounding text", result["repairs"])

    def test_brackets_inside_strings_do_not_confuse_extraction(self):
        text = 'pre {"note": "a } brace and [ bracket"} post'
        result = repair_json(text)
        self.assertEqual(result["data"], {"note": "a } brace and [ bracket"})


class RepairTestCase(unittest.TestCase):
    def test_trailing_commas_removed(self):
        result = repair_json('{"a": 1, "b": [1, 2, 3,],}')
        self.assertEqual(result["data"], {"a": 1, "b": [1, 2, 3]})
        self.assertIn("removed trailing commas", result["repairs"])
        self.assertTrue(result["repaired"])

    def test_single_quoted_keys_and_strings(self):
        result = repair_json("{'name': 'Ada', 'tags': ['x', 'y']}")
        self.assertEqual(result["data"], {"name": "Ada", "tags": ["x", "y"]})
        self.assertIn("converted single-quoted strings to double-quoted",
                      result["repairs"])

    def test_single_quoted_string_with_double_quote_inside(self):
        result = repair_json("""{'quote': 'she said "hi"'}""")
        self.assertEqual(result["data"], {"quote": 'she said "hi"'})

    def test_combined_breakage(self):
        text = "Output: ```json\n{'a': 1, 'b': 2,}\n``` done"
        result = repair_json(text)
        self.assertEqual(result["data"], {"a": 1, "b": 2})
        self.assertTrue(result["repaired"])
        self.assertGreaterEqual(len(result["repairs"]), 3)


class FailureTestCase(unittest.TestCase):
    def test_no_json_at_all(self):
        with self.assertRaises(ValueError):
            repair_json("there is no data here at all")

    def test_unrecoverable_json(self):
        with self.assertRaises(ValueError):
            repair_json('{"a": }')

    def test_non_string_input(self):
        with self.assertRaises(ValueError):
            repair_json(42)


if __name__ == "__main__":
    unittest.main()
