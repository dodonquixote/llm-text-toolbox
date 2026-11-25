import unittest

from llm_text_toolbox.tools.tokens import estimate_tokens


class EstimateTestCase(unittest.TestCase):
    def test_empty_text_is_zero(self):
        result = estimate_tokens("")
        self.assertEqual(result["estimated_tokens"], 0)
        self.assertEqual(result["characters"], 0)
        self.assertEqual(result["words"], 0)

    def test_whitespace_only_is_zero(self):
        self.assertEqual(estimate_tokens("   \n  ")["estimated_tokens"], 0)

    def test_short_text_in_sane_range(self):
        # "Hello world" is 2 tokens for real BPE tokenizers; a heuristic
        # landing anywhere in 1..5 is doing its job.
        result = estimate_tokens("Hello world")
        self.assertGreaterEqual(result["estimated_tokens"], 1)
        self.assertLessEqual(result["estimated_tokens"], 5)
        self.assertEqual(result["words"], 2)
        self.assertEqual(result["characters"], 11)

    def test_longer_text_means_more_tokens(self):
        short = estimate_tokens("The cat sat on the mat.")
        long = estimate_tokens("The cat sat on the mat. " * 40)
        self.assertGreater(long["estimated_tokens"],
                           short["estimated_tokens"] * 10)

    def test_roughly_linear_scaling(self):
        once = estimate_tokens("Some sample prose for estimation. " * 10)
        twice = estimate_tokens("Some sample prose for estimation. " * 20)
        ratio = twice["estimated_tokens"] / once["estimated_tokens"]
        self.assertGreater(ratio, 1.8)
        self.assertLess(ratio, 2.2)


class CostTestCase(unittest.TestCase):
    def test_no_price_means_no_cost(self):
        result = estimate_tokens("Hello world")
        self.assertIsNone(result["estimated_cost"])
        self.assertIsNone(result["price_per_1k_tokens"])

    def test_cost_follows_price(self):
        result = estimate_tokens("word " * 750, price_per_1k_tokens=2.0)
        expected = result["estimated_tokens"] / 1000 * 2.0
        self.assertAlmostEqual(result["estimated_cost"], expected, places=6)
        self.assertGreater(result["estimated_cost"], 0)

    def test_zero_price_is_free(self):
        result = estimate_tokens("Hello world", price_per_1k_tokens=0)
        self.assertEqual(result["estimated_cost"], 0.0)

    def test_negative_price_rejected(self):
        with self.assertRaises(ValueError):
            estimate_tokens("Hello", price_per_1k_tokens=-1.0)


if __name__ == "__main__":
    unittest.main()
