import unittest

from llm_text_toolbox.tools.summary import extractive_summary

CATS = (
    "Cats are small domesticated mammals kept as pets around the world. "
    "The weather in Jakarta was rainy yesterday afternoon. "
    "Cats sleep for most of the day and cats hunt at night by instinct. "
    "A completely unrelated sentence about quantum computing hardware. "
    "Because cats groom themselves constantly, cats stay remarkably clean."
)


class SelectionTestCase(unittest.TestCase):
    def test_picks_topic_sentences(self):
        result = extractive_summary(CATS, sentence_count=2)
        self.assertEqual(result["sentence_count"], 2)
        self.assertEqual(result["total_sentences"], 5)
        for sentence in result["sentences"]:
            self.assertIn("cats", sentence.lower())

    def test_top_one_is_a_cat_sentence(self):
        result = extractive_summary(CATS, sentence_count=1)
        self.assertIn("cats", result["sentences"][0].lower())

    def test_original_order_preserved(self):
        result = extractive_summary(CATS, sentence_count=3)
        positions = [CATS.index(s) for s in result["sentences"]]
        self.assertEqual(positions, sorted(positions))

    def test_summary_is_the_sentences_joined(self):
        result = extractive_summary(CATS, sentence_count=2)
        self.assertEqual(result["summary"], " ".join(result["sentences"]))


class EdgeCaseTestCase(unittest.TestCase):
    def test_fewer_sentences_than_requested_returns_all(self):
        result = extractive_summary("Only one sentence here.",
                                    sentence_count=5)
        self.assertEqual(result["sentence_count"], 1)
        self.assertEqual(result["total_sentences"], 1)

    def test_empty_text(self):
        result = extractive_summary("")
        self.assertEqual(result["summary"], "")
        self.assertEqual(result["sentence_count"], 0)
        self.assertEqual(result["total_sentences"], 0)

    def test_invalid_sentence_count(self):
        with self.assertRaises(ValueError):
            extractive_summary(CATS, sentence_count=0)


if __name__ == "__main__":
    unittest.main()
