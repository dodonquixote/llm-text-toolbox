import unittest

from llm_text_toolbox.tools.chunking import chunk_text


class BasicChunkingTestCase(unittest.TestCase):
    def test_short_text_is_one_chunk(self):
        result = chunk_text("One two three.", chunk_size=10, overlap=2)
        self.assertEqual(result["chunk_count"], 1)
        self.assertEqual(result["chunks"], ["One two three."])
        self.assertEqual(result["total_words"], 3)

    def test_empty_text(self):
        result = chunk_text("   ", chunk_size=10, overlap=2)
        self.assertEqual(result["chunks"], [])
        self.assertEqual(result["chunk_count"], 0)
        self.assertEqual(result["total_words"], 0)

    def test_no_chunk_exceeds_chunk_size(self):
        text = " ".join(f"word{i}" for i in range(95)) + "."
        result = chunk_text(text, chunk_size=10, overlap=3)
        self.assertGreater(result["chunk_count"], 5)
        for chunk in result["chunks"]:
            self.assertLessEqual(len(chunk.split()), 10)

    def test_all_words_covered_without_overlap(self):
        text = " ".join(f"w{i}" for i in range(30)) + "."
        result = chunk_text(text, chunk_size=10, overlap=0)
        covered = " ".join(result["chunks"]).split()
        self.assertEqual(covered, text.split())
        self.assertEqual(len(covered), 30)


class OverlapTestCase(unittest.TestCase):
    def test_overlap_words_repeat_between_chunks(self):
        text = " ".join(f"w{i}" for i in range(20)) + "."
        result = chunk_text(text, chunk_size=10, overlap=3)
        first = result["chunks"][0].split()
        second = result["chunks"][1].split()
        self.assertEqual(second[:3], first[-3:])

    def test_zero_overlap_means_no_repetition(self):
        text = " ".join(f"w{i}" for i in range(20)) + "."
        result = chunk_text(text, chunk_size=10, overlap=0)
        first = result["chunks"][0].split()
        second = result["chunks"][1].split()
        self.assertFalse(set(first) & set(second))


class BoundaryTestCase(unittest.TestCase):
    def test_sentences_are_not_split_when_they_fit(self):
        text = ("Alpha beta gamma. Delta epsilon zeta. "
                "Eta theta iota. Kappa lambda mu.")
        result = chunk_text(text, chunk_size=6, overlap=0)
        self.assertEqual(result["chunk_count"], 2)
        for chunk in result["chunks"]:
            self.assertTrue(chunk.endswith("."))

    def test_oversized_sentence_is_hard_split(self):
        text = " ".join(["long"] * 25)  # one sentence, no punctuation
        result = chunk_text(text, chunk_size=10, overlap=0)
        self.assertEqual(result["chunk_count"], 3)
        self.assertEqual([len(c.split()) for c in result["chunks"]],
                         [10, 10, 5])


class ValidationTestCase(unittest.TestCase):
    def test_overlap_must_be_smaller_than_chunk_size(self):
        with self.assertRaises(ValueError):
            chunk_text("some text here", chunk_size=5, overlap=5)

    def test_negative_overlap_rejected(self):
        with self.assertRaises(ValueError):
            chunk_text("some text here", chunk_size=5, overlap=-1)

    def test_zero_chunk_size_rejected(self):
        with self.assertRaises(ValueError):
            chunk_text("some text here", chunk_size=0, overlap=0)


if __name__ == "__main__":
    unittest.main()
