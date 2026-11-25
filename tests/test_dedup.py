import unittest

from llm_text_toolbox.tools.dedup import dedup_texts, jaccard, word_shingles

BASE = ("the quick brown fox jumps over the lazy dog near the river bank "
        "while the sun rises over the quiet hills of the valley")


class ShingleAndJaccardTestCase(unittest.TestCase):
    def test_shingles_are_word_trigrams(self):
        shingles = word_shingles("a b c d", 3)
        self.assertEqual(shingles, {("a", "b", "c"), ("b", "c", "d")})

    def test_short_text_is_one_shingle(self):
        self.assertEqual(word_shingles("only two", 3), {("only", "two")})

    def test_empty_text_has_no_shingles(self):
        self.assertEqual(word_shingles("   ", 3), set())

    def test_jaccard_extremes(self):
        self.assertEqual(jaccard({1, 2}, {1, 2}), 1.0)
        self.assertEqual(jaccard({1}, {2}), 0.0)
        self.assertEqual(jaccard(set(), set()), 1.0)
        self.assertEqual(jaccard(set(), {1}), 0.0)


class ClusteringTestCase(unittest.TestCase):
    def test_identical_texts_cluster(self):
        result = dedup_texts([BASE, BASE, "something else entirely here"])
        self.assertEqual(result["group_count"], 1)
        self.assertEqual(result["groups"][0]["indices"], [0, 1])
        self.assertEqual(result["duplicate_count"], 2)
        self.assertEqual(result["text_count"], 3)

    def test_small_edit_still_clusters(self):
        edited = BASE.replace("quick", "swift")
        result = dedup_texts([BASE, edited])
        self.assertEqual(result["group_count"], 1)

    def test_unrelated_texts_do_not_cluster(self):
        result = dedup_texts([
            BASE,
            "quantum computers use qubits and superposition for calculation",
            "the recipe calls for flour sugar butter eggs and vanilla",
        ])
        self.assertEqual(result["group_count"], 0)
        self.assertEqual(result["duplicate_count"], 0)

    def test_transitive_clustering(self):
        # A ~ B and B ~ C should land in one group even if A !~ C directly.
        a = "one two three four five six seven eight nine ten"
        b = "one two three four five six seven eight nine eleven"
        c = "one two three four five six seven eight twelve eleven"
        result = dedup_texts([a, b, c], threshold=0.5)
        self.assertEqual(result["group_count"], 1)
        self.assertEqual(result["groups"][0]["indices"], [0, 1, 2])

    def test_empty_and_single_inputs(self):
        self.assertEqual(dedup_texts([])["group_count"], 0)
        self.assertEqual(dedup_texts([BASE])["group_count"], 0)


class ThresholdTestCase(unittest.TestCase):
    def test_strict_threshold_separates_near_duplicates(self):
        edited = BASE.replace("quick", "swift")
        result = dedup_texts([BASE, edited], threshold=0.99)
        self.assertEqual(result["group_count"], 0)

    def test_threshold_one_still_matches_identical(self):
        result = dedup_texts([BASE, BASE], threshold=1.0)
        self.assertEqual(result["group_count"], 1)

    def test_invalid_threshold_rejected(self):
        with self.assertRaises(ValueError):
            dedup_texts([BASE], threshold=0)
        with self.assertRaises(ValueError):
            dedup_texts([BASE], threshold=1.5)

    def test_invalid_shingle_size_rejected(self):
        with self.assertRaises(ValueError):
            dedup_texts([BASE], shingle_size=0)

    def test_non_string_items_rejected(self):
        with self.assertRaises(ValueError):
            dedup_texts([BASE, 42])


if __name__ == "__main__":
    unittest.main()
