from pathlib import Path
import math
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from timerec.data import DTYPE, download, history, load_ratings, temporal_split
from timerec.evaluate import evaluate, paired_interval, ranking_metrics, top_items
from timerec.model import popularity, train_bpr


def example_rows():
    return np.array([(u, i, 5 if (u + i) % 3 else 2, (i * 7 + u * 3))
                     for u in range(1, 7) for i in range(1, 13)], dtype=DTYPE)


class TemporalTests(unittest.TestCase):
    def test_global_time_prevents_cross_user_future_training(self):
        train, validation, test, _ = temporal_split(example_rows())
        self.assertLess(train["timestamp"].max(), validation["timestamp"].min())
        self.assertLess(validation["timestamp"].max(), test["timestamp"].min())
        self.assertEqual(len(train) + len(validation) + len(test), 72)

    def test_equal_timestamp_never_straddles_boundaries(self):
        rows = example_rows()
        rows["timestamp"] //= 9
        partitions = temporal_split(rows)[:3]
        for i in range(3):
            for j in range(i + 1, 3):
                self.assertFalse(set(partitions[i]["timestamp"]) & set(partitions[j]["timestamp"]))

    def test_input_permutation_does_not_change_split(self):
        rows = example_rows()
        first = temporal_split(rows)
        second = temporal_split(rows[np.random.default_rng(9).permutation(len(rows))])
        for a, b in zip(first[:3], second[:3]):
            np.testing.assert_array_equal(a, b)

    def test_duplicate_ratings_rejected_instead_of_silently_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.tsv"
            path.write_text("1 2 4 20\n1 2 5 22\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_ratings(path)

    def test_invalid_ratings_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.tsv"
            path.write_text("1 2 6 20\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid"):
                load_ratings(path)

    def test_cached_dataset_must_match_pinned_digest_before_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "ml-100k.zip").write_bytes(b"not-the-verified-dataset")
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                download(Path(tmp))
            self.assertFalse((Path(tmp) / "u.data").exists())

    def test_history_masks_low_ratings_as_well_as_positive_ratings(self):
        rows = np.array([(1, 1, 1, 1), (1, 2, 5, 2)], dtype=DTYPE)
        self.assertEqual(top_items([99, 98, 1], [1, 2, 3], history(rows)[1]), [3])

    def test_deterministic_tie_break_is_by_item_identifier(self):
        self.assertEqual(top_items([1, 1, float("nan")], [9, 3, 2], set()), [3, 9])

    def test_metrics_have_known_denominators_and_discount(self):
        metrics = ranking_metrics([2, 8, 4], {2, 4, 10})
        self.assertAlmostEqual(metrics["recall_at_10"], 2 / 3)
        expected = (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3) + 1 / math.log2(4))
        self.assertAlmostEqual(metrics["ndcg_at_10"], expected)
        with self.assertRaises(ValueError):
            ranking_metrics([], set())

    def test_cold_start_is_excluded_and_counted_not_granted_future_embedding(self):
        prior = np.array([(1, 1, 5, 1)], dtype=DTYPE)
        heldout = np.array([(1, 2, 5, 2), (2, 2, 5, 3), (1, 99, 5, 4)], dtype=DTYPE)
        metrics, records = evaluate(heldout, prior, [1], [1, 2], lambda user: np.array([1, 2]))
        self.assertEqual(metrics["users_evaluated"], 1)
        self.assertEqual(metrics["excluded_cold_users"], 1)
        self.assertEqual(metrics["supported_positive_ratings"], 1)
        self.assertEqual(metrics["heldout_positive_ratings"], 3)
        self.assertAlmostEqual(metrics["positive_target_coverage"], 1 / 3)
        self.assertEqual(records[0]["recall_at_10"], 1)

    def test_empty_eligible_population_is_an_explicit_error(self):
        heldout = np.array([(2, 2, 5, 3)], dtype=DTYPE)
        prior = np.array([(1, 1, 5, 1)], dtype=DTYPE)
        with self.assertRaisesRegex(ValueError, "No warm"):
            evaluate(heldout, prior, [1], [1], lambda _: np.array([1]))

    def test_explicit_fallback_can_include_cold_users_without_future_factors(self):
        prior = np.array([(1, 1, 5, 1)], dtype=DTYPE)
        heldout = np.array([(2, 2, 5, 3)], dtype=DTYPE)
        metrics, _ = evaluate(heldout, prior, [1], [1, 2], lambda _: np.array([1, 2]), allow_cold_users=True)
        self.assertEqual(metrics["cold_users_evaluated"], 1)
        self.assertEqual(metrics["positive_target_coverage"], 1)
        self.assertEqual(metrics["recall_at_10"], 1)

    def test_unavailable_target_count_covers_warm_and_cold_users(self):
        prior = np.array([(1, 1, 5, 1)], dtype=DTYPE)
        heldout = np.array([(1, 99, 5, 2), (2, 99, 5, 3), (3, 2, 5, 4)], dtype=DTYPE)
        metrics, _ = evaluate(heldout, prior, [1], [1, 2], lambda _: np.array([1, 2]), allow_cold_users=True)
        self.assertEqual(metrics["excluded_users_without_available_positive"], 2)
        self.assertNotIn("excluded_warm_users_without_available_positive", metrics)
        self.assertEqual(metrics["users_evaluated"], 1)

    def test_future_only_items_cannot_enter_training_model_or_negative_pool(self):
        rows = example_rows()
        train, _, _, _ = temporal_split(rows)
        model, _ = train_bpr(train, factors=4, epochs=1)
        self.assertEqual(model.items, sorted(set(map(int, train["item"]))))
        self.assertNotIn(999, model.item_map)

    def test_torch_training_reproducible_for_fixed_seed(self):
        train = temporal_split(example_rows())[0]
        first, losses_a = train_bpr(train, factors=4, epochs=2, seed=7)
        second, losses_b = train_bpr(train, factors=4, epochs=2, seed=7)
        np.testing.assert_array_equal(first.scores(1), second.scores(1))
        self.assertEqual(losses_a, losses_b)

    def test_observed_negative_only_user_requires_fallback_not_random_factor(self):
        rows = np.array([(1, 1, 5, 1), (1, 2, 1, 2), (2, 2, 1, 3)], dtype=DTYPE)
        model, _ = train_bpr(rows, factors=4, epochs=1)
        self.assertIn(2, model.user_map)
        self.assertNotIn(2, model.trained_users)
        with self.assertRaisesRegex(ValueError, "fallback"):
            model.scores(2)

    def test_training_without_a_contrastive_candidate_fails_clearly(self):
        rows = np.array([(1, 1, 5, 1)], dtype=DTYPE)
        with self.assertRaisesRegex(ValueError, "non-positive"):
            train_bpr(rows, factors=4, epochs=1)

    def test_popularity_uses_only_positive_training_ratings(self):
        rows = np.array([(1, 1, 1, 1), (2, 1, 4, 1), (3, 2, 5, 1)], dtype=DTYPE)
        np.testing.assert_array_equal(popularity(rows, [1, 2, 3]), [1, 1, 0])

    def test_bootstrap_requires_matching_population_and_is_reproducible(self):
        a = [{"evaluation_user": 1, "ndcg_at_10": .6}, {"evaluation_user": 2, "ndcg_at_10": .4}]
        b = [{"evaluation_user": 1, "ndcg_at_10": .3}, {"evaluation_user": 2, "ndcg_at_10": .2}]
        self.assertEqual(paired_interval(a, b), paired_interval(a, b))
        self.assertAlmostEqual(paired_interval(a, b)["mean_difference"], .25)
        with self.assertRaises(ValueError):
            paired_interval(a, list(reversed(b)))


if __name__ == "__main__":
    unittest.main()
