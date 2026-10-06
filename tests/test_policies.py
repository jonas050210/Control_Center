"""Policy cache behaviour: LRU eviction and safe fallbacks."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from server.policies import HEURISTIC, MAX_CACHED_POLICIES, PolicyCache, policy_choices


class PolicyCacheTests(unittest.TestCase):
    def test_heuristic_needs_no_model_file(self) -> None:
        cache = PolicyCache()
        model, normalizer, error = cache.load(HEURISTIC, Path("/nonexistent"))
        self.assertIsNone(model)
        self.assertIsNone(normalizer)
        self.assertIsNone(error)

    def test_missing_model_reports_a_message_instead_of_raising(self) -> None:
        cache = PolicyCache()
        with tempfile.TemporaryDirectory() as folder:
            _, _, error = cache.load("best_model.zip", Path(folder))
        self.assertIn("not found", error)

    def test_cache_evicts_the_least_recently_used_entry(self) -> None:
        cache = PolicyCache(max_entries=2)
        cache._remember("a", ("model-a", None, None))
        cache._remember("b", ("model-b", None, None))
        cache._remember("c", ("model-c", None, None))
        self.assertEqual(list(cache._cache), ["b", "c"])

        cache._remember("b", ("model-b", None, None))  # refresh "b"
        cache._remember("d", ("model-d", None, None))
        self.assertEqual(list(cache._cache), ["b", "d"])

    def test_eviction_can_be_disabled_by_a_large_cap(self) -> None:
        cache = PolicyCache(max_entries=MAX_CACHED_POLICIES)
        for index in range(MAX_CACHED_POLICIES + 5):
            cache._remember(f"model-{index}", (index, None, None))
        self.assertEqual(len(cache._cache), MAX_CACHED_POLICIES)

    def test_policy_choices_always_start_with_the_heuristic(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            models = Path(folder)
            (models / "run_a.zip").write_bytes(b"")
            (models / "run_a_vecnormalize.pkl").write_bytes(b"")
            (models / "notes.txt").write_text("not a model", encoding="utf-8")
            choices = policy_choices(models)
        self.assertEqual(choices[0], HEURISTIC)
        self.assertIn("run_a.zip", choices)
        self.assertNotIn("notes.txt", choices)


if __name__ == "__main__":
    unittest.main()
