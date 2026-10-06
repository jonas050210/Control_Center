"""Policy cache behaviour: LRU eviction and safe fallbacks."""

from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from server.policies import HEURISTIC, MAX_CACHED_POLICIES, PolicyCache, policy_choices


class _StrippedNormalizer:
    """Mimics ``VecNormalize.save``: the wrapped env is gone after unpickling.

    SB3's ``VecEnv.__getattr__`` then recurses until the interpreter gives up, so
    every missing attribute raises ``RecursionError`` - exactly the bug that made
    the automatic evaluation of a freshly trained checkpoint fail.
    """

    def __init__(self, mean, var, norm_obs):
        self.obs_rms = SimpleNamespace(mean=mean, var=var)
        self.norm_obs = norm_obs
        self.clip_obs = 10.0

    def __getattr__(self, name):  # only called when the attribute is missing
        raise RecursionError(f"maximum recursion depth exceeded ({name})")

    def save(self, path):
        state = dict(self.__dict__)
        del state["obs_rms"]          # the env (and its rms) are not written either
        with open(path, "wb") as handle:
            pickle.dump(SimpleNamespace(**state), handle)


class NormalizerLoadingTests(unittest.TestCase):
    def test_stripped_sidecar_neither_crashes_nor_normalises(self) -> None:
        from training.evaluation import load_normalizer, normalize

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "run_vecnormalize.pkl"
            payload = SimpleNamespace(obs_rms=SimpleNamespace(mean=[0.0, 1.0], var=[1.0, 4.0]),
                                      norm_obs=True, clip_obs=10.0)
            with path.open("wb") as handle:
                pickle.dump(payload, handle)
            statistics = load_normalizer(path)
        self.assertIsNotNone(statistics)
        self.assertTrue(statistics.norm_obs)
        observation = np.asarray([2.0, 5.0], dtype=np.float32)
        scaled = normalize(observation, statistics)
        self.assertAlmostEqual(float(scaled[0]), 2.0, places=5)
        self.assertAlmostEqual(float(scaled[1]), 2.0, places=5)
        # A sidecar written by ``VecNormalize.save`` only keeps the raw numbers.
        self.assertAlmostEqual(float(statistics.obs_rms.mean[0]), 0.0, places=6)

    def test_sidecar_without_statistics_is_ignored(self) -> None:
        from training.evaluation import load_normalizer

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "empty_vecnormalize.pkl"
            path.write_bytes(b"")
            with self.assertRaises(Exception):
                load_normalizer(path)

    def test_norm_obs_false_keeps_the_raw_perception(self) -> None:
        from training.evaluation import normalize

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "raw_vecnormalize.pkl"
            payload = SimpleNamespace(obs_rms=SimpleNamespace(mean=[1.0], var=[1.0]),
                                      norm_obs=False, clip_obs=10.0)
            with path.open("wb") as handle:
                pickle.dump(payload, handle)
            from training.evaluation import load_normalizer

            statistics = load_normalizer(path)
        observation = np.asarray([0.25], dtype=np.float32)
        self.assertTrue(np.allclose(normalize(observation, statistics), observation))

    def test_recursive_attribute_lookup_cannot_break_normalisation(self) -> None:
        from training.evaluation import normalize

        normalizer = _StrippedNormalizer([0.0], [1.0], True)
        normalizer.__dict__.pop("norm_obs")  # missing -> __getattr__ would recurse
        observation = np.asarray([0.5], dtype=np.float32)
        self.assertTrue(np.allclose(normalize(observation, normalizer), observation))


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
