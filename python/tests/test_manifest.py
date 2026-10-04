"""Run-manifest provenance: code, host, simulator and selection rule."""

import json
import os
import stat
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from sandboxai.config import TrainingConfig
from sandboxai.manifest import (
    MANIFEST_FORMAT,
    build_manifest,
    code_provenance,
    contract_fingerprint,
    godot_snapshot,
    host_snapshot,
    package_versions,
    write_manifest,
)


class ProvenanceSectionTests(unittest.TestCase):
    def test_contract_fingerprint_is_the_frozen_contract(self):
        fingerprint = contract_fingerprint()
        self.assertEqual(fingerprint["version"], 5)
        self.assertEqual(fingerprint["observation_dim"], 126)
        self.assertEqual(fingerprint["action_nvec"], [3, 3, 3, 3, 2, 2])

    def test_package_versions_report_every_tracked_dependency(self):
        versions = package_versions()
        self.assertIn("sandboxai", versions)
        for package in ("torch", "stable_baselines3", "gymnasium", "numpy"):
            self.assertIn(package, versions)

    def test_code_provenance_reports_commit_branch_and_dirtiness(self):
        code = code_provenance()
        self.assertIn("commit", code)
        self.assertIn("branch", code)
        # Never "clean by omission": unknown is None, not False.
        self.assertIn(code["dirty"], (True, False, None))

    def test_code_provenance_survives_a_missing_git(self):
        with mock.patch("sandboxai.manifest.subprocess.run", side_effect=OSError("no git")):
            code = code_provenance()
        self.assertEqual(code["commit"], "")
        self.assertIsNone(code["dirty"], "an unknown tree state must not be reported as clean")

    def test_host_snapshot_records_the_execution_environment(self):
        host = host_snapshot()
        for key in ("python", "system", "machine", "logical_cpus", "wsl"):
            self.assertIn(key, host)
        self.assertGreaterEqual(host["logical_cpus"], 1)
        self.assertIsInstance(host["wsl"], bool)

    def test_godot_snapshot_degrades_when_the_engine_is_absent(self):
        snapshot = godot_snapshot(
            TrainingConfig(godot_executable="definitely-not-installed-godot"), probe=True
        )
        self.assertEqual(snapshot["configured"], "definitely-not-installed-godot")
        self.assertIsNone(snapshot["version"])

    @unittest.skipUnless(
        os.name == "posix", "fake bridge executable requires POSIX shebang support"
    )
    def test_godot_snapshot_reports_the_probed_version_when_available(self):
        # Regression test: godot_snapshot used to import a class named
        # GodotRuntimeValidator that does not exist (the real class is
        # RuntimeValidator), so a broad `except Exception` silently
        # swallowed the ImportError and every manifest's godot.version was
        # None even when Godot ran successfully. A fake executable that
        # answers `--version --headless` like the real engine catches that
        # class of bug instead of only exercising the "absent" path.
        with TemporaryDirectory() as tmp:
            script = Path(tmp) / "fake_godot.py"
            script.write_text(
                "import sys\nif '--version' in sys.argv:\n    print('4.7.2.stable.official')\n",
                encoding="utf-8",
            )
            wrapper = Path(tmp) / "fake_godot"
            wrapper.write_text(
                f"#!/bin/sh\nexec '{sys.executable}' '{script}' \"$@\"\n", encoding="utf-8"
            )
            wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            snapshot = godot_snapshot(TrainingConfig(godot_executable=str(wrapper)), probe=True)
        self.assertEqual(snapshot["version"], "4.7.2.stable.official")

    def test_godot_snapshot_can_skip_the_subprocess_probe(self):
        with mock.patch("sandboxai.manifest.subprocess.run") as run:
            snapshot = godot_snapshot(TrainingConfig(), probe=False)
        run.assert_not_called()
        self.assertIsNone(snapshot["version"])


class ManifestDocumentTests(unittest.TestCase):
    def _manifest(self, **overrides):
        config = TrainingConfig(
            environment_count=2,
            env_workers=2,
            run_id="manifest_test",
            experiment_id="exp",
            **overrides,
        ).validate()
        return build_manifest(config, Path("."), "cpu", None, probe_godot=False), config

    def test_manifest_has_the_current_format_and_core_sections(self):
        manifest, config = self._manifest()
        self.assertEqual(manifest["format"], MANIFEST_FORMAT)
        for section in (
            "contract",
            "godot",
            "host",
            "curriculum",
            "evaluation",
            "checkpoint_selection",
            "hyperparameters",
            "parallelism",
            "code",
            "package_versions",
        ):
            self.assertIn(section, manifest, f"missing manifest section: {section}")
        self.assertEqual(manifest["seed"], config.seed)
        self.assertEqual(manifest["run_id"], "manifest_test")
        self.assertEqual(manifest["algorithm"], "PPO")
        self.assertEqual(manifest["status"], "running")
        self.assertEqual(manifest["checkpoints"], [])
        self.assertIsNone(manifest["dataset"])

    def test_manifest_records_the_parallelism_actually_used(self):
        manifest, _ = self._manifest()
        parallelism = manifest["parallelism"]
        self.assertEqual(parallelism["environment_count"], 2)
        self.assertEqual(parallelism["env_workers"], 2)
        self.assertEqual(parallelism["resolved_env_workers"], 2)

    def test_manifest_records_the_checkpoint_selection_rule(self):
        manifest, _ = self._manifest(
            checkpoint_selection_metric="win_rate",
            checkpoint_selection_goal="max",
            checkpoint_selection_min_delta=0.01,
        )
        self.assertEqual(
            manifest["checkpoint_selection"],
            {"metric": "win_rate", "goal": "max", "min_delta": 0.01},
        )

    def test_legacy_code_revision_field_is_preserved(self):
        manifest, _ = self._manifest()
        self.assertEqual(manifest["code_revision"], manifest["code"]["commit"])

    def test_manifest_is_json_serialisable_and_roundtrips(self):
        manifest, _ = self._manifest()
        with TemporaryDirectory() as tmp:
            path = write_manifest(Path(tmp), manifest)
            roundtrip = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(roundtrip["format"], MANIFEST_FORMAT)
        self.assertEqual(roundtrip["host"]["python"], manifest["host"]["python"])

    def test_manifest_never_raises_on_a_hostile_environment(self):
        # git gone, torch import broken, godot missing: a manifest must
        # still be written, because it is provenance, not a precondition.
        with mock.patch("sandboxai.manifest.subprocess.run", side_effect=OSError("boom")):
            with mock.patch.dict("sys.modules", {"torch": None}):
                manifest, _ = self._manifest()
        self.assertEqual(manifest["format"], MANIFEST_FORMAT)
        self.assertEqual(manifest["code_revision"], "")


if __name__ == "__main__":
    unittest.main()
