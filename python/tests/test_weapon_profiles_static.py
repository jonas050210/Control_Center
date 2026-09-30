from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


class WeaponProfileStaticTests(unittest.TestCase):
    def test_weapon_state_declares_ttk_style_profiles(self):
        source = (REPO_ROOT / "scripts/weapon/weapon_state.gd").read_text(encoding="utf-8")
        profiles = set(re.findall(r'const PROFILE_\w+: String = "([a-z_]+)"', source))
        self.assertTrue({"rifle", "shotgun", "pistol", "smg"}.issubset(profiles))
        self.assertIn("func projectile_directions", source)
        self.assertIn("func projectile_damage", source)

    def test_weapon_drill_scenarios_are_opt_in_not_random_curriculum(self):
        source = (REPO_ROOT / "scripts/scenario/scenario_library.gd").read_text(encoding="utf-8")
        for scenario_id, profile_id in (
            ("rifle_lane_drill", "rifle"),
            ("shotgun_breach_drill", "shotgun"),
            ("sidearm_finish_drill", "pistol"),
        ):
            pattern = (
                rf'"id": "{scenario_id}"[\s\S]*?'
                rf'"weapon_profile": "{profile_id}"[\s\S]*?'
                r'"training_pool": false'
            )
            self.assertRegex(source, pattern, f"{scenario_id} must be an explicit drill")
        self.assertIn("static func training_ids()", source)
        reset_source = (REPO_ROOT / "scripts/env/environment_reset.gd").read_text(encoding="utf-8")
        self.assertIn("ScenarioLibrary.training_ids()", reset_source)

    def test_episode_plans_can_stage_weapon_profiles(self):
        manager_source = (REPO_ROOT / "scripts/core/simulation_manager.gd").read_text(
            encoding="utf-8"
        )
        reset_source = (REPO_ROOT / "scripts/env/environment_reset.gd").read_text(encoding="utf-8")
        env_source = (REPO_ROOT / "scripts/env/environment_core.gd").read_text(encoding="utf-8")
        self.assertIn("set_weapon_profile", env_source)
        self.assertIn('plan.get("weapon_profile"', manager_source)
        self.assertIn("unknown weapon_profile", reset_source)


if __name__ == "__main__":
    unittest.main()
