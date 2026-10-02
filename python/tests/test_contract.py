"""Consistency checks for the Observation/Action contract description.

These tests validate the Python-side contract module for the local Godot
bridge. They intentionally do not require a running Godot process; keeping
OBSERVATION_SPEC in sync with
scripts/core/observation.gd is a manual responsibility documented in
contract.py's module docstring. The GodotSourceDriftTests below narrow that
gap statically: they parse the GDScript sources and fail loudly when the
Godot-side constants/field layout no longer match the Python contract.
"""

import re
import unittest
from pathlib import Path

from sandboxai.contract import (
    ACTION_NVEC,
    ACTION_SPEC,
    OBJECT_KIND_NAMES,
    OBSERVATION_COUNT_NORMALIZER,
    OBSERVATION_FIELD_COUNT,
    OBSERVATION_HIGH,
    OBSERVATION_LOW,
    OBSERVATION_MAX_TRACKED_ENEMIES,
    OBSERVATION_SPEC,
    validate_observation_spec,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ObservationContractTests(unittest.TestCase):
    def test_field_indices_are_contiguous_and_cover_the_full_vector(self):
        validate_observation_spec()

    def test_observation_field_count_matches_godot_contract(self):
        # Mirrors Observation.FIELD_COUNT in scripts/core/observation.gd.
        self.assertEqual(OBSERVATION_FIELD_COUNT, 106)

    def test_observation_bounds_are_symmetric_and_normalized(self):
        self.assertEqual(OBSERVATION_LOW, -1.0)
        self.assertEqual(OBSERVATION_HIGH, 1.0)

    def test_tracked_enemy_budget_matches_primary_plus_two_extra(self):
        # primary (implicit) + secondary + tertiary == 3 tracked enemies.
        self.assertEqual(OBSERVATION_MAX_TRACKED_ENEMIES, 3)

    def test_field_names_are_unique(self):
        names = [field.name for field in OBSERVATION_SPEC]
        self.assertEqual(len(names), len(set(names)))


class ActionContractTests(unittest.TestCase):
    def test_action_nvec_matches_multidiscrete_shape(self):
        self.assertEqual(ACTION_NVEC, (3, 3, 3, 3, 2, 2))

    def test_action_field_indices_are_sequential(self):
        for expected_index, field in enumerate(ACTION_SPEC):
            self.assertEqual(field.index, expected_index)


class GodotSourceDriftTests(unittest.TestCase):
    """Static cross-language drift guards for the observation/action contract.

    The contract is implemented twice (GDScript + Python) and no build step
    generates one from the other. These tests parse the Godot sources and
    compare the declared constants and the to_array() field layout against
    the Python contract, so an accidental change on either side fails here
    instead of silently producing incompatible checkpoints.
    """

    def _godot_source(self, relative: str) -> str:
        path = PROJECT_ROOT / relative
        self.assertTrue(path.is_file(), f"missing Godot source: {path}")
        return path.read_text(encoding="utf-8")

    def test_godot_observation_field_count_matches_python_contract(self):
        source = self._godot_source("scripts/core/observation.gd")
        match = re.search(r"const FIELD_COUNT:\s*int\s*=\s*(\d+)", source)
        self.assertIsNotNone(match, "Observation.FIELD_COUNT declaration not found")
        self.assertEqual(
            int(match.group(1)),
            OBSERVATION_FIELD_COUNT,
            "Observation.FIELD_COUNT in scripts/core/observation.gd no longer matches "
            "OBSERVATION_FIELD_COUNT in python/sandboxai/contract.py",
        )

    def test_godot_to_array_assigns_every_contract_index_exactly_once(self):
        source = self._godot_source("scripts/core/observation.gd")
        start = source.find("func to_array()")
        end = source.find("\nfunc ", start + 1)
        self.assertGreater(start, -1, "to_array() not found in observation.gd")
        body = source[start : end if end != -1 else len(source)]
        indices = [int(value) for value in re.findall(r"arr\[(\d+)\]\s*=", body)]
        self.assertEqual(
            sorted(indices),
            list(range(OBSERVATION_FIELD_COUNT)),
            "to_array() must assign exactly indices 0..N-1 with no gaps or duplicates; "
            "update the field table in docs/OBSERVATION_ACTION_CONTRACT.md and "
            "python/sandboxai/contract.py together with scripts/core/observation.gd",
        )
        self.assertIn(
            "arr.resize(FIELD_COUNT)",
            body,
            "to_array() must pre-allocate the packed array to FIELD_COUNT",
        )

    def test_count_normalizer_matches_the_godot_constant(self):
        """Every count field is "count / N, clamped"; N must be one number.

        The Stats page converts a normalized count back for display, so a
        silent change to `Observation.COUNT_NORMALIZER` would make the GUI
        report a different number of visible objects than the vector encodes.
        """
        source = self._godot_source("scripts/core/observation.gd")
        match = re.search(r"const COUNT_NORMALIZER:\s*int\s*=\s*(\d+)", source)
        self.assertIsNotNone(match, "Observation.COUNT_NORMALIZER declaration not found")
        assert match is not None  # narrowing for type checkers
        self.assertEqual(int(match.group(1)), OBSERVATION_COUNT_NORMALIZER)

    def test_object_kind_names_match_the_godot_enum_order(self):
        """``object_k_kind_norm`` is only readable if the ordinals agree.

        The contract stores a normalized ordinal, so the *order* of
        ``Obstacle.Kind`` is a wire detail: inserting a kind in the middle
        would relabel every recorded replay without changing a single value.
        Parsing the enum here is what keeps the Python mirror honest.
        """
        source = self._godot_source("scripts/world/obstacle.gd")
        match = re.search(r"enum Kind \{(.*?)\}", source, re.DOTALL)
        self.assertIsNotNone(match, "Obstacle.Kind enum not found in obstacle.gd")
        assert match is not None  # narrowing for type checkers
        declared = re.findall(r"^\s*([A-Z_]+)\s*=\s*\d+", match.group(1), re.MULTILINE)
        self.assertEqual(
            tuple(name.lower() for name in declared),
            OBJECT_KIND_NAMES,
            "Obstacle.Kind in scripts/world/obstacle.gd no longer matches "
            "contract.OBJECT_KIND_NAMES; the normalized kind in every recorded "
            "replay depends on this order",
        )

    def test_only_one_place_computes_a_bearing_sign(self):
        """One bearing convention, one implementation - enforced, not hoped.

        The vector's nine bearing fields were split between two sign
        conventions: `Observation` measured a yaw difference (positive = the
        agent's right, matching `look_yaw_axis`) while the world, sound and
        memory queries took the sign of `forward.cross(direction).y`, which is
        the opposite in Godot's right-handed frame. A policy and a human
        reading the Stats page then saw a contact and the cover next to it on
        opposite sides. Every bearing now goes through
        `scripts/core/vector_math.gd`; this test refuses a second copy.
        """
        pattern = re.compile(r"\.cross\([^)]*\)\s*\.y")
        offenders = sorted(
            str(path.relative_to(PROJECT_ROOT))
            for path in (PROJECT_ROOT / "scripts").rglob("*.gd")
            if path.name != "vector_math.gd" and pattern.search(path.read_text(encoding="utf-8"))
        )
        self.assertEqual(
            offenders,
            [],
            "a bearing sign must come from VectorMath.signed_bearing_*\n"
            "taking the sign of cross().y in another file re-introduces the "
            "second convention",
        )

        # And the users must actually go through it, so the rule above cannot
        # be satisfied by deleting a sign calculation outright.
        for relative, helper in (
            ("scripts/core/observation.gd", "VectorMath.signed_bearing_deg"),
            ("scripts/perception/perception_system.gd", "VectorMath.signed_bearing_deg"),
            ("scripts/perception/sound_bus.gd", "VectorMath.signed_bearing_between"),
            ("scripts/world/arena_world.gd", "VectorMath.signed_bearing_between"),
        ):
            with self.subTest(source=relative):
                self.assertIn(helper, self._godot_source(relative))

    def test_only_one_place_computes_a_yaw_or_an_elevation(self):
        """The other two angle conventions live in `VectorMath` as well.

        Yaw (`rad_to_deg(atan2(x, -z))`) was spelled out in the agent, the
        enemies, the stub controller, the scenario/world generators and the
        environment reset; elevation (`atan2(y, horizontal)`) existed twice,
        in the perception system and in `Observation`. Either pair could have
        drifted apart the way the bearing sign did, silently. Both are now
        one function each, and this test refuses a new copy of the formula.
        """
        yaw = re.compile(r"atan2\(\s*[A-Za-z_][\w.]*\.x\s*,\s*-")
        elevation = re.compile(r"atan2\(\s*[A-Za-z_][\w.]*\.y\s*,")
        offenders = []
        for path in sorted((PROJECT_ROOT / "scripts").rglob("*.gd")):
            if path.name == "vector_math.gd":
                continue
            source = path.read_text(encoding="utf-8")
            if yaw.search(source) or elevation.search(source):
                offenders.append(str(path.relative_to(PROJECT_ROOT)))
        self.assertEqual(
            offenders,
            [],
            "yaw/elevation must come from VectorMath; a copy of the formula can "
            "drift away from the convention it is supposed to share",
        )

        for relative, helper in (
            ("scripts/agent/agent_state.gd", "VectorMath.yaw_deg_from_direction"),
            ("scripts/enemy/enemy_state.gd", "VectorMath.yaw_deg_from_direction"),
            ("scripts/input/ai_stub_controller.gd", "VectorMath.yaw_deg_from_direction"),
            ("scripts/scenario/scenario_library.gd", "VectorMath.yaw_deg_from_direction"),
            ("scripts/world/world_generator.gd", "VectorMath.yaw_deg_from_direction"),
            ("scripts/env/environment_reset.gd", "VectorMath.yaw_deg_from_direction"),
            ("scripts/perception/perception_system.gd", "VectorMath.elevation_deg"),
            ("scripts/core/observation.gd", "VectorMath.elevation_deg"),
        ):
            with self.subTest(source=relative):
                self.assertIn(helper, self._godot_source(relative))

    def test_godot_field_spec_matches_python_observation_spec(self):
        """Observation.FIELD_SPEC is the Godot-side label source of truth.

        The Control Center's Observation Inspector renders field names from
        it instead of keeping its own list, so it must stay identical to
        OBSERVATION_SPEC (names, indices and widths, in order).
        """
        source = self._godot_source("scripts/core/observation.gd")
        start = source.find("const FIELD_SPEC")
        self.assertGreater(start, -1, "Observation.FIELD_SPEC declaration not found")
        end = source.find("\n]", start)
        self.assertGreater(end, start, "Observation.FIELD_SPEC is not terminated")
        body = source[start:end]
        entries = re.findall(
            r'"index":\s*(\d+),\s*"width":\s*(\d+),\s*"name":\s*"([a-z_0-9.]+)"',
            re.sub(r"\s+", " ", body),
        )
        self.assertEqual(
            len(entries),
            len(OBSERVATION_SPEC),
            "FIELD_SPEC in scripts/core/observation.gd has a different number of entries "
            "than OBSERVATION_SPEC in python/sandboxai/contract.py",
        )
        parsed = [(int(index), int(width), name) for index, width, name in entries]
        expected = [(field.index, field.width, field.name) for field in OBSERVATION_SPEC]
        self.assertEqual(
            parsed,
            expected,
            "FIELD_SPEC and OBSERVATION_SPEC disagree on field order, index or width",
        )
        self.assertEqual(
            sum(width for _, width, _ in parsed),
            OBSERVATION_FIELD_COUNT,
            "FIELD_SPEC widths must sum to the contract field count",
        )

    def test_godot_field_name_helpers_are_derived_not_duplicated(self):
        """field_names() must be generated from FIELD_SPEC, not a second list."""
        source = self._godot_source("scripts/core/observation.gd")
        start = source.find("static func field_names()")
        self.assertGreater(start, -1, "Observation.field_names() not found")
        end = source.find("\nstatic func ", start + 1)
        body = source[start : end if end != -1 else len(source)]
        self.assertIn("FIELD_SPEC", body, "field_names() must iterate FIELD_SPEC")

    def test_godot_action_nvec_matches_python_contract(self):
        source = self._godot_source("scripts/core/action.gd")
        match = re.search(r"const MULTI_DISCRETE_NVECS:\s*Array\s*=\s*\[([0-9,\s]+)\]", source)
        self.assertIsNotNone(match, "Action.MULTI_DISCRETE_NVECS declaration not found")
        self.assertEqual(
            tuple(int(value) for value in match.group(1).split(",")),
            ACTION_NVEC,
            "Action.MULTI_DISCRETE_NVECS in scripts/core/action.gd no longer matches "
            "ACTION_NVEC in python/sandboxai/contract.py",
        )

    def test_godot_tracked_enemy_budget_matches_python_contract(self):
        source = self._godot_source("scripts/core/sandbox_config.gd")
        match = re.search(r"const OBSERVATION_MAX_TRACKED_ENEMIES:\s*int\s*=\s*(\d+)", source)
        self.assertIsNotNone(
            match, "SandboxConfig.OBSERVATION_MAX_TRACKED_ENEMIES declaration not found"
        )
        self.assertEqual(int(match.group(1)), OBSERVATION_MAX_TRACKED_ENEMIES)


if __name__ == "__main__":
    unittest.main()


class ObservationGroupTests(unittest.TestCase):
    """The semantic observation-group partition."""

    def test_groups_partition_every_field_exactly_once(self):
        from sandboxai.contract import OBSERVATION_GROUPS

        grouped = [name for names in OBSERVATION_GROUPS.values() for name in names]
        self.assertEqual(len(grouped), len(set(grouped)))
        self.assertEqual(set(grouped), {field.name for field in OBSERVATION_SPEC})

    def test_every_observation_group_is_present(self):
        from sandboxai.contract import OBSERVATION_GROUPS

        self.assertEqual(
            set(OBSERVATION_GROUPS),
            {
                "self_state",
                "movement",
                "combat",
                "targets",
                "perception",
                "memory",
                "sound",
                "world",
                "conditions",
                "contacts",
                "target",
                "exploration",
                "objects",
            },
        )
