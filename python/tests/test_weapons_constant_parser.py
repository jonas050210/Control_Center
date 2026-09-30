"""The GDScript ``const`` expression evaluator behind the weapon tables.

This used to be a bare ``eval`` with a stripped ``__builtins__``, which
is not a security boundary: ``__builtins__`` can be reached back through
any object's class hierarchy. The replacement walks an ``ast`` whitelist,
so these tests pin both halves of the contract - the arithmetic that must
keep working, and the constructs that must be refused rather than run.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from sandboxai import weapons

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SANDBOX_CONFIG = REPOSITORY_ROOT / "scripts/core/sandbox_config.gd"


class SupportedExpressionTests(unittest.TestCase):
    """Every expression form the checked-in GDScript actually uses."""

    def _parse(self, body: str) -> dict[str, float]:
        return weapons._parse_scalar_constants(body)

    def test_plain_literals(self) -> None:
        parsed = self._parse("const A: float = 4.5\nconst B: int = 7\n")
        self.assertEqual(parsed, {"A": 4.5, "B": 7.0})

    def test_products_and_quotients_of_earlier_constants(self) -> None:
        source = "const A: float = 4.0\nconst B: float = A * 2.0\nconst C: float = B / 4.0\n"
        self.assertEqual(self._parse(source), {"A": 4.0, "B": 8.0, "C": 2.0})

    def test_sums_differences_and_unary_minus(self) -> None:
        source = "const A: float = 10.0\nconst B: float = A - 3.0\nconst C: float = -B + 1.0\n"
        self.assertEqual(self._parse(source), {"A": 10.0, "B": 7.0, "C": -6.0})

    def test_parentheses_and_powers(self) -> None:
        source = "const A: float = (2.0 + 3.0) * 2.0\nconst B: float = 2.0 ** 3.0\n"
        self.assertEqual(self._parse(source), {"A": 10.0, "B": 8.0})

    def test_bit_shifts(self) -> None:
        # scripts/rl/rl_server.gd: const STDIN_BUFFER_SIZE: int = 1 << 20
        self.assertEqual(self._parse("const A: int = 1 << 20\n"), {"A": 1048576.0})

    def test_a_trailing_comment_is_not_part_of_the_expression(self) -> None:
        self.assertEqual(self._parse("const A: float = 2.0  # metres\n"), {"A": 2.0})


class RefusedExpressionTests(unittest.TestCase):
    """Anything outside the whitelist is skipped, never guessed and never run."""

    def _parse(self, body: str) -> dict[str, float]:
        return weapons._parse_scalar_constants(body)

    def test_a_call_is_refused(self) -> None:
        self.assertEqual(self._parse("const A: float = maxf(1.0, 2.0)\n"), {})

    def test_an_attribute_lookup_is_refused(self) -> None:
        self.assertEqual(self._parse("const A: float = Vector3.ZERO\n"), {})

    def test_a_dunder_escape_is_refused(self) -> None:
        # The exact shape that makes `eval` with an emptied __builtins__
        # unsafe: reach the class hierarchy, walk back to a real builtin.
        source = "const A: float = ().__class__.__bases__\n"
        self.assertEqual(self._parse(source), {})

    def test_an_unknown_name_is_refused_rather_than_defaulted(self) -> None:
        self.assertEqual(self._parse("const A: float = SOMETHING_ELSE * 2.0\n"), {})

    def test_a_forward_reference_is_refused(self) -> None:
        # B is declared after A, so A cannot use it; silently treating it
        # as zero would produce a plausible but wrong weapon table.
        source = "const A: float = B * 2.0\nconst B: float = 3.0\n"
        self.assertEqual(self._parse(source), {"B": 3.0})

    def test_a_syntax_error_is_skipped(self) -> None:
        self.assertEqual(self._parse("const A: float = 2.0 +\n"), {})

    def test_division_by_zero_is_skipped(self) -> None:
        self.assertEqual(self._parse("const A: float = 1.0 / 0.0\n"), {})

    def test_a_boolean_is_not_a_number(self) -> None:
        # `True` is an int in Python; accepting it would turn a GDScript
        # `true` into the value 1.0.
        self.assertEqual(self._parse("const A: int = True\n"), {})

    def test_a_string_literal_is_refused(self) -> None:
        self.assertEqual(self._parse('const A: float = "3.0"\n'), {})

    def test_a_non_numeric_declaration_is_ignored_entirely(self) -> None:
        self.assertEqual(self._parse('const A: String = "rifle"\n'), {})


class RealConfigTests(unittest.TestCase):
    """The parser has to keep working on the file it exists for."""

    def test_sandbox_config_constants_are_parsed(self) -> None:
        source = SANDBOX_CONFIG.read_text(encoding="utf-8")
        constants = weapons._parse_scalar_constants(source)
        self.assertGreater(len(constants), 100)
        for name in ("AGENT_MAX_HEALTH", "ARENA_HALF_EXTENT", "ARENA_MAX_DISTANCE"):
            with self.subTest(constant=name):
                self.assertIn(name, constants)
                self.assertIsInstance(constants[name], float)

    def test_a_derived_constant_matches_its_gdscript_definition(self) -> None:
        constants = weapons._parse_scalar_constants(SANDBOX_CONFIG.read_text(encoding="utf-8"))
        # ARENA_MAX_DISTANCE is written as an expression over the arena
        # extents, so this asserts the arithmetic path, not a literal.
        self.assertAlmostEqual(constants["ARENA_MAX_DISTANCE"], 28.284272, places=5)


if __name__ == "__main__":  # pragma: no cover - manual entry point
    unittest.main()
