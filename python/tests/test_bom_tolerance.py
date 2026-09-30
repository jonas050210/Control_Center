"""Files written on Windows carry a byte-order mark; reading must survive it.

Windows is a first-class target for this project, and PowerShell's
``Out-File -Encoding utf8`` prefixes a UTF-8 BOM. Python's ``utf-8``
codec hands that BOM to ``json.loads``, which fails with the singularly
unhelpful ``Expecting value: line 1 column 1``. ``utf-8-sig`` strips a
BOM when present and is byte-identical to ``utf-8`` when it is not, so
every *read* of a file a user might have produced uses it.

These tests pin the behaviour rather than the spelling: they write real
BOM-prefixed files and load them through the public entry points.
"""

from __future__ import annotations

import ast
import json
import re
import tempfile
import unittest
from pathlib import Path

from sandboxai.config import TrainingConfig
from sandboxai.run_inspection import read_json

PACKAGE = Path(__file__).resolve().parents[2] / "python" / "sandboxai"


class BomToleranceTests(unittest.TestCase):
    def test_training_config_loads_a_powershell_written_file(self) -> None:
        config = TrainingConfig()
        payload = json.dumps(config.to_dict(), indent=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(payload, encoding="utf-8-sig")  # PowerShell's utf8
            restored = TrainingConfig.load(path)
        self.assertEqual(restored.to_dict(), config.to_dict())

    def test_training_config_still_loads_a_plain_utf8_file(self) -> None:
        config = TrainingConfig()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
            self.assertEqual(TrainingConfig.load(path).to_dict(), config.to_dict())

    def test_read_json_accepts_a_bom(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.json"
            path.write_text(json.dumps({"reward": 1.5}), encoding="utf-8-sig")
            value, problem = read_json(path)
        self.assertIsNone(problem)
        self.assertEqual(value, {"reward": 1.5})

    def test_a_bom_does_not_leak_into_the_first_key(self) -> None:
        """utf-8 would keep the BOM as a character inside the first key."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.json"
            path.write_text(json.dumps({"alpha": 1}), encoding="utf-8-sig")
            value, _ = read_json(path)
        self.assertEqual(list(value), ["alpha"])
        self.assertFalse(any(key.startswith("\ufeff") for key in value))


class NoPlainUtf8JsonReadsRemainTests(unittest.TestCase):
    """A new ``json.loads(path.read_text(encoding="utf-8"))`` reintroduces the bug.

    This is a source-level check because the failure only appears with a
    file nobody has in the test suite: one produced by a Windows tool.
    Catching it at review time is cheaper than catching it in a bug
    report six months later.
    """

    PATTERN = re.compile(r'json\.loads\([^\n]*?read_text\(encoding="utf-8"\)')

    def test_no_module_reads_json_as_plain_utf8(self) -> None:
        offenders: list[str] = []
        for path in sorted(PACKAGE.glob("*.py")):
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if self.PATTERN.search(line):
                    offenders.append(f"{path.name}:{number}")
        self.assertEqual(
            offenders,
            [],
            "these read JSON as plain utf-8 and will reject a Windows BOM; "
            "use utf-8-sig for reads: " + ", ".join(offenders),
        )

    def test_writes_still_use_plain_utf8(self) -> None:
        """Only reads are lenient. Writing a BOM would be the opposite bug."""
        offenders: list[str] = []
        for path in sorted(PACKAGE.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = getattr(node.func, "attr", "")
                if name not in {"write_text", "open"}:
                    continue
                encodings = [
                    keyword.value.value
                    for keyword in node.keywords
                    if keyword.arg == "encoding" and isinstance(keyword.value, ast.Constant)
                ]
                mode = next(
                    (
                        argument.value
                        for argument in node.args
                        if isinstance(argument, ast.Constant) and argument.value in {"w", "a"}
                    ),
                    None,
                )
                is_write = name == "write_text" or mode in {"w", "a"}
                if is_write and "utf-8-sig" in encodings:
                    offenders.append(f"{path.name}:{node.lineno}")
        self.assertEqual(offenders, [], "these would write a BOM: " + ", ".join(offenders))


if __name__ == "__main__":  # pragma: no cover - manual entry point
    unittest.main()
