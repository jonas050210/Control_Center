"""Static analysis for the Godot/GDScript half of SandboxAI.

Godot itself cannot always be executed in every CI/dev sandbox (no engine
binary, no GPU, restricted network). The GDScript sources are nevertheless
the authoritative simulation, so this module provides the strongest
verification that is possible *without* an engine:

1. **Syntax** — every ``.gd`` file is parsed with ``gdtoolkit``'s real
   GDScript grammar (the same parser ``gdformat``/``gdlint`` use).
2. **Dependency resolution** — every ``preload("res://...")`` /
   ``load("res://...")`` literal must point at a file that exists.
3. **Symbol resolution** — for any ``Alias.MEMBER`` access where ``Alias``
   is a ``const Alias = preload(...)`` or a project ``class_name``, the
   member must actually be declared by that script (following ``extends``
   to project-local base classes).
4. **Contract mirroring** — helpers used by ``python/tests/test_contract.py``
   to compare the Godot observation/action contract against
   ``python/sandboxai/contract.py``.

It is deliberately conservative: anything it cannot resolve with certainty
is skipped rather than reported, so a finding is a real finding. This is
static analysis, NOT a substitute for running the engine test-suite
(``godot --headless --path . --script res://tests/run_tests.gd``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Iterable

RES_PREFIX = "res://"

_PRELOAD_CONST_RE = re.compile(
    r"^\s*const\s+([A-Za-z_]\w*)\s*(?::\s*[\w\.]+\s*)?=\s*(?:preload|load)\(\s*\"(res://[^\"]+)\"\s*\)"
)
_RES_PATH_RE = re.compile(r"\"(res://[^\"]+)\"")
_CLASS_NAME_RE = re.compile(r"^\s*class_name\s+([A-Za-z_]\w*)")
_EXTENDS_RE = re.compile(r"^\s*extends\s+([A-Za-z_][\w\.]*)")
_CONST_RE = re.compile(r"^\s*const\s+([A-Za-z_]\w*)")
_VAR_RE = re.compile(r"^\s*(?:@export[^\s]*\s+|@onready\s+|static\s+)*var\s+([A-Za-z_]\w*)")
_FUNC_RE = re.compile(r"^\s*(?:static\s+)?func\s+([A-Za-z_]\w*)\s*\(")
_SIGNAL_RE = re.compile(r"^\s*signal\s+([A-Za-z_]\w*)")
_ENUM_RE = re.compile(r"^\s*enum\s+([A-Za-z_]\w*)?\s*\{")
_INNER_CLASS_RE = re.compile(r"^\s*class\s+([A-Za-z_]\w*)")
_MEMBER_ACCESS_RE = re.compile(r"\b([A-Z][A-Za-z0-9_]*)\.([A-Za-z_]\w*)")

## Godot built-in globals whose members we intentionally never check.
BUILTIN_TYPES = frozenset(
    {
        "Vector2", "Vector2i", "Vector3", "Vector3i", "Vector4", "Color", "Rect2", "Rect2i",
        "Transform2D", "Transform3D", "Basis", "Quaternion", "Plane", "AABB", "Projection",
        "String", "StringName", "NodePath", "RID", "Callable", "Signal", "Dictionary", "Array",
        "PackedByteArray", "PackedInt32Array", "PackedInt64Array", "PackedFloat32Array",
        "PackedFloat64Array", "PackedStringArray", "PackedVector2Array", "PackedVector3Array",
        "PackedColorArray", "JSON", "OS", "Engine", "Input", "InputEvent", "InputEventKey",
        "InputEventMouseButton", "InputEventMouseMotion", "Time", "ProjectSettings", "ResourceLoader",
        "DisplayServer", "ThemeDB", "Control", "Node", "Node3D", "Node2D", "CanvasItem", "Label",
        "Button", "CheckButton", "CheckBox", "HSlider", "VSlider", "SpinBox", "OptionButton",
        "LineEdit", "TextEdit", "RichTextLabel", "ItemList", "Tree", "PanelContainer", "VBoxContainer",
        "HBoxContainer", "GridContainer", "MarginContainer", "ScrollContainer", "TabContainer",
        "SplitContainer", "HSplitContainer", "VSplitContainer", "Camera3D", "MeshInstance3D",
        "BoxMesh", "SphereMesh", "CapsuleMesh", "CylinderMesh", "PlaneMesh", "QuadMesh",
        "StandardMaterial3D", "BaseMaterial3D", "ORMMaterial3D", "Material", "Mesh", "ArrayMesh",
        "ImmediateMesh", "SurfaceTool", "DirectionalLight3D", "OmniLight3D", "SpotLight3D",
        "WorldEnvironment", "Environment", "Sky", "Texture2D", "Image", "ImageTexture", "Font",
        "FontFile", "Theme", "StyleBox", "StyleBoxFlat", "StyleBoxEmpty", "RandomNumberGenerator",
        "SceneTree", "Window", "Viewport", "SubViewport", "SubViewportContainer", "Timer", "Tween",
        "FileAccess", "DirAccess", "Resource", "RefCounted", "Object", "Script", "GDScript",
        "PackedScene", "Performance", "RenderingServer", "PhysicsServer3D", "Geometry3D", "Geometry2D",
        "TranslationServer", "AudioServer", "EditorInterface", "ClassDB", "Marshalls", "Shader",
        "ShaderMaterial", "Label3D", "Sprite3D", "Skeleton3D", "AnimationPlayer", "Curve", "Gradient",
        "MultiMesh", "MultiMeshInstance3D", "TextServer", "Expression", "SceneState", "ConfigFile",
        "SystemFont", "CanvasLayer", "ColorRect", "TextureRect", "NinePatchRect", "Separator",
        "HSeparator", "VSeparator", "ProgressBar", "TextureProgressBar", "AcceptDialog",
        "ConfirmationDialog", "FileDialog", "PopupMenu", "MenuButton", "LinkButton", "TextureButton",
    }
)

## Members every Object/Node exposes; skipped for project classes too because
## a project class may inherit them through a non-project base.
UNIVERSAL_MEMBERS = frozenset(
    {
        "new", "duplicate", "get", "set", "call", "call_deferred", "has_method", "get_script",
        "free", "queue_free", "is_instance_valid", "connect", "disconnect", "emit", "emit_signal",
        "name", "get_parent", "add_child", "remove_child", "get_children", "get_node",
        "get_node_or_null", "set_script", "to_string", "get_class", "is_class", "resource_path",
        "instantiate", "can_instantiate", "get_instance_id", "notification", "set_process",
        "set_physics_process", "set_process_unhandled_input", "set_process_input", "propagate_call",
        "get_property_list", "get_method_list", "has_signal", "get_signal_list", "reference",
        "unreference", "get_reference_count", "set_meta", "get_meta", "has_meta",
    }
)


@dataclass
class ScriptInfo:
    """Declared surface of one GDScript file."""

    path: Path
    res_path: str
    class_name_: str | None = None
    extends: str | None = None
    members: set[str] = field(default_factory=set)
    functions: dict[str, tuple[int, int]] = field(default_factory=dict)
    preloads: dict[str, str] = field(default_factory=dict)
    res_references: set[str] = field(default_factory=set)
    lines: list[str] = field(default_factory=list)


@dataclass
class Finding:
    """One static-analysis problem."""

    path: str
    line: int
    kind: str
    message: str

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return f"{self.path}:{self.line}: [{self.kind}] {self.message}"


def project_root(start: Path | None = None) -> Path:
    """Repository root (the directory containing ``project.godot``)."""
    here = (start or Path(__file__)).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "project.godot").is_file():
            return candidate
    raise RuntimeError("could not locate project.godot above %s" % here)


def iter_gd_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for folder in ("scripts", "tests"):
        base = root / folder
        if base.is_dir():
            files.extend(sorted(base.rglob("*.gd")))
    return files


def _strip_strings_and_comments(line: str) -> str:
    """Removes string literals and trailing comments from one source line."""
    out: list[str] = []
    quote: str | None = None
    index = 0
    while index < len(line):
        char = line[index]
        if quote is not None:
            if char == "\\":
                index += 2
                continue
            if char == quote:
                quote = None
            index += 1
            continue
        if char in "\"'":
            quote = char
            # Placeholder token: keeps argument counts correct while hiding
            # string contents from the identifier/member regexes.
            out.append("0")
            index += 1
            continue
        if char == "#":
            break
        out.append(char)
        index += 1
    return "".join(out)


def _parse_enum_values(lines: list[str], start_index: int) -> set[str]:
    """Collects the identifiers declared by an ``enum {...}`` block."""
    depth = 0
    values: set[str] = set()
    for line in lines[start_index:]:
        cleaned = _strip_strings_and_comments(line)
        depth += cleaned.count("{") - cleaned.count("}")
        body = cleaned
        if "{" in body:
            body = body.split("{", 1)[1]
        if "}" in body:
            body = body.split("}", 1)[0]
        for chunk in body.split(","):
            match = re.match(r"\s*([A-Za-z_]\w*)", chunk)
            if match:
                values.add(match.group(1))
        if depth <= 0:
            break
    return values


def _function_arity(signature: str) -> tuple[int, int]:
    """Returns (required, maximum) parameter counts for a ``func`` signature."""
    inside = signature.split("(", 1)[1]
    depth = 1
    collected: list[str] = []
    for char in inside:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth == 0:
                break
        collected.append(char)
    params = _split_top_level("".join(collected))
    required = 0
    maximum = 0
    for param in params:
        if not param.strip():
            continue
        maximum += 1
        if "=" not in param:
            required += 1
    return required, maximum


def _split_top_level(text: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    quote: str | None = None
    for char in text:
        if quote is not None:
            current.append(char)
            if char == quote:
                quote = None
            continue
        if char in "\"'":
            quote = char
            current.append(char)
            continue
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(current))
            current = []
            continue
        current.append(char)
    if current:
        parts.append("".join(current))
    return parts


def parse_script(path: Path, root: Path) -> ScriptInfo:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    info = ScriptInfo(
        path=path,
        res_path=RES_PREFIX + path.relative_to(root).as_posix(),
        lines=lines,
    )
    for index, raw in enumerate(lines):
        stripped = raw.rstrip()
        for res_match in _RES_PATH_RE.finditer(stripped):
            candidate = res_match.group(1)
            # Format templates such as "res://tests/%s" are runtime-built.
            if "%" not in candidate:
                info.res_references.add(candidate)
        cleaned = _strip_strings_and_comments(raw)

        preload_match = _PRELOAD_CONST_RE.match(stripped)
        if preload_match:
            info.preloads[preload_match.group(1)] = preload_match.group(2)

        class_match = _CLASS_NAME_RE.match(cleaned)
        if class_match:
            info.class_name_ = class_match.group(1)
            continue
        extends_match = _EXTENDS_RE.match(cleaned)
        if extends_match and info.extends is None:
            info.extends = extends_match.group(1)
            continue
        enum_match = _ENUM_RE.match(cleaned)
        if enum_match:
            if enum_match.group(1):
                info.members.add(enum_match.group(1))
            else:
                info.members.update(_parse_enum_values(lines, index))
            continue
        func_match = _FUNC_RE.match(cleaned)
        if func_match:
            name = func_match.group(1)
            info.members.add(name)
            # A GDScript signature may wrap across several lines; accumulate
            # until the parameter parentheses balance before counting.
            signature = cleaned
            depth = signature.count("(") - signature.count(")")
            cursor = index
            while depth > 0 and cursor + 1 < len(lines):
                cursor += 1
                extra = _strip_strings_and_comments(lines[cursor])
                signature += " " + extra.strip()
                depth += extra.count("(") - extra.count(")")
            try:
                info.functions[name] = _function_arity(signature)
            except (IndexError, ValueError):  # pragma: no cover - defensive
                pass
            continue
        for pattern in (_CONST_RE, _VAR_RE, _SIGNAL_RE, _INNER_CLASS_RE):
            member_match = pattern.match(cleaned)
            if member_match:
                info.members.add(member_match.group(1))
                break
    return info


class ProjectIndex:
    """All project scripts, indexed by res:// path and by ``class_name``."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.by_res: dict[str, ScriptInfo] = {}
        self.by_class: dict[str, ScriptInfo] = {}
        for path in iter_gd_files(root):
            info = parse_script(path, root)
            self.by_res[info.res_path] = info
            if info.class_name_:
                self.by_class[info.class_name_] = info

    def resolve(self, name: str, origin: ScriptInfo) -> ScriptInfo | None:
        res_path = origin.preloads.get(name)
        if res_path is not None:
            return self.by_res.get(res_path)
        return self.by_class.get(name)

    def all_members(self, info: ScriptInfo, _seen: set[str] | None = None) -> set[str] | None:
        """Members of ``info`` plus its project-local base classes.

        Returns ``None`` when the inheritance chain leaves the project (the
        base is an engine class), meaning membership cannot be decided.
        """
        seen = _seen or set()
        if info.res_path in seen:
            return set(info.members)
        seen.add(info.res_path)
        members = set(info.members)
        base = info.extends
        if not base or base in ("RefCounted", "Object"):
            return members
        if base in BUILTIN_TYPES or "." in base:
            return None
        base_info = self.by_class.get(base) or self.by_res.get(base)
        if base_info is None:
            base_res = info.preloads.get(base)
            base_info = self.by_res.get(base_res) if base_res else None
        if base_info is None:
            return None
        parent_members = self.all_members(base_info, seen)
        if parent_members is None:
            return None
        return members | parent_members


def check_resource_paths(index: ProjectIndex) -> list[Finding]:
    findings: list[Finding] = []
    for info in index.by_res.values():
        for res_path in sorted(info.res_references):
            relative = res_path[len(RES_PREFIX):]
            if not (index.root / relative).exists():
                line = next(
                    (i + 1 for i, text in enumerate(info.lines) if res_path in text), 1
                )
                findings.append(
                    Finding(
                        info.res_path,
                        line,
                        "missing-resource",
                        f"referenced resource does not exist: {res_path}",
                    )
                )
    return findings


def check_symbols(index: ProjectIndex) -> list[Finding]:
    findings: list[Finding] = []
    for info in index.by_res.values():
        for line_number, raw in enumerate(info.lines, start=1):
            cleaned = _strip_strings_and_comments(raw)
            if not cleaned.strip() or cleaned.lstrip().startswith("#"):
                continue
            for match in _MEMBER_ACCESS_RE.finditer(cleaned):
                alias, member = match.group(1), match.group(2)
                if alias in BUILTIN_TYPES or member in UNIVERSAL_MEMBERS:
                    continue
                # Only alias tokens that are unambiguously a script reference.
                if alias not in info.preloads and alias not in index.by_class:
                    continue
                # Skip locals that shadow a class name (`var Foo = ...`).
                target = index.resolve(alias, info)
                if target is None:
                    continue
                members = index.all_members(target)
                if members is None:
                    continue
                if member not in members:
                    findings.append(
                        Finding(
                            info.res_path,
                            line_number,
                            "unknown-member",
                            f"{alias}.{member} is not declared by {target.res_path}",
                        )
                    )
    return findings


def check_call_arity(index: ProjectIndex) -> list[Finding]:
    findings: list[Finding] = []
    call_re = re.compile(r"\b([A-Z][A-Za-z0-9_]*)\.([a-z_]\w*)\s*\(")
    for info in index.by_res.values():
        source = "\n".join(_strip_strings_and_comments(line) for line in info.lines)
        for match in call_re.finditer(source):
            alias, method = match.group(1), match.group(2)
            if alias in BUILTIN_TYPES or method in UNIVERSAL_MEMBERS:
                continue
            if alias not in info.preloads and alias not in index.by_class:
                continue
            target = index.resolve(alias, info)
            if target is None or method not in target.functions:
                continue
            args_text, complete = _extract_call_args(source, match.end())
            if not complete:
                continue
            args = [part for part in _split_top_level(args_text) if part.strip()]
            required, maximum = target.functions[method]
            if len(args) < required or len(args) > maximum:
                line_number = source[: match.start()].count("\n") + 1
                findings.append(
                    Finding(
                        info.res_path,
                        line_number,
                        "call-arity",
                        (
                            f"{alias}.{method}() called with {len(args)} argument(s); "
                            f"{target.res_path} declares {required}..{maximum}"
                        ),
                    )
                )
    return findings


def _extract_call_args(source: str, start: int) -> tuple[str, bool]:
    depth = 1
    quote: str | None = None
    out: list[str] = []
    for char in source[start:]:
        if quote is not None:
            out.append(char)
            if char == quote:
                quote = None
            continue
        if char in "\"'":
            quote = char
            out.append(char)
            continue
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth == 0:
                return "".join(out), True
        out.append(char)
    return "".join(out), False


def parse_all(root: Path | None = None) -> list[Finding]:
    """Runs the gdtoolkit grammar over every project script."""
    root = root or project_root()
    findings: list[Finding] = []
    try:
        from gdtoolkit.parser import parser as gd_parser  # type: ignore
    except ImportError:
        return [
            Finding(
                "<gdtoolkit>",
                0,
                "parser-unavailable",
                "gdtoolkit is not installed; GDScript syntax was NOT verified",
            )
        ]
    for path in iter_gd_files(root):
        try:
            gd_parser.parse(path.read_text(encoding="utf-8"), gather_metadata=False)
        except Exception as exc:  # noqa: BLE001 - lark raises many types
            findings.append(
                Finding(
                    RES_PREFIX + path.relative_to(root).as_posix(),
                    0,
                    "syntax",
                    str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__,
                )
            )
    return findings


def analyze(root: Path | None = None) -> list[Finding]:
    """Full static analysis: syntax + resources + symbols + call arity."""
    root = root or project_root()
    index = ProjectIndex(root)
    findings = parse_all(root)
    findings.extend(check_resource_paths(index))
    findings.extend(check_symbols(index))
    findings.extend(check_call_arity(index))
    return findings


def format_findings(findings: Iterable[Finding]) -> str:
    return "\n".join(str(finding) for finding in findings)


if __name__ == "__main__":  # pragma: no cover - CLI helper
    import sys

    results = analyze()
    if results:
        print(format_findings(results))
        sys.exit(1)
    print("GDScript static analysis: no findings")
