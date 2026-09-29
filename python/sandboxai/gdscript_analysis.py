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
4. **Undefined local calls** — every bare ``_helper()`` call must be
   declared by the calling script or one of its project-local base classes
   (Godot rejects undeclared bare calls at compile time and invalidates the
   whole script).
5. **Contract mirroring** — helpers used by ``python/tests/test_contract.py``
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
## Captures the optional `static` prefix separately so the analyzer can tell
## class-level (static) functions from instance functions.
_FUNC_DECL_RE = re.compile(r"^\s*(static\s+)?func\s+([A-Za-z_]\w*)\s*\(")
_FUNC_RE = re.compile(r"^\s*(?:static\s+)?func\s+([A-Za-z_]\w*)\s*\(")
_SIGNAL_RE = re.compile(r"^\s*signal\s+([A-Za-z_]\w*)")
_ENUM_RE = re.compile(r"^\s*enum\s+([A-Za-z_]\w*)?\s*\{")
_INNER_CLASS_RE = re.compile(r"^\s*class\s+([A-Za-z_]\w*)")
_MEMBER_ACCESS_RE = re.compile(r"\b([A-Z][A-Za-z0-9_]*)\.([A-Za-z_]\w*)")
## `var name: SomeType` at class scope (the `= value` part is optional).
_TYPED_MEMBER_RE = re.compile(
    r"^\s*(?:@export[^\s]*\s+|@onready\s+|static\s+)*var\s+([A-Za-z_]\w*)\s*:\s*"
    r"([A-Z][A-Za-z0-9_]*)"
)
## `Alias.Enum.MEMBER`
_NESTED_ENUM_RE = re.compile(r"\b([A-Z][A-Za-z0-9_]*)\.([A-Z][A-Za-z0-9_]*)\.([A-Za-z_]\w*)")

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

## Underscore methods the ENGINE may declare on a base class (virtual
## callbacks such as ``_ready``). Their absence from a user script is normal,
## so they can never be reported as undefined; engine-facing API is public by
## convention, which is what makes every other bare ``_name()`` call a
## project-private helper that must be declared somewhere in the chain.
ENGINE_VIRTUAL_METHODS = frozenset(
    {
        # Object
        "_init", "_notification", "_to_string", "_get", "_set", "_get_property_list",
        "_validate_property", "_property_can_revert", "_property_get_revert", "_script_exited",
        # Node / SceneTree main loop
        "_ready", "_enter_tree", "_exit_tree", "_process", "_physics_process", "_input",
        "_unhandled_input", "_unhandled_key_input", "_initialize", "_finalize",
        # CanvasItem / Control
        "_draw", "_gui_input", "_has_point", "_clips_input", "_make_custom_tooltip",
        "_get_minimum_size", "_theme_changed",
        # BaseButton / Range virtual signal handlers
        "_pressed", "_toggled",
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
    ## Names declared as `static func`: the ONLY functions that may be called
    ## through the script class itself (`Alias.name(...)`), besides the
    ## constructor and the Script-resource methods.
    static_functions: set[str] = field(default_factory=set)
    preloads: dict[str, str] = field(default_factory=dict)
    res_references: set[str] = field(default_factory=set)
    ## Named enums -> their member identifiers. `enum Level { A = 1 }`
    ## puts "Level" in `members` (so `Alias.Level` resolves) and the member
    ## names here (so `Alias.Level.A` can be validated too).
    enum_values: dict[str, set[str]] = field(default_factory=dict)
    ## Type-annotated members -> the annotation, e.g. `var episode:
    ## EpisodeState` gives {"episode": "EpisodeState"}. Lets a one-hop
    ## chain like `env.episode.to_metrics()` be resolved and checked.
    member_types: dict[str, str] = field(default_factory=dict)
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


def iter_gd_files(root: Path | str) -> list[Path]:
    # Accept a str as well as a Path: `analyze("/path")` used to raise
    # `TypeError: unsupported operand type(s) for /: 'str' and 'str'`
    # because the root was never coerced. The public entry points are
    # documented as taking a path, and a string is a path.
    root = Path(root)
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
                info.enum_values[enum_match.group(1)] = _parse_enum_values(lines, index)
            else:
                info.members.update(_parse_enum_values(lines, index))
            continue
        func_match = _FUNC_DECL_RE.match(cleaned)
        if func_match:
            name = func_match.group(2)
            info.members.add(name)
            if func_match.group(1):
                info.static_functions.add(name)
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
                typed = _TYPED_MEMBER_RE.match(cleaned)
                if typed:
                    info.member_types[typed.group(1)] = typed.group(2)
                break
    return info


class ProjectIndex:
    """All project scripts, indexed by res:// path and by ``class_name``."""

    def __init__(self, root: Path | str) -> None:
        root = Path(root)
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

    def all_static_functions(self, info: ScriptInfo, _seen: set[str] | None = None) -> set[str] | None:
        """Static functions of ``info`` plus its project-local base classes.

        Static functions are inherited like any other member, so a call
        through a derived script's alias may resolve into a base script.
        Returns ``None`` when the chain leaves the project.
        """
        seen = _seen or set()
        if info.res_path in seen:
            return set(info.static_functions)
        seen.add(info.res_path)
        statics = set(info.static_functions)
        base = info.extends
        if not base or base in ("RefCounted", "Object"):
            return statics
        if base in BUILTIN_TYPES or "." in base:
            return None
        base_info = self.by_class.get(base) or self.by_res.get(base)
        if base_info is None:
            base_res = info.preloads.get(base)
            base_info = self.by_res.get(base_res) if base_res else None
        if base_info is None:
            return None
        parent_statics = self.all_static_functions(base_info, seen)
        if parent_statics is None:
            return None
        return statics | parent_statics


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


def check_static_calls(index: ProjectIndex) -> list[Finding]:
    """Flags ``Alias.member(...)`` where ``Alias`` resolves to a project
    script and ``member`` exists there but is NOT a static function.

    Godot 4.x rejects this at COMPILE time ("Static function ... not found in
    base ..." / "Cannot call non-static function ... on the class ... directly"
    / "Member ... is not a function."), which invalidates the whole calling
    script. An invalid script still loads as a resource, so ``preload`` chains
    survive and the engine keeps running: ``Alias.new()`` silently returns
    ``null`` and every structure built from it comes back empty. That exact
    cascade turned the self-play reset observation into ``[]`` while every
    single-agent check still passed, which is why it deserves its own static
    check rather than trust in the (Godot-only) test suite.

    Members that do not exist at all are already reported by
    :func:`check_symbols` as ``unknown-member``; this check is only about the
    "exists but is an instance member" case, which symbol resolution alone
    cannot see.
    """
    findings: list[Finding] = []
    call_re = re.compile(r"\b([A-Z][A-Za-z0-9_]*)\.([a-z_]\w*)\s*\(")
    for info in index.by_res.values():
        source = "\n".join(_strip_strings_and_comments(line) for line in info.lines)
        for match in call_re.finditer(source):
            alias, member = match.group(1), match.group(2)
            if alias in BUILTIN_TYPES or member in UNIVERSAL_MEMBERS:
                continue
            if alias not in info.preloads and alias not in index.by_class:
                continue
            target = index.resolve(alias, info)
            if target is None:
                continue
            statics = index.all_static_functions(target)
            if statics is None or member in statics:
                continue
            members = index.all_members(target)
            if members is None or member not in members:
                # Missing members are check_symbols' business; inheritance may
                # also make this undecidable here.
                continue
            kind = "instance function" if member in target.functions else "instance member"
            line_number = source[: match.start()].count("\n") + 1
            findings.append(
                Finding(
                    info.res_path,
                    line_number,
                    "nonstatic-call",
                    (
                        f"{alias}.{member}() is an {kind} of {target.res_path}; "
                        "calling it through the script class is a Godot compile "
                        "error and invalidates this whole script"
                    ),
                )
            )
    return findings


## Bare call of a private-by-convention helper: `_name(` not preceded by a
## word character or a dot (so `obj._name(` / `Alias._name(` / `super._name(`
## are member calls on another object, not this check's business).
_LOCAL_CALL_RE = re.compile(r"(?<![\w.])(_[A-Za-z_]\w*)\s*\(")


def _declared_by_project_chain(index: ProjectIndex, info: ScriptInfo) -> set[str]:
    """Members declared by ``info`` itself plus every PROJECT-LOCAL base
    class in its ``extends`` chain.

    Unlike :meth:`ProjectIndex.all_members` this never returns "undecidable"
    when an ancestor extends an engine class: an engine base contributes
    nothing except the allow-listed virtuals anyway, so the decidable set is
    simply the script's own declarations plus its project ancestors.
    """
    members = set(info.members)
    seen = {info.res_path}
    current = info
    while True:
        base = current.extends
        if not base or base in ("RefCounted", "Object"):
            break
        if base in BUILTIN_TYPES or "." in base:
            break
        base_info = index.by_class.get(base) or index.by_res.get(base)
        if base_info is None:
            base_res = current.preloads.get(base)
            base_info = index.by_res.get(base_res) if base_res else None
        if base_info is None or base_info.res_path in seen:
            break
        seen.add(base_info.res_path)
        members |= base_info.members
        current = base_info
    return members


def check_local_method_calls(index: ProjectIndex) -> list[Finding]:
    """Flags bare calls to ``_private()`` helpers that neither the script nor
    its project-local base classes declare.

    Godot resolves bare calls at COMPILE time against the script and its base
    classes; an undeclared ``_helper()`` is a parse error ("Function
    ``_helper()`` not found in base self") that invalidates the whole script.
    An invalid script still loads as a (non-instantiable) resource, so
    ``preload`` chains survive and the failure only surfaces at runtime:
    ``Helper.new()`` aborts with "Nonexistent function 'new' in base
    'GDScript'". That exact shape once broke ``system_monitor.gd`` (two
    helpers referenced but never defined), made
    ``ControlCenterSession._init`` abort on ``ControlCenterSystemMonitor.new()``,
    and cascaded through ~37 Control Center tests while the Python suite
    stayed green — which is why this deserves a static check of its own.

    Scope is deliberately narrow so a finding is always real:

    * only underscore-prefixed names (engine API is public by convention, so
      a bare ``_name()`` is a project helper or an engine virtual);
    * engine virtuals (``_ready`` etc., see :data:`ENGINE_VIRTUAL_METHODS`)
      are allow-listed — the engine declares them, not the script;
    * names declared by the script OR any project-local base class are fine
      (:func:`_declared_by_project_chain` follows the chain; ``var``/``const``
      declarations count too, because a Callable may be stored in one);
    * ``obj._name()`` / ``Alias._name()`` / ``super._name()`` (preceded by a
      dot) are member calls on another object — covered elsewhere or not at
      all — and are skipped here.

    When the inheritance chain leaves the project (``extends Node`` etc.) the
    engine base cannot be indexed, but it cannot declare project-private
    helpers either, so anything not allow-listed above is a genuine finding.
    """
    findings: list[Finding] = []
    for info in index.by_res.values():
        members = _declared_by_project_chain(index, info)
        for line_number, raw in enumerate(info.lines, start=1):
            cleaned = _strip_strings_and_comments(raw)
            if not cleaned.strip():
                continue
            for match in _LOCAL_CALL_RE.finditer(cleaned):
                name = match.group(1)
                if name in ENGINE_VIRTUAL_METHODS:
                    continue
                if members is not None and name in members:
                    continue
                findings.append(
                    Finding(
                        info.res_path,
                        line_number,
                        "unknown-local-call",
                        (
                            f"{name}() is called but declared neither here nor in a "
                            "project base class (a Godot compile error that "
                            "invalidates this whole script)"
                        ),
                    )
                )
    return findings


## `var name := Alias.new(...)` / `var name: Alias = ...` — a local whose
## type is pinned to a project script, and is therefore checkable.
_TYPED_LOCAL_RE = re.compile(
    r"^\s*var\s+([a-z_]\w*)\s*(?::\s*([A-Z][A-Za-z0-9_]*)\s*=|:=\s*([A-Z][A-Za-z0-9_]*)\.new\s*\()"
)
## A call on one of those locals. The lookbehind stops it matching the
## tail of a longer chain (`env.episode.to_metrics(`), which is handled
## separately by `_LOCAL_CHAIN_CALL_RE`.
_LOCAL_MEMBER_CALL_RE = re.compile(r"(?<![.\w])([a-z_]\w*)\.([a-z_]\w*)\s*\(")
## `local.property.method(` — one hop through a type-annotated member.
_LOCAL_CHAIN_CALL_RE = re.compile(r"(?<![.\w])([a-z_]\w*)\.([a-z_]\w*)\.([a-z_]\w*)\s*\(")
## Any other assignment to the local invalidates our type knowledge.
_LOCAL_REBIND_RE = re.compile(r"^\s*([a-z_]\w*)\s*=\s*(?!=)")


def _arity_finding(
    info: ScriptInfo,
    target: ScriptInfo,
    label: str,
    method: str,
    source: str,
    call_end: int,
    line_number: int,
) -> Finding | None:
    """Argument-count check for a resolved call, or None when it is fine."""
    if method not in target.functions:
        return None
    args_text, complete = _extract_call_args(source, call_end)
    if not complete:
        return None
    args = [part for part in _split_top_level(args_text) if part.strip()]
    required, maximum = target.functions[method]
    if required <= len(args) <= maximum:
        return None
    return Finding(
        info.res_path,
        line_number,
        "call-arity",
        (
            f"{label}.{method}() called with {len(args)} argument(s); "
            f"{target.res_path} declares {required}..{maximum}"
        ),
    )


def check_typed_local_calls(index: ProjectIndex) -> list[Finding]:
    """Checks method calls on locals whose type is a known project script.

    ``check_symbols`` only sees ``Alias.member``, i.e. accesses through a
    ``class_name`` or a ``preload`` constant. The overwhelmingly common
    shape in this repository is different::

        var env := EnvironmentCore.new(0, 1)
        env.get_weapon_state()            # <- checked here
        env.episode.to_metrics()          # <- and one hop further

    Both were previously invisible to every static check, and both are
    runtime errors no Python test can reach. On a machine without the
    Godot binary this is the only thing between a GDScript typo and a
    failure in CI. Membership *and* argument count are verified, because
    a wrong arity is the same class of compile error.

    Scope is kept narrow so a finding is always real:

    * only locals declared with an explicit project type in the same
      function body; the scope resets at every ``func`` declaration;
    * the type must resolve to a project script whose full member set is
      knowable (``all_members`` returns ``None`` once the inheritance
      chain leaves the project, e.g. ``extends Node``);
    * the one-hop form additionally needs the property to carry a type
      annotation that resolves to another project script;
    * a local that is later reassigned is dropped, since the new value
      may be of any type;
    * ``UNIVERSAL_MEMBERS`` (``free``, ``call``, ``get`` ...) are engine
      API available on everything.
    """
    findings: list[Finding] = []
    for info in index.by_res.values():
        ## local name -> the script it was declared as, until the function ends.
        scope: dict[str, ScriptInfo] = {}
        for line_number, raw in enumerate(info.lines, start=1):
            cleaned = _strip_strings_and_comments(raw)
            if not cleaned.strip():
                continue
            if _FUNC_DECL_RE.match(cleaned):
                scope = {}
                continue

            rebind = _LOCAL_REBIND_RE.match(cleaned)
            if rebind:
                scope.pop(rebind.group(1), None)

            declaration = _TYPED_LOCAL_RE.match(cleaned)
            if declaration:
                local = declaration.group(1)
                alias = declaration.group(2) or declaration.group(3)
                scope.pop(local, None)
                # Note: no `continue`. The right-hand side of a declaration
                # is the most common place to call a method on an existing
                # local (`var d: Dictionary = env.get_metrics()`), so the
                # line still has to be scanned for calls below.
                if alias not in BUILTIN_TYPES and (
                    alias in info.preloads or alias in index.by_class
                ):
                    target = index.resolve(alias, info)
                    if target is not None and index.all_members(target) is not None:
                        scope[local] = target

            if not scope:
                continue

            # One hop through a typed property: `local.prop.method(...)`.
            for match in _LOCAL_CHAIN_CALL_RE.finditer(cleaned):
                local, prop, method = match.groups()
                holder = scope.get(local)
                if holder is None or method in UNIVERSAL_MEMBERS:
                    continue
                prop_type = holder.member_types.get(prop)
                if prop_type is None or prop_type in BUILTIN_TYPES:
                    continue
                target = index.resolve(prop_type, holder) or index.by_class.get(prop_type)
                if target is None:
                    continue
                members = index.all_members(target)
                if members is None:
                    continue
                label = f"{local}.{prop}"
                if method not in members:
                    findings.append(
                        Finding(
                            info.res_path,
                            line_number,
                            "unknown-member",
                            f"{label}.{method}() is not declared by {target.res_path}",
                        )
                    )
                    continue
                problem = _arity_finding(
                    info, target, label, method, cleaned, match.end(), line_number
                )
                if problem is not None:
                    findings.append(problem)

            # Direct call on the local: `local.method(...)`.
            for match in _LOCAL_MEMBER_CALL_RE.finditer(cleaned):
                local, method = match.group(1), match.group(2)
                target = scope.get(local)
                if target is None or method in UNIVERSAL_MEMBERS:
                    continue
                members = index.all_members(target)
                if members is None:
                    continue
                if method not in members:
                    findings.append(
                        Finding(
                            info.res_path,
                            line_number,
                            "unknown-member",
                            f"{local}.{method}() is not declared by the type of {local}",
                        )
                    )
                    continue
                problem = _arity_finding(
                    info, target, local, method, cleaned, match.end(), line_number
                )
                if problem is not None:
                    findings.append(problem)
    return findings


def _enum_members(index: "ProjectIndex", info: ScriptInfo, name: str) -> set[str] | None:
    """Members of the named enum ``name`` on ``info`` or a project base."""
    seen: set[str] = set()
    current: ScriptInfo | None = info
    while current is not None and current.res_path not in seen:
        seen.add(current.res_path)
        if name in current.enum_values:
            return current.enum_values[name]
        base = current.extends
        if not base or base in BUILTIN_TYPES or "." in base:
            return None
        nxt = index.by_class.get(base) or index.by_res.get(base)
        if nxt is None:
            base_res = current.preloads.get(base)
            nxt = index.by_res.get(base_res) if base_res else None
        current = nxt
    return None


def check_enum_members(index: ProjectIndex) -> list[Finding]:
    """Flags `Alias.Enum.MEMBER` where MEMBER is not in that enum.

    ``check_symbols`` stops one level too early: it validates that
    ``CurriculumConfig.Level`` exists and never looks at what follows, so
    ``CurriculumConfig.Level.STATIC_TARGETS`` (the real name is
    ``STATIONARY_TARGET``) sailed through and only failed when the engine
    compiled the script. Enum members are compile-time constants in
    GDScript, so this is decidable statically.
    """
    findings: list[Finding] = []
    for info in index.by_res.values():
        for line_number, raw in enumerate(info.lines, start=1):
            cleaned = _strip_strings_and_comments(raw)
            if not cleaned.strip():
                continue
            for match in _NESTED_ENUM_RE.finditer(cleaned):
                alias, enum_name, member = match.groups()
                if alias in BUILTIN_TYPES:
                    continue
                if alias not in info.preloads and alias not in index.by_class:
                    continue
                target = index.resolve(alias, info)
                if target is None:
                    continue
                values = _enum_members(index, target, enum_name)
                if values is None or member in values:
                    continue
                findings.append(
                    Finding(
                        info.res_path,
                        line_number,
                        "unknown-enum-member",
                        (
                            f"{alias}.{enum_name}.{member} is not a member of "
                            f"enum {enum_name} in {target.res_path}"
                        ),
                    )
                )
    return findings


def parse_all(root: Path | str | None = None) -> list[Finding]:
    """Runs the gdtoolkit grammar over every project script."""
    root = Path(root) if root else project_root()
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


def lint_all(root: Path | str | None = None) -> list[Finding]:
    """Runs gdtoolkit's ``gdlint`` style/complexity rules over the project.

    Separate from :func:`analyze` on purpose: ``analyze`` answers "is this
    code *wrong*" (syntax, missing resources, unknown members, wrong arity)
    while this answers "is this code *unidiomatic*". They fail different
    tests so a style nit is never mistaken for a broken simulation.

    Returns a ``parser-unavailable`` finding (never an exception) when
    gdtoolkit is not installed, matching :func:`parse_all`.
    """
    root = Path(root) if root else project_root()
    try:
        from gdtoolkit.linter import lint_code  # type: ignore
        from gdtoolkit.linter import DEFAULT_CONFIG  # type: ignore
    except ImportError:
        return [
            Finding(
                "<gdtoolkit>",
                0,
                "linter-unavailable",
                "gdtoolkit is not installed; GDScript style was NOT verified",
            )
        ]
    config = dict(DEFAULT_CONFIG)
    findings: list[Finding] = []
    for path in iter_gd_files(root):
        res_path = RES_PREFIX + path.relative_to(root).as_posix()
        try:
            problems = lint_code(path.read_text(encoding="utf-8"), config)
        except Exception as exc:  # noqa: BLE001 - lark/gdtoolkit raise many types
            findings.append(Finding(res_path, 0, "lint-error", str(exc).splitlines()[0]))
            continue
        for problem in problems:
            findings.append(
                Finding(res_path, int(problem.line), str(problem.name), str(problem.description))
            )
    return findings


def analyze(root: Path | str | None = None) -> list[Finding]:
    """Full static analysis: syntax + resources + symbols + call arity +
    static calls + undefined local-method calls + typed-local calls."""
    root = Path(root) if root else project_root()
    index = ProjectIndex(root)
    findings = parse_all(root)
    findings.extend(check_resource_paths(index))
    findings.extend(check_symbols(index))
    findings.extend(check_call_arity(index))
    findings.extend(check_static_calls(index))
    findings.extend(check_local_method_calls(index))
    findings.extend(check_typed_local_calls(index))
    findings.extend(check_enum_members(index))
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
