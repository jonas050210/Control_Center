"""Movable page widgets and savable UI presets for the Control Center.

The desktop Control Center lets an operator rearrange and hide the cards on
its pages and keep the result as a named preset. That is a data problem
before it is a Tk problem, so it lives here, display-free and unit-tested:

* :class:`WidgetSpec` - what a card is (id, title, allowed column spans).
* :class:`LayoutState` - which cards are visible, in which order, at which
  span, per page.
* :func:`normalize` - the single rule that makes an unknown, duplicated or
  half-missing layout safe to load: drop what does not exist, append what is
  missing, clamp what is out of range.
* :class:`PresetStore` - named presets under ``.sandboxai/ui/presets`` with
  atomic writes.

Nothing here imports Tk, so the whole feature is testable in CI without a
display, and a corrupt preset can never stop the window from opening.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "LAYOUT_VERSION",
    "PRESET_MAX_NAME",
    "LayoutState",
    "PresetError",
    "PresetStore",
    "WidgetPlacement",
    "WidgetSpec",
    "default_layout",
    "deserialize",
    "layout_dir",
    "move",
    "normalize",
    "reset_all",
    "reset_page",
    "serialize",
    "set_span",
    "set_visible",
]

LAYOUT_VERSION = 1

#: Preset names become file names; this is the length limit after sanitising.
PRESET_MAX_NAME = 48

_NAME_PATTERN = re.compile(r"[^A-Za-z0-9 _.-]+")


class PresetError(RuntimeError):
    """Raised when a preset name cannot be turned into a safe file name."""


@dataclass(frozen=True)
class WidgetSpec:
    """One movable card on a page."""

    widget_id: str
    title: str
    description: str = ""
    default_span: int = 1
    min_span: int = 1
    max_span: int = 3
    removable: bool = True
    default_visible: bool = True


@dataclass
class WidgetPlacement:
    """Where one card currently sits (order is the position in the flow)."""

    widget_id: str
    span: int
    visible: bool
    order: int

    def clamped(self, spec: WidgetSpec) -> WidgetPlacement:
        span = max(spec.min_span, min(spec.max_span, int(self.span)))
        return WidgetPlacement(
            widget_id=self.widget_id,
            span=span,
            visible=bool(self.visible) or not spec.removable,
            order=int(self.order),
        )


@dataclass
class LayoutState:
    """The whole movable layout: one ordered placement list per page."""

    pages: dict[str, list[WidgetPlacement]] = field(default_factory=dict)
    version: int = LAYOUT_VERSION

    def placements(self, page: str) -> list[WidgetPlacement]:
        return list(self.pages.get(page, ()))

    def visible(self, page: str) -> list[WidgetPlacement]:
        return [placement for placement in self.placements(page) if placement.visible]

    def span_of(self, page: str, widget_id: str) -> int:
        for placement in self.placements(page):
            if placement.widget_id == widget_id:
                return placement.span
        return 1

    def is_visible(self, page: str, widget_id: str) -> bool:
        for placement in self.placements(page):
            if placement.widget_id == widget_id:
                return placement.visible
        return False


#: Type alias for the page -> specs registry pages hand to the layout engine.
SpecRegistry = dict[str, tuple[WidgetSpec, ...]]


def default_layout(registry: SpecRegistry) -> LayoutState:
    """A layout with every registered widget visible in its declared order."""
    pages: dict[str, list[WidgetPlacement]] = {}
    for page, specs in registry.items():
        pages[page] = [
            WidgetPlacement(
                widget_id=spec.widget_id,
                span=spec.default_span,
                visible=spec.default_visible,
                order=index,
            )
            for index, spec in enumerate(specs)
        ]
    return LayoutState(pages=pages)


def _dedupe(placements: list[WidgetPlacement]) -> list[WidgetPlacement]:
    seen: set[str] = set()
    unique: list[WidgetPlacement] = []
    for placement in sorted(placements, key=lambda item: item.order):
        if placement.widget_id in seen:
            continue
        seen.add(placement.widget_id)
        unique.append(placement)
    return unique


def normalize(layout: LayoutState, registry: SpecRegistry) -> LayoutState:
    """Make any loaded layout consistent with the current widget registry.

    Unknown widget ids are dropped (a preset from an older build must not
    resurrect a deleted card), missing ones are appended with their declared
    defaults (a new card must appear for everyone), and every span/order is
    clamped. The result is always a usable layout.
    """
    pages: dict[str, list[WidgetPlacement]] = {}
    for page, specs in registry.items():
        by_id = {spec.widget_id: spec for spec in specs}
        kept = [
            placement.clamped(by_id[placement.widget_id])
            for placement in _dedupe(layout.placements(page))
            if placement.widget_id in by_id
        ]
        known = {placement.widget_id for placement in kept}
        next_order = max((placement.order for placement in kept), default=-1) + 1
        for spec in specs:
            if spec.widget_id in known:
                continue
            kept.append(
                WidgetPlacement(
                    widget_id=spec.widget_id,
                    span=spec.default_span,
                    visible=spec.default_visible,
                    order=next_order,
                )
            )
            next_order += 1
        for index, placement in enumerate(sorted(kept, key=lambda item: item.order)):
            placement.order = index
            if not placement.visible and not by_id[placement.widget_id].removable:
                placement.visible = True
        pages[page] = kept
    return LayoutState(pages=pages, version=LAYOUT_VERSION)


def move(layout: LayoutState, page: str, widget_id: str, delta: int) -> LayoutState:
    """Move one widget ``delta`` positions within its page (clamped)."""
    placements = _dedupe(layout.placements(page))
    index = next(
        (position for position, item in enumerate(placements) if item.widget_id == widget_id),
        None,
    )
    if index is None or delta == 0:
        return layout
    target = max(0, min(len(placements) - 1, index + int(delta)))
    if target == index:
        return layout
    placements.insert(target, placements.pop(index))
    for position, placement in enumerate(placements):
        placement.order = position
    pages = dict(layout.pages)
    pages[page] = placements
    return LayoutState(pages=pages, version=LAYOUT_VERSION)


def set_span(layout: LayoutState, page: str, widget_id: str, span: int) -> LayoutState:
    """Change one widget's column span; the caller clamps against its spec."""
    pages = dict(layout.pages)
    pages[page] = [
        WidgetPlacement(
            widget_id=item.widget_id,
            span=int(span) if item.widget_id == widget_id else item.span,
            visible=item.visible,
            order=item.order,
        )
        for item in _dedupe(layout.placements(page))
    ]
    return LayoutState(pages=pages, version=LAYOUT_VERSION)


def set_visible(layout: LayoutState, page: str, widget_id: str, visible: bool) -> LayoutState:
    """Show or hide one widget (a non-removable widget stays visible)."""
    pages = dict(layout.pages)
    pages[page] = [
        WidgetPlacement(
            widget_id=item.widget_id,
            span=item.span,
            visible=bool(visible) if item.widget_id == widget_id else item.visible,
            order=item.order,
        )
        for item in _dedupe(layout.placements(page))
    ]
    return LayoutState(pages=pages, version=LAYOUT_VERSION)


def reset_page(layout: LayoutState, page: str, registry: SpecRegistry) -> LayoutState:
    """Restore one page to its declared default layout."""
    fresh = default_layout({page: registry.get(page, ())})
    pages = dict(layout.pages)
    pages.update(fresh.pages)
    return LayoutState(pages=pages, version=LAYOUT_VERSION)


def reset_all(registry: SpecRegistry) -> LayoutState:
    """Restore every page to its declared default layout."""
    return default_layout(registry)


def serialize(layout: LayoutState) -> dict[str, Any]:
    """Plain-JSON form of a layout (stable key order, ints only)."""
    return {
        "version": LAYOUT_VERSION,
        "pages": {
            page: [asdict(placement) for placement in _dedupe(placements)]
            for page, placements in sorted(layout.pages.items())
        },
    }


def deserialize(data: Any, registry: SpecRegistry) -> LayoutState:
    """Read a serialized layout, tolerating anything a previous build wrote."""
    if not isinstance(data, dict):
        return default_layout(registry)
    raw_pages = data.get("pages")
    if not isinstance(raw_pages, dict):
        return default_layout(registry)
    pages: dict[str, list[WidgetPlacement]] = {}
    for page, raw in raw_pages.items():
        if not isinstance(page, str) or not isinstance(raw, list):
            continue
        placements: list[WidgetPlacement] = []
        for index, item in enumerate(raw):
            if not isinstance(item, dict):
                continue
            widget_id = item.get("widget_id")
            if not isinstance(widget_id, str):
                continue
            try:
                span = int(item.get("span", 1))
                order = int(item.get("order", index))
            except (TypeError, ValueError):
                span, order = 1, index
            placements.append(
                WidgetPlacement(
                    widget_id=widget_id,
                    span=span,
                    visible=bool(item.get("visible", True)),
                    order=order,
                )
            )
        pages[page] = placements
    return normalize(LayoutState(pages=pages, version=LAYOUT_VERSION), registry)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


def layout_dir(project_root: str | Path) -> Path:
    """``<project>/.sandboxai/ui/presets``."""
    return Path(project_root) / ".sandboxai" / "ui" / "presets"


def safe_preset_name(name: str) -> str:
    """Turn user text into a file-safe preset name, or raise :class:`PresetError`."""
    cleaned = _NAME_PATTERN.sub("", str(name)).strip().strip(".")
    cleaned = re.sub(r"\s+", " ", cleaned)
    if not cleaned:
        raise PresetError("a preset name must contain at least one letter or digit")
    if cleaned in {".", ".."}:
        raise PresetError("that preset name is not allowed")
    return cleaned[:PRESET_MAX_NAME]


class PresetStore:
    """Named UI presets on disk, one JSON file each."""

    def __init__(self, project_root: str | Path) -> None:
        self.project_root = Path(project_root)
        self.directory = layout_dir(self.project_root)
        self.error: str | None = None

    def _path(self, name: str) -> Path:
        return self.directory / f"{safe_preset_name(name)}.json"

    def list_presets(self) -> list[str]:
        """Every readable preset name, sorted case-insensitively."""
        try:
            entries = sorted(self.directory.glob("*.json"))
        except OSError as exc:
            self.error = str(exc)
            return []
        return sorted((entry.stem for entry in entries), key=str.lower)

    def save(
        self,
        name: str,
        layout: LayoutState,
        *,
        appearance: dict[str, Any] | None = None,
        note: str = "",
    ) -> str:
        """Persist one preset; returns the sanitised name that was written."""
        clean = safe_preset_name(name)
        payload = {
            "version": LAYOUT_VERSION,
            "name": clean,
            "note": str(note)[:400],
            "appearance": {key: value for key, value in (appearance or {}).items()},
            "layout": serialize(layout),
        }
        path = self._path(clean)
        try:
            self._atomic_json(path, payload)
        except OSError as exc:
            self.error = f"{path}: {exc}"
            raise PresetError(str(exc)) from exc
        self.error = None
        return clean

    def load(self, name: str) -> dict[str, Any] | None:
        """Read one preset document (raw dict), or ``None`` when unreadable."""
        try:
            path = self._path(name)
        except PresetError as exc:
            self.error = str(exc)
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            self.error = f"{path}: {exc}"
            return None
        if not isinstance(raw, dict):
            self.error = f"{path}: expected a JSON object"
            return None
        return raw

    def load_layout(self, name: str, registry: SpecRegistry) -> LayoutState | None:
        """Read one preset's layout, normalised against the live registry."""
        raw = self.load(name)
        if raw is None:
            return None
        layout = raw.get("layout")
        if not isinstance(layout, dict):
            self.error = f"{name}: preset has no layout"
            return None
        return deserialize(layout, registry)

    def delete(self, name: str) -> bool:
        """Delete a preset, reporting whether a file was actually removed."""
        try:
            path = self._path(name)
        except PresetError as exc:
            self.error = str(exc)
            return False
        try:
            path.unlink()
        except FileNotFoundError:
            return False
        except OSError as exc:
            self.error = f"{path}: {exc}"
            return False
        return True

    def rename(self, old: str, new: str) -> str:
        """Rename a preset; returns the new sanitised name.

        Refuses a name that is already taken. Overwriting would delete the
        other preset without a word - a rename is a tidy-up action, not a
        way to lose an arrangement the operator saved earlier.
        """
        clean = safe_preset_name(new)
        raw = self.load(old)
        if raw is None:
            raise PresetError(self.error or f"preset '{old}' could not be read")
        try:
            source = self._path(old)
            target = self._path(clean)
        except PresetError as exc:
            raise PresetError(str(exc)) from exc
        if target != source and target.exists():
            raise PresetError(f"a preset named '{clean}' already exists")
        raw["name"] = clean
        try:
            self._atomic_json(target, raw)
        except OSError as exc:
            raise PresetError(str(exc)) from exc
        if target != source:
            with contextlib.suppress(PresetError):
                self.delete(old)
        return clean

    @staticmethod
    def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, sort_keys=True)
                stream.write("\n")
            os.replace(temp_name, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(temp_name)
            raise
