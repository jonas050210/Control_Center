"""Map selector, geometry preview, map stats, and custom cover editing."""

from __future__ import annotations

import hashlib
import json

import pandas as pd
import streamlit as st

from env.map_io import map_from_json, save_map
from env.maps import ArenaObject, MAP_NAMES, create_map, randomize_cover, serialize_map
from gui.common import CUSTOM_MAP_PATH, get_custom_map
from gui.visuals import DETAIL_PRESETS, build_map_figure


OBJECT_TYPES = ("wall", "crate", "barrel", "ramp", "pillar", "shelf", "platform")


def _map_for_preview(name: str):
    overrides = st.session_state.setdefault("map_preview_overrides", {})
    if name in overrides:
        return overrides[name]
    if name == "Custom":
        return get_custom_map(st)
    return create_map(name)


def _render_custom_editor(arena_map) -> None:
    st.markdown("#### CUSTOM OBJECT EDITOR")
    st.caption("Custom starts empty. Add axis-aligned boxes or a walkable ramp; all sizes are metres. Layouts save locally.")
    export_col, import_col = st.columns(2)
    with export_col:
        st.download_button(
            "⬇ Export Custom JSON",
            data=json.dumps(serialize_map(arena_map), indent=2, ensure_ascii=False, allow_nan=False),
            file_name="custom_map.json",
            mime="application/json",
            key="custom_map_export",
            use_container_width=True,
        )
    with import_col:
        uploaded = st.file_uploader("Import a Custom map JSON", type=["json"], key="custom_map_upload")
        if uploaded is not None:
            raw_data = uploaded.getvalue()
            fingerprint = hashlib.sha256(raw_data).hexdigest()
            if fingerprint != st.session_state.get("custom_map_import_fingerprint"):
                try:
                    imported = map_from_json(raw_data, force_custom_name=True)
                    save_map(CUSTOM_MAP_PATH, imported)
                    st.session_state.map_preview_overrides["Custom"] = imported
                    st.session_state["custom_map_import_fingerprint"] = fingerprint
                    st.session_state["custom_map_notice"] = "Imported and saved Custom map layout."
                    st.session_state.pop("custom_map_load_error", None)
                    st.rerun()
                except (ValueError, OSError) as exc:
                    st.error(f"Could not import map: {exc}")
    with st.form("custom_object_form", clear_on_submit=True):
        left, middle, right = st.columns(3)
        with left:
            object_type = st.selectbox("Type", OBJECT_TYPES, key="custom_object_type")
            x = st.number_input("X centre", min_value=-25.0, max_value=25.0, value=0.0, step=1.0)
        with middle:
            y = st.number_input("Y centre", min_value=-25.0, max_value=25.0, value=0.0, step=1.0)
            width = st.number_input("Width", min_value=0.5, max_value=30.0, value=2.0, step=0.5)
        with right:
            height = st.number_input("Height", min_value=0.25, max_value=10.0, value=1.5, step=0.25)
            depth = st.number_input("Depth", min_value=0.5, max_value=30.0, value=2.0, step=0.5)
        submitted = st.form_submit_button("＋ Add Object", use_container_width=True)
    if submitted:
        updated = arena_map.copy()
        updated.objects.append(ArenaObject(
            x=float(x), y=float(y), z=0.0, width=float(width), height=float(height),
            depth=float(depth), kind=object_type,
            name=f"Custom {object_type.title()} {len(updated.objects) + 1}",
        ))
        st.session_state.map_preview_overrides["Custom"] = updated
        try:
            save_map(CUSTOM_MAP_PATH, updated)
            st.session_state["custom_map_notice"] = "Custom layout saved to data/custom_map.json."
            st.session_state.pop("custom_map_load_error", None)
        except (OSError, ValueError) as exc:
            st.session_state["custom_map_load_error"] = f"Custom map could not be saved: {exc}"
        st.rerun()


def render() -> None:
    st.subheader("🗺️ MAP LAB · 3D GEOMETRY PREVIEW")
    map_name = st.selectbox("SELECT MAP", MAP_NAMES, key="map_lab_selection")
    arena_map = _map_for_preview(map_name)
    notice = st.session_state.pop("custom_map_notice", None)
    if notice:
        st.success(notice)
    load_error = st.session_state.get("custom_map_load_error")
    if load_error:
        st.warning(load_error)

    button_cols = st.columns([1, 1, 2])
    with button_cols[0]:
        if st.button("🎲 Randomize Cover", key="map_randomize", use_container_width=True):
            seed = int(st.session_state.get("map_preview_seed", 2026)) + 1
            st.session_state.map_preview_seed = seed
            randomized = randomize_cover(arena_map, seed)
            st.session_state.map_preview_overrides[map_name] = randomized
            if map_name == "Custom":
                try:
                    save_map(CUSTOM_MAP_PATH, randomized)
                    st.session_state["custom_map_notice"] = "Randomized Custom layout saved to data/custom_map.json."
                except (OSError, ValueError) as exc:
                    st.session_state["custom_map_load_error"] = f"Custom map could not be saved: {exc}"
            st.rerun()
    with button_cols[1]:
        if st.button("↺ Reset Layout", key="map_reset", use_container_width=True):
            st.session_state.map_preview_overrides.pop(map_name, None)
            if map_name == "Custom":
                try:
                    CUSTOM_MAP_PATH.unlink(missing_ok=True)
                    st.session_state.pop("custom_map_load_error", None)
                    st.session_state["custom_map_notice"] = "Custom layout reset to an empty 50 × 50 m map."
                except OSError as exc:
                    st.session_state["custom_map_load_error"] = f"Saved Custom map could not be removed: {exc}"
            st.rerun()
    with button_cols[2]:
        st.caption(arena_map.description)

    stats = st.columns(4)
    stats[0].metric("MAP SIZE", f"{arena_map.width:g} × {arena_map.depth:g} m")
    stats[1].metric("OBJECTS", str(len(arena_map.objects)))
    stats[2].metric("COVER DENSITY", f"{arena_map.cover_density:.1f}%")
    stats[3].metric("AVG SIGHTLINE", f"{arena_map.average_sightline:.1f} m")
    quality = st.selectbox("3D DETAIL", tuple(DETAIL_PRESETS), index=1, key="map_visual_quality")

    st.plotly_chart(build_map_figure(arena_map, show_spawns=True, height=690,
                                     detail_count=DETAIL_PRESETS[quality]),
                    use_container_width=True, key="map_preview_figure")
    breakdown = {}
    for item in arena_map.objects:
        breakdown[item.kind] = breakdown.get(item.kind, 0) + 1
    if breakdown:
        frame = pd.DataFrame([{"Object type": key.title(), "Count": value}
                              for key, value in sorted(breakdown.items())])
        st.dataframe(frame, use_container_width=True, hide_index=True)
    else:
        st.info("This map has no cover objects yet.")

    if map_name == "Custom":
        _render_custom_editor(arena_map)
