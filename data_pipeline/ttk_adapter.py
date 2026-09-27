"""Game Window Adapter & Isolation Layer for SandboxAI (M1).

Provides non-invasive OS-level window bounding box detection for screen capture.
TTK Testing / Roblox logic is kept completely isolated here so the data pipeline
can seamlessly target the Godot sandbox, desktop, or synthetic mock sources.

STRICT BOUNDARY: TTK Testing is ONLY a manual data source.
This module does NOT read process memory, inject code, or automate game inputs.
"""

from __future__ import annotations

import ctypes
import platform
from typing import Any, List, Optional, Tuple


class GameSourceConfig:
    """Configuration presets for supported capture targets."""

    PRESETS = {
        "ttk_testing": {
            "source_name": "ttk_testing",
            "window_title_patterns": ["Roblox", "TTK Testing", "RobloxPlayerBeta"],
            "description": "Manual gameplay recording from TTK Testing [HARDPOINT] on Roblox",
        },
        "godot_sandbox": {
            "source_name": "godot_sandbox",
            "window_title_patterns": ["Godot", "FPS", "VirtualCamera", "SandboxAI"],
            "description": "Manual gameplay recording from local Godot sandbox",
        },
        "desktop": {
            "source_name": "desktop",
            "window_title_patterns": [],
            "description": "Full primary desktop screen capture",
        },
    }


def find_game_window_rect(
    window_patterns: Optional[List[str]] = None,
) -> Optional[Tuple[int, int, int, int]]:
    """Attempts to find the client rectangle (left, top, width, height) of the target game window.

    Uses standard Windows Win32 APIs via ctypes (read-only).
    Returns (left, top, width, height) or None if not found / non-Windows.
    """
    if platform.system() != "Windows":
        return None

    if not window_patterns:
        window_patterns = ["Roblox", "TTK", "Godot"]

    try:
        user32 = ctypes.windll.user32

        # Data structure for EnumWindows callback
        matched_hwnd = None

        def enum_windows_callback(hwnd: int, extra: Any) -> bool:
            nonlocal matched_hwnd
            if not user32.IsWindowVisible(hwnd):
                return True

            length = user32.GetWindowTextLengthW(hwnd)
            if length == 0:
                return True

            buff = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buff, length + 1)
            title = buff.value

            for pattern in window_patterns:
                if pattern.lower() in title.lower():
                    matched_hwnd = hwnd
                    return False  # stop enumeration
            return True

        WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_int, ctypes.c_int)
        user32.EnumWindows(WNDENUMPROC(enum_windows_callback), 0)

        if matched_hwnd is None:
            return None

        # Get client area (inner window excluding OS title bars / borders)
        class RECT(ctypes.Structure):
            _fields_ = [
                ("left", ctypes.c_long),
                ("top", ctypes.c_long),
                ("right", ctypes.c_long),
                ("bottom", ctypes.c_long),
            ]

        class POINT(ctypes.Structure):
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

        client_rect = RECT()
        user32.GetClientRect(matched_hwnd, ctypes.byref(client_rect))

        pt = POINT(client_rect.left, client_rect.top)
        user32.ClientToScreen(matched_hwnd, ctypes.byref(pt))

        width = client_rect.right - client_rect.left
        height = client_rect.bottom - client_rect.top

        if width > 100 and height > 100:
            return (pt.x, pt.y, width, height)

    except Exception:
        pass

    return None
