"""Tests for Timestamp Synchronization and Action Alignment (M1)."""

import pytest

from data_pipeline.actions import MouseBinner
from data_pipeline.input_listener import RawInputEvent
from data_pipeline.sync import ActionSynchronizer


def test_synchronizer_mouse_integration() -> None:
    sync = ActionSynchronizer()
    t_prev = 10.00
    t_curr = 10.0667

    events = [
        RawInputEvent(t=10.01, event_type="mouse_move", data={"dx": 2.5, "dy": -1.0}),
        RawInputEvent(t=10.03, event_type="mouse_move", data={"dx": 3.0, "dy": -0.5}),
        RawInputEvent(t=10.05, event_type="mouse_move", data={"dx": -1.0, "dy": 2.0}),
    ]

    keys, buttons, dx, dy, wheel = sync.process_interval(events, t_prev, t_curr)
    assert abs(dx - 4.5) < 1e-5
    assert abs(dy - 0.5) < 1e-5
    assert wheel == 0


def test_synchronizer_tap_event_captured() -> None:
    """A fast key tap (press and release within the same 66ms frame window) must be captured."""
    sync = ActionSynchronizer()
    t_prev = 1.000
    t_curr = 1.066

    events = [
        # Jump pressed at 1.02 and released at 1.04
        RawInputEvent(t=1.020, event_type="key_down", data={"key": "space"}),
        RawInputEvent(t=1.040, event_type="key_up", data={"key": "space"}),
    ]

    sample = sync.create_sample(
        step_idx=1,
        t_curr=t_curr,
        t_prev=t_prev,
        frame_file="frames/01.jpg",
        events=events,
    )

    assert sample.actions.jump == 1
    # But after interval ends, persistent held key state should be released
    assert "space" not in sync._held_keys


def test_synchronizer_held_key_across_multiple_frames() -> None:
    sync = ActionSynchronizer()

    # Frame 1: player presses W at t=1.02
    events_f1 = [
        RawInputEvent(t=1.02, event_type="key_down", data={"key": "w"}),
    ]
    s1 = sync.create_sample(1, t_curr=1.066, t_prev=1.000, frame_file="f1.jpg", events=events_f1)
    assert s1.actions.move_y == 1

    # Frame 2: no new key events (W still held)
    s2 = sync.create_sample(2, t_curr=1.133, t_prev=1.066, frame_file="f2.jpg", events=[])
    assert s2.actions.move_y == 1

    # Frame 3: player releases W at t=1.15
    events_f3 = [
        RawInputEvent(t=1.15, event_type="key_up", data={"key": "w"}),
    ]
    s3 = sync.create_sample(3, t_curr=1.200, t_prev=1.133, frame_file="f3.jpg", events=events_f3)
    # During frame 3, W was active in window
    assert s3.actions.move_y == 1

    # Frame 4: no events (W is now released)
    s4 = sync.create_sample(4, t_curr=1.266, t_prev=1.200, frame_file="f4.jpg", events=[])
    assert s4.actions.move_y == 0


def test_synchronizer_mouse_clicks_and_holds() -> None:
    sync = ActionSynchronizer()

    # Left click press in frame 1
    events_f1 = [
        RawInputEvent(t=0.03, event_type="mouse_click", data={"button": "left", "pressed": True}),
    ]
    s1 = sync.create_sample(1, t_curr=0.066, t_prev=0.000, frame_file="f1.jpg", events=events_f1)
    assert s1.actions.fire == 1

    # Held in frame 2
    s2 = sync.create_sample(2, t_curr=0.133, t_prev=0.066, frame_file="f2.jpg", events=[])
    assert s2.actions.fire == 1

    # Released in frame 3
    events_f3 = [
        RawInputEvent(t=0.15, event_type="mouse_click", data={"button": "left", "pressed": False}),
    ]
    s3 = sync.create_sample(3, t_curr=0.200, t_prev=0.133, frame_file="f3.jpg", events=events_f3)
    assert s3.actions.fire == 1

    # Inactive in frame 4
    s4 = sync.create_sample(4, t_curr=0.266, t_prev=0.200, frame_file="f4.jpg", events=[])
    assert s4.actions.fire == 0


def test_synchronizer_out_of_order_events() -> None:
    sync = ActionSynchronizer()
    t_prev = 1.000
    t_curr = 1.0667

    # Events in unordered timestamps
    events = [
        RawInputEvent(t=1.050, event_type="mouse_move", data={"dx": 1.0, "dy": 0.0}),
        RawInputEvent(t=1.010, event_type="key_down", data={"key": "w"}),
        RawInputEvent(t=1.030, event_type="mouse_move", data={"dx": 2.0, "dy": 0.0}),
    ]

    sample = sync.create_sample(1, t_curr=t_curr, t_prev=t_prev, frame_file="f.jpg", events=events)
    assert sample.actions.move_y == 1
    assert abs(sample.actions.mouse_dx - 3.0) < 1e-5


def test_synchronizer_empty_event_list() -> None:
    sync = ActionSynchronizer()
    sample = sync.create_sample(0, t_curr=0.0, t_prev=0.0, frame_file="f0.jpg", events=[])
    assert sample.step_idx == 0
    assert sample.actions.move_x == 0
    assert sample.actions.move_y == 0
    assert sample.actions.mouse_dx == 0.0
    assert sample.actions.mouse_dy == 0.0

