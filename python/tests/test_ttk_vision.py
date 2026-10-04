"""Tests for the frame reader: PNG decoding, shadow lift and figure boxes.

These are the tests that can be written without a display, a Godot binary
or a Roblox client - which is the point of keeping every step of
``ttk_vision`` a pure function over arrays. The scenes below are synthetic
but built the way a capture is: a gradient background with sensor noise,
figures that differ from their surroundings only in brightness, and a
bright strip at the frame edge that is a wall and not a player.
"""

import struct
import tempfile
import unittest
import zlib

import numpy as np

from sandboxai.ttk_vision import (
    FrameBox,
    FrameError,
    analyze_frame,
    annotate_frame,
    decode_png,
    detect_figures,
    encode_png,
    figure_rows,
    lift_shadows,
    luminance,
    read_png,
    visibility_view,
    write_png,
)

HEIGHT = 360
WIDTH = 640


def _noise(rng: np.random.Generator, shape: tuple[int, ...], scale: float) -> np.ndarray:
    return rng.normal(0.0, scale, shape)


def build_scene(
    *,
    seed: int = 0,
    brightness: float = 1.0,
    figures: tuple[tuple[int, int, int, int, tuple[int, int, int]], ...] = (),
    noise: float = 1.5,
    bright_strip: bool = True,
    height: int = HEIGHT,
    width: int = WIDTH,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """A frame with the given figures, plus their ground-truth rectangles.

    The background rises towards the bottom of the frame so that "brighter
    than the neighbourhood" is not the same as "bright", which is the
    mistake a global threshold would make.
    """
    rng = np.random.default_rng(seed)
    rows = np.mgrid[0:height, 0:width][0]
    background = np.clip(
        (12 + 26 * (rows / height)) * brightness + _noise(rng, (height, width), noise),
        0.0,
        255.0,
    )
    frame = np.repeat(background[:, :, None], 3, axis=2).astype(np.uint8)
    truths: list[tuple[int, int, int, int]] = []
    for center_x, center_y, body_h, body_w, tone in figures:
        color = np.clip(np.array(tone, dtype=float) * brightness, 0, 255)
        top = int(center_y - body_h / 2)
        bottom = int(center_y + body_h / 2)
        left = int(center_x - body_w / 2)
        right = int(center_x + body_w / 2)
        frame[top:bottom, left:right] = color
        head = int(body_w * 0.55)
        frame[max(0, top - head) : top, max(0, center_x - head) : center_x + head] = color
        truths.append((left, max(0, top - head), right, bottom))
    if bright_strip:
        frame[:, width - 40 : width] = min(255, int(190 * brightness))
    return frame, truths


def _iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    x0 = max(first[0], second[0])
    y0 = max(first[1], second[1])
    x1 = min(first[2], second[2])
    y1 = min(first[3], second[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    area_first = (first[2] - first[0]) * (first[3] - first[1])
    area_second = (second[2] - second[0]) * (second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union else 0.0


def _as_tuple(box: FrameBox) -> tuple[int, int, int, int]:
    return (box.x, box.y, box.x + box.width, box.y + box.height)


def _encode_with_filters(image: np.ndarray, filters: tuple[int, ...]) -> bytes:
    """Encode ``image`` with the real PNG filter definitions, one per row.

    The module's own encoder writes unfiltered rows, so the decoder's five
    filter branches are only reachable through a stream built here.
    """

    def paeth(a: int, b: int, c: int) -> int:
        estimate = a + b - c
        distance_left = abs(estimate - a)
        distance_up = abs(estimate - b)
        distance_up_left = abs(estimate - c)
        if distance_left <= distance_up and distance_left <= distance_up_left:
            return a
        if distance_up <= distance_up_left:
            return b
        return c

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    height, width, channels = image.shape
    body = bytearray()
    above = [[0] * channels for _ in range(width)]
    for row in range(height):
        pixels = [[int(value) for value in pixel] for pixel in image[row]]
        kind = filters[row % len(filters)]
        encoded = bytearray([kind])
        for column in range(width):
            for channel in range(channels):
                left = pixels[column - 1][channel] if column else 0
                up = above[column][channel]
                up_left = above[column - 1][channel] if column else 0
                if kind == 0:
                    predicted = 0
                elif kind == 1:
                    predicted = left
                elif kind == 2:
                    predicted = up
                elif kind == 3:
                    predicted = (left + up) // 2
                else:
                    predicted = paeth(left, up, up_left)
                encoded.append((pixels[column][channel] - predicted) % 256)
        body += encoded
        above = pixels
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes(body)))
        + chunk(b"IEND", b"")
    )


class PngRoundTripTests(unittest.TestCase):
    def test_every_filter_type_survives_the_round_trip(self):
        rng = np.random.default_rng(4)
        image = rng.integers(0, 256, size=(23, 31, 3)).astype(np.uint8)
        for filters in ((0,), (1,), (2,), (3,), (4,), (0, 1, 2, 3, 4), (4, 3, 2, 1, 0)):
            with self.subTest(filters=filters):
                decoded = decode_png(_encode_with_filters(image, filters))
                self.assertEqual(decoded.shape, image.shape)
                np.testing.assert_array_equal(decoded, image)

    def test_the_own_encoder_writes_frames_it_can_read_back(self):
        rng = np.random.default_rng(5)
        image = rng.integers(0, 256, size=(17, 25, 3)).astype(np.uint8)
        decoded = decode_png(encode_png(image))
        np.testing.assert_array_equal(decoded, image)

    def test_greyscale_and_alpha_frames_become_three_channels(self):
        rng = np.random.default_rng(6)
        values = rng.integers(0, 256, size=(9, 11)).astype(np.uint8)
        for color_type, plane in (
            (0, values[:, :, None]),
            (6, np.repeat(values[:, :, None], 4, axis=2)),
        ):
            with self.subTest(color_type=color_type):
                height, width = values.shape
                body = bytearray()
                for row in range(height):
                    body.append(0)
                    body += plane[row].astype(np.uint8).tobytes()
                header = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)

                def chunk(tag: bytes, payload: bytes) -> bytes:
                    return (
                        struct.pack(">I", len(payload))
                        + tag
                        + payload
                        + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
                    )

                stream = (
                    b"\x89PNG\r\n\x1a\n"
                    + chunk(b"IHDR", header)
                    + chunk(b"IDAT", zlib.compress(bytes(body)))
                    + chunk(b"IEND", b"")
                )
                decoded = decode_png(stream)
                self.assertEqual(decoded.shape, (height, width, 3))
                np.testing.assert_array_equal(decoded[:, :, 0], values)

    def test_a_frame_that_cannot_be_read_says_why(self):
        rng = np.random.default_rng(7)
        image = rng.integers(0, 256, size=(8, 8, 3)).astype(np.uint8)
        good = encode_png(image)
        sixteen_bit = bytearray(good)
        # IHDR: signature(8) + length(4) + 'IHDR'(4) -> bit depth is byte 24.
        sixteen_bit[24] = 16
        interlaced = bytearray(good)
        interlaced[28] = 1
        cases = {
            "not a png": b"not a png at all, just text",
            "16-bit": bytes(sixteen_bit),
            "interlaced": bytes(interlaced),
            "truncated": good[: len(good) // 2],
            "empty": b"",
        }
        for label, stream in cases.items():
            with self.subTest(case=label):
                with self.assertRaises(FrameError) as caught:
                    decode_png(stream)
                self.assertNotEqual(str(caught.exception).strip(), "")

    def test_write_and_read_agree_on_disk(self):
        rng = np.random.default_rng(8)
        image = rng.integers(0, 256, size=(13, 21, 3)).astype(np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = write_png(image, f"{directory}/frame.png")
            self.assertTrue(path.is_file())
            np.testing.assert_array_equal(read_png(path), image)


class ShadowLiftTests(unittest.TestCase):
    def test_a_dark_frame_comes_into_view_without_blowing_out(self):
        frame, _truths = build_scene(seed=1, brightness=0.3, figures=())
        before = visibility_view(frame)
        lifted = lift_shadows(frame)
        after = visibility_view(frame, lifted)
        self.assertGreater(before["shadow_fraction"], 0.5, "the scene must start dark")
        self.assertLess(
            after["enhanced_shadow_fraction"],
            before["shadow_fraction"] / 2.0,
            "half the frame was unreadable and still is",
        )
        self.assertGreater(after["enhanced_mean_luminance"], before["mean_luminance"])
        # A lift that clips everything to white is not a lift.
        self.assertLess(after["enhanced_mean_luminance"], 0.85)

    def test_the_lift_keeps_a_coloured_figure_the_same_colour(self):
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        frame[:, :] = (10, 10, 10)
        frame[100:200, 200:260] = (90, 40, 30)  # a warm figure in a cold room
        lifted = lift_shadows(frame)
        before = frame[150, 230].astype(float)
        after = lifted[150, 230].astype(float)
        self.assertGreater(after[0], after[1])
        self.assertGreater(after[1], after[2])
        self.assertGreater(
            float(after.sum()),
            float(before.sum()),
            "the figure was supposed to become visible, not dimmer",
        )

    def test_an_already_bright_frame_is_left_roughly_where_it_was(self):
        rng = np.random.default_rng(9)
        frame = np.clip(180 + _noise(rng, (HEIGHT, WIDTH), 4.0), 0, 255).astype(np.uint8)
        frame = np.repeat(frame[:, :, None], 3, axis=2)
        before = visibility_view(frame)
        after = visibility_view(frame, lift_shadows(frame))
        self.assertLess(abs(after["enhanced_mean_luminance"] - before["mean_luminance"]), 0.25)


class FigureDetectionTests(unittest.TestCase):
    def test_both_figures_of_a_dim_room_are_boxed(self):
        frame, truths = build_scene(
            seed=0,
            figures=(
                (160, 200, 150, 44, (58, 44, 38)),
                (430, 210, 130, 40, (96, 70, 60)),
            ),
        )
        boxes = detect_figures(frame)
        self.assertEqual(len(truths), 2)
        for truth in truths:
            with self.subTest(truth=truth):
                best = max((_iou(_as_tuple(box), truth) for box in boxes), default=0.0)
                self.assertGreaterEqual(best, 0.5, f"no box landed on {truth}: {boxes}")

    def test_distant_figures_are_boxed_too(self):
        frame, truths = build_scene(
            seed=3,
            figures=(
                (120, 210, 60, 20, (70, 50, 44)),
                (300, 205, 52, 18, (110, 80, 66)),
                (520, 215, 70, 22, (60, 80, 120)),
            ),
        )
        boxes = detect_figures(frame)
        self.assertGreaterEqual(len(boxes), len(truths))
        for truth in truths:
            with self.subTest(truth=truth):
                best = max((_iou(_as_tuple(box), truth) for box in boxes), default=0.0)
                self.assertGreaterEqual(best, 0.5)

    def test_the_same_figures_are_found_in_a_nearly_black_frame(self):
        frame, truths = build_scene(
            seed=2,
            brightness=0.12,
            figures=(
                (200, 190, 160, 46, (58, 44, 38)),
                (450, 220, 120, 38, (96, 70, 60)),
            ),
        )
        self.assertGreater(visibility_view(frame)["shadow_fraction"], 0.9)
        boxes = detect_figures(frame)
        for truth in truths:
            with self.subTest(truth=truth):
                best = max((_iou(_as_tuple(box), truth) for box in boxes), default=0.0)
                self.assertGreaterEqual(best, 0.5)

    def test_an_empty_frame_produces_no_boxes(self):
        frame, _truths = build_scene(seed=5)
        self.assertEqual(detect_figures(frame), [])

    def test_a_bright_wall_is_not_a_figure(self):
        """A lit strip down the frame edge is a wall, not a contact.

        It is the only object in the frame and it is far brighter than its
        surroundings, so a detector that only asks "does it stand out"
        reports a player that is not there.
        """
        frame, _truths = build_scene(seed=5, figures=(), bright_strip=True)
        boxes = detect_figures(frame)
        strip = (WIDTH - 40, 0, WIDTH, HEIGHT)
        for box in boxes:
            with self.subTest(box=_as_tuple(box)):
                self.assertLess(_iou(_as_tuple(box), strip), 0.5)
        self.assertEqual(boxes, [])

    def test_a_figure_running_off_the_frame_edge_is_marked_clipped(self):
        frame, _truths = build_scene(seed=0, figures=((2, 200, 150, 44, (70, 52, 44)),))
        boxes = detect_figures(frame)
        self.assertTrue(boxes, "a figure at the frame edge is still in view")
        self.assertTrue(
            any(box.clipped for box in boxes),
            f"a box touching the frame edge must say so: {[box.as_dict() for box in boxes]}",
        )

    def test_noise_alone_does_not_become_a_crowd(self):
        rng = np.random.default_rng(11)
        noise = np.clip(40 + _noise(rng, (HEIGHT, WIDTH), 12.0), 0, 255).astype(np.uint8)
        frame = np.repeat(noise[:, :, None], 3, axis=2)
        self.assertLessEqual(len(detect_figures(frame)), 1)


class AnalysisTests(unittest.TestCase):
    def test_the_report_agrees_with_the_boxes_it_drew(self):
        frame, truths = build_scene(
            seed=0,
            figures=(
                (160, 200, 150, 44, (58, 44, 38)),
                (430, 210, 130, 40, (96, 70, 60)),
            ),
        )
        report = analyze_frame(frame)
        self.assertEqual(report["contact_count"], len(report["contacts"]))
        self.assertEqual(len(report["contact_rows"]), report["contact_count"])
        self.assertEqual(report["width"], WIDTH)
        self.assertEqual(report["height"], HEIGHT)
        self.assertTrue(report["contacts"])
        annotated = report["annotated"]
        self.assertFalse(
            np.array_equal(annotated, report["enhanced"]),
            "the annotated frame must carry the boxes",
        )
        for box in report["contacts"]:
            self.assertTrue(
                0 <= box.x < WIDTH and 0 <= box.y < HEIGHT,
                "a box is always inside the frame it came from",
            )

    def test_the_drawn_outline_lands_on_the_box_it_describes(self):
        frame = np.zeros((60, 80, 3), dtype=np.uint8)
        frame[:, :] = (20, 20, 24)
        boxes = [FrameBox(x=20, y=10, width=30, height=40)]
        annotated = annotate_frame(frame, boxes, color=(255, 0, 0))
        self.assertFalse(np.array_equal(annotated, frame), "the original is untouched")
        self.assertEqual(frame[10, 20].tolist(), [20, 20, 24])
        self.assertEqual(annotated[10, 25].tolist(), [255, 0, 0])
        self.assertEqual(annotated[30, 20].tolist(), [255, 0, 0])

    def test_rows_are_ranked_and_say_what_the_box_is(self):
        boxes = [
            FrameBox(x=0, y=0, width=10, height=30, score=0.1, contrast=0.2),
            FrameBox(x=0, y=0, width=10, height=30, score=0.9, contrast=0.8),
        ]
        rows = figure_rows(boxes)
        self.assertEqual([row["rank"] for row in rows], [1, 2])
        self.assertEqual(rows[0]["score"], 0.9)
        self.assertTrue(all(row["on_screen"] for row in rows))
        self.assertIn("clipped", rows[0])

    def test_visibility_is_reported_before_and_after_the_lift(self):
        frame, _truths = build_scene(seed=1, brightness=0.25)
        view = visibility_view(frame)
        for key in (
            "mean_luminance",
            "median_luminance",
            "shadow_fraction",
            "enhanced_mean_luminance",
            "enhanced_shadow_fraction",
            "contrast_spread",
        ):
            with self.subTest(key=key):
                self.assertIn(key, view)
        # Honesty: a dark capture stays dark in the raw numbers even though
        # the lifted one can be read.
        self.assertGreater(view["shadow_fraction"], 0.5)

    def test_luminance_matches_the_rec709_weights(self):
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        frame[0, 0] = (255, 255, 255)
        frame[0, 1] = (255, 0, 0)
        frame[1, 0] = (0, 255, 0)
        values = luminance(frame)
        self.assertAlmostEqual(float(values[0, 0]), 1.0, places=5)
        self.assertAlmostEqual(float(values[0, 1]), 0.2126, places=3)
        self.assertAlmostEqual(float(values[1, 0]), 0.7152, places=3)


if __name__ == "__main__":
    unittest.main()
