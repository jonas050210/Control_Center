"""Seeing a captured frame: reading pixels, lifting shadow, boxing figures.

The TTK bridge could already photograph the Roblox client and report the
file's size. What it could not do is *look at the picture* - so "analyse
the HUD" answered a question about the PNG header, and a frame that was
too dark to read produced exactly the same report as a clear one.

This module is the missing half, and it is deliberately built on nothing
but :mod:`numpy` and the standard library (:mod:`zlib` for the PNG
container): a screenshot analysis that refuses to run unless an optional
imaging wheel is installed is an analysis that mostly does not run.

What it can and cannot do, stated once so no caller has to guess:

* :func:`lift_shadows` is local tone mapping - each pixel is compared
  with the illumination of its neighbourhood, which is what makes a
  figure standing in a dark corridor readable without washing out the
  bright parts of the same frame. It changes how the frame *is shown*;
  it invents no detail that is not in the file.
* :func:`detect_figures` is an edge-and-shape heuristic, not a trained
  detector. It finds compact, roughly upright, high-contrast blobs and
  scores them. It will miss a player who is the same colour as the wall
  behind them, and it will happily box a lamp post. Every box it
  returns carries the score and the contrast it was chosen for, so a
  caller (or an operator) can judge the answer instead of trusting it.

Everything here is pure: arrays in, arrays and plain dictionaries out.
No path handling, no GUI, no Roblox knowledge - :mod:`sandboxai.ttk_testing`
decides *which* frame to read and where the annotated copy goes.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "PNG_SIGNATURE",
    "FrameBox",
    "analyze_frame",
    "annotate_frame",
    "decode_png",
    "detect_figures",
    "encode_png",
    "figure_rows",
    "lift_shadows",
    "luminance",
    "read_png",
    "visibility_view",
    "write_png",
]

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

#: Channel count per PNG colour type. Palette (3) is intentionally absent:
#: a palette frame would need the PLTE chunk applied first, and no capture
#: path in this project produces one.
_CHANNELS_BY_COLOR_TYPE: dict[int, int] = {0: 1, 2: 3, 4: 2, 6: 4}

#: Luminance weights (Rec. 709): the eye's own green bias, which is also
#: what a player's eye uses to pick a silhouette out of a background.
_LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


class FrameError(ValueError):
    """Raised when bytes that claim to be a PNG cannot be read as one.

    A separate type because the callers all react the same way - show the
    operator the reason instead of an empty analysis - and because
    ``ValueError`` alone would be indistinguishable from a bad argument
    passed by our own code.
    """


@dataclass(frozen=True)
class FrameBox:
    """One rectangle the detector believes is a figure, in source pixels.

    The score is *why* it was returned, and the contrast is how much it
    stands out from the ring around it. Both are in ``[0, 1]``. A caller
    that wants certainty it does not have should read them rather than
    the rectangle alone.
    """

    x: int
    y: int
    width: int
    height: int
    score: float = 0.0
    contrast: float = 0.0
    clipped: bool = False

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)

    @property
    def aspect(self) -> float:
        """Width / height: a standing figure is narrower than it is tall."""
        return float(self.width) / float(self.height) if self.height else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "center_x": round(self.center[0], 1),
            "center_y": round(self.center[1], 1),
            "aspect": round(self.aspect, 3),
            "score": round(self.score, 4),
            "contrast": round(self.contrast, 4),
            "clipped": self.clipped,
        }


# ---------------------------------------------------------------------------
# PNG container (read/write)
# ---------------------------------------------------------------------------


def _chunk(tag: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + tag
        + payload
        + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
    )


def _parse_png(data: bytes) -> tuple[tuple[int, int, int, int, int], bytes]:
    """Split a PNG stream into its ``(width, height, depth, colour, interlace)``
    header and the concatenated compressed image data.

    Chunk lengths and CRCs are what make a truncated download detectable,
    so a short chunk is an error and not a silent truncation to whatever
    happened to arrive.
    """
    pos = 8
    header: tuple[int, int, int, int, int] | None = None
    idat = bytearray()
    seen_end = False
    total = len(data)
    while pos + 8 <= total:
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        tag = bytes(data[pos + 4 : pos + 8])
        body = bytes(data[pos + 8 : pos + 8 + length])
        if len(body) != length:
            raise FrameError("truncated PNG chunk")
        pos += 12 + length
        if tag == b"IHDR":
            if length != 13:
                raise FrameError("IHDR chunk has an unexpected length")
            width, height, depth, color_type, _comp, _filter, interlace = struct.unpack(
                ">IIBBBBB", body
            )
            header = (width, height, depth, color_type, interlace)
        elif tag == b"IDAT":
            idat += body
        elif tag == b"IEND":
            seen_end = True
            break
    if header is None:
        raise FrameError("PNG has no IHDR chunk")
    if not seen_end or not idat:
        raise FrameError("PNG is truncated (no image data)")
    return header, bytes(idat)


def decode_png(data: bytes) -> np.ndarray:
    """Decode 8-bit non-interlaced PNG bytes into an ``(H, W, 3)`` uint8 array.

    Greyscale and grey+alpha are expanded to three channels, and an alpha
    channel is dropped: the analysis works on what the frame shows, and a
    transparent capture is not a case this project can produce.

    The row unfiltering walks the frame along its anti-diagonals
    (``i + j == const``) instead of row by row. Both PNG filters that
    depend on a neighbour (Sub and Average/Paeth) make a pixel depend on
    the one to its left and the one above it, so pixels on the same
    anti-diagonal are independent of each other and can be reconstructed
    in one vectorised step. Walking rows instead would leave a scan of
    ``width`` sequential bytes per row, which is a pure-Python loop over
    every pixel in the frame - seconds for a 1080p capture.
    """
    if not isinstance(data, (bytes, bytearray)) or bytes(data[:8]) != PNG_SIGNATURE:
        raise FrameError("not a PNG file (bad signature)")

    (width, height, depth, color_type, interlace), idat = _parse_png(bytes(data))
    if width <= 0 or height <= 0:
        raise FrameError(f"PNG declares an empty frame ({width}x{height})")
    if depth != 8:
        raise FrameError(f"only 8-bit PNG is supported (file uses {depth} bits)")
    if interlace:
        raise FrameError("interlaced PNG is not supported")
    if color_type not in _CHANNELS_BY_COLOR_TYPE:
        raise FrameError(f"unsupported PNG colour type {color_type}")

    channels = _CHANNELS_BY_COLOR_TYPE[color_type]
    stride = width * channels
    try:
        raw = zlib.decompress(idat)
    except zlib.error as exc:
        raise FrameError(f"PNG image data is corrupt: {exc}") from exc
    expected = height * (stride + 1)
    if len(raw) < expected:
        raise FrameError(
            f"PNG image data is short ({len(raw)} of {expected} bytes for {width}x{height})"
        )

    flat = np.frombuffer(raw, dtype=np.uint8, count=expected).reshape(height, stride + 1)
    filters = flat[:, 0].astype(np.int16)
    plane = flat[:, 1:].reshape(height, width, channels)
    pixels = _unfilter(plane, filters, channels)

    if channels == 1:
        return np.repeat(pixels, 3, axis=2)
    if channels == 2:
        return np.repeat(pixels[:, :, :1], 3, axis=2)
    return pixels[:, :, :3].copy()


def _unfilter(plane: np.ndarray, filters: np.ndarray, channels: int) -> np.ndarray:
    """Reverses the per-row PNG filters; see :func:`decode_png` for the order.

    Two paths, both exact. When no row uses Average or Paeth the three
    remaining filters (None, Sub, Up) reconstruct a whole row in one
    vectorised step each, which is the common case and costs milliseconds.
    Anything else goes to :func:`_unfilter_diagonals`, whose order is
    correct for all five filters.
    """
    kinds = {int(kind) for kind in filters}
    if not kinds - {0, 1, 2}:
        return _unfilter_rows(plane, filters, channels)
    return _unfilter_diagonals(plane, filters, channels)


def _unfilter_rows(plane: np.ndarray, filters: np.ndarray, channels: int) -> np.ndarray:
    """Rows with filters None, Sub and Up: one step per row, no left scan."""
    height, _width, _ = plane.shape
    out = np.zeros_like(plane, dtype=np.int16)
    for row in range(height):
        kind = int(filters[row])
        if kind == 0:  # None
            out[row] = plane[row]
        elif kind == 1:  # Sub: a prefix sum per channel, mod 256 keeps it a byte
            out[row] = np.cumsum(plane[row].astype(np.int16), axis=0) % 256
        else:  # Up: the row above, which is already reconstructed
            out[row] = plane[row] + (out[row - 1] if row else 0)
    return out.astype(np.uint8)


def _unfilter_diagonals(plane: np.ndarray, filters: np.ndarray, channels: int) -> np.ndarray:
    """Reconstruct every row along the frame's anti-diagonals.

    See :func:`decode_png` for why the anti-diagonal order is the one that
    vectorises: along ``i + j == const`` no pixel depends on another, so
    one step fills a whole diagonal - and both neighbours a PNG filter can
    ask for (left, above) were filled on earlier diagonals.
    """
    height, width, _ = plane.shape
    out = np.zeros((height, width, channels), dtype=np.int16)
    for diagonal in range(height + width - 1):
        i_start = max(0, diagonal - width + 1)
        i_stop = min(height - 1, diagonal)
        if i_stop < i_start:
            continue
        i_idx = np.arange(i_start, i_stop + 1, dtype=np.intp)
        j_idx = diagonal - i_idx
        has_left = j_idx > 0
        has_up = i_idx > 0
        left_col = np.where(has_left, j_idx - 1, 0)
        up_row = np.where(has_up, i_idx - 1, 0)

        raw = plane[i_idx, j_idx].astype(np.int16)
        left = np.where(has_left[:, None], out[i_idx, left_col], 0)
        up = np.where(has_up[:, None], out[up_row, j_idx], 0)
        up_left = np.where(has_up[:, None] & has_left[:, None], out[up_row, left_col], 0)
        kind = filters[i_idx][:, None]

        average = (left + up) >> 1
        estimate = left + up - up_left
        distance_left = np.abs(estimate - left)
        distance_up = np.abs(estimate - up)
        distance_up_left = np.abs(estimate - up_left)
        paeth = np.where(
            (distance_left <= distance_up) & (distance_left <= distance_up_left),
            left,
            np.where(distance_up <= distance_up_left, up, up_left),
        )
        prediction = np.where(
            kind == 0,
            0,
            np.where(
                kind == 1,
                left,
                np.where(kind == 2, up, np.where(kind == 3, average, paeth)),
            ),
        )
        out[i_idx, j_idx] = (raw + prediction) & 0xFF
    return out.astype(np.uint8)


def encode_png(image: np.ndarray, *, compression: int = 6) -> bytes:
    """Encode an ``(H, W, 3)`` uint8 array as 8-bit RGB PNG bytes.

    Filter type 0 (None) on every row: the encoder's job here is to write
    an annotated copy a human opens in an image viewer, and every viewer
    decodes unfiltered rows.
    """
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise FrameError(f"expected an (H, W, 3) image, got {array.shape}")
    height, width, _ = array.shape
    body = np.zeros((height, width * 3 + 1), dtype=np.uint8)
    body[:, 1:] = array.reshape(height, width * 3)
    payload = zlib.compress(body.tobytes(), compression)
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return PNG_SIGNATURE + _chunk(b"IHDR", header) + _chunk(b"IDAT", payload) + _chunk(b"IEND", b"")


def read_png(path: str | Path) -> np.ndarray:
    """Read one PNG file into an ``(H, W, 3)`` uint8 array."""
    data = Path(path).read_bytes()
    return decode_png(data)


def write_png(image: np.ndarray, path: str | Path, *, compression: int = 6) -> Path:
    """Write an ``(H, W, 3)`` uint8 array to ``path`` and return the path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(encode_png(image, compression=compression))
    return target


# ---------------------------------------------------------------------------
# Visibility: the part that answers "a screen in the GUI cannot see the dark"
# ---------------------------------------------------------------------------


def luminance(image: np.ndarray) -> np.ndarray:
    """Per-pixel Rec. 709 luminance in ``[0, 1]``, shape ``(H, W)``."""
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        return np.clip(array / 255.0, 0.0, 1.0)
    return np.clip((array[:, :, :3] * _LUMA_WEIGHTS).sum(axis=2) / 255.0, 0.0, 1.0)


def _box_mean(values: np.ndarray, radius: int) -> np.ndarray:
    """Mean over a ``(2r+1)`` square, via an integral image.

    Cost is independent of ``radius``, which is the point: the lift needs a
    neighbourhood big enough to be "the lighting here" (a head's width at
    1080p), and a direct convolution at that radius is the slowest thing
    in the pipeline.
    """
    if radius < 1:
        return values
    height, width = values.shape
    size = 2 * radius + 1
    padded = np.pad(values.astype(np.float32), radius, mode="edge")
    integral = np.zeros((height + 2 * radius + 1, width + 2 * radius + 1), dtype=np.float64)
    integral[1:, 1:] = padded.cumsum(axis=0).cumsum(axis=1)
    total = (
        integral[size:, size:]
        - integral[:height, size:]
        - integral[size:, :width]
        + integral[:height, :width]
    )
    return (total / float(size * size)).astype(np.float32)


def lift_shadows(
    image: np.ndarray,
    *,
    radius: int = 12,
    gamma: float = 0.55,
    saturation: float = 1.25,
    floor: float = 0.06,
) -> np.ndarray:
    """Return the frame with its shadowed detail brought into view.

    A gain is computed per pixel: how the pixel's brightness compares with
    the brightness of its neighbourhood. Dividing by that local
    illumination is what makes a dark figure in a dark corner readable -
    the same reason a camera's shadow lift works - while a bright sky
    behind it stays where it is instead of clipping to white. The gain is
    applied to all three channels, so hues survive the lift; ``saturation``
    then pushes the colours apart, because "which team is that" is a
    colour question and tone mapping alone flattens it.

    ``radius`` is the neighbourhood in pixels (about a head's width at
    1080p), ``gamma`` below 1 lifts mid-tones, ``floor`` keeps a
    completely black neighbourhood from dividing by zero.
    """
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] < 3:
        raise FrameError(f"expected an (H, W, 3) image, got {array.shape}")
    rgb = np.clip(array[:, :, :3] / 255.0, 0.0, 1.0)
    lum = luminance(rgb)
    local = _box_mean(lum, max(1, int(radius)))
    # Reflectance: how this pixel compares with the light falling here.
    shaped = np.power(lum / (local + floor), max(0.05, float(gamma)))

    # Percentile stretch: a frame's usable range is where its pixels
    # actually are, and a dark capture occupies a sliver of [0, 1].
    low = float(np.percentile(shaped, 1.0))
    high = float(np.percentile(shaped, 99.0))
    if high - low < 1e-6:
        high = low + 1e-6
    target = np.clip((shaped - low) / (high - low), 0.0, 1.0)

    # Colour is carried over as chromaticity, not as a gain: dividing by the
    # old luminance would amplify sensor noise in the darkest pixels, while
    # "this pixel's hue, at the new brightness" keeps the lifted frame's
    # luminance exactly where the tone mapping put it.
    chromatic = rgb / np.maximum(lum, 0.02)[:, :, None]
    chromatic_luma = (chromatic * _LUMA_WEIGHTS).sum(axis=2, keepdims=True)
    chromatic = chromatic / np.maximum(chromatic_luma, 1e-4)
    lifted = np.clip(target[:, :, None] * chromatic, 0.0, 1.0)
    if saturation != 1.0:
        lifted = np.clip(
            target[:, :, None] + float(saturation) * (lifted - target[:, :, None]), 0.0, 1.0
        )
    return (lifted * 255.0 + 0.5).astype(np.uint8)


# ---------------------------------------------------------------------------
# Figures: the part that answers "boxes around the enemies that are in view"
# ---------------------------------------------------------------------------


def _edge_map(values: np.ndarray) -> np.ndarray:
    """Gradient magnitude: where the frame has a boundary, not a colour."""
    down = np.abs(np.diff(values, axis=0, prepend=values[:1]))
    right = np.abs(np.diff(values, axis=1, prepend=values[:, :1]))
    return down + right


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """4-connected regions of a boolean grid, as lists of ``(row, col)``.

    Iterative flood fill over a coarse grid: the grid is the frame pooled
    down to a few thousand cells, so the per-cell Python cost is what a
    single numpy pass over the same pixels would be.
    """
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    groups: list[list[tuple[int, int]]] = []
    for row in range(height):
        row_start = None
        for col in range(width + 1):
            filled = col < width and bool(mask[row, col]) and not seen[row, col]
            if filled:
                if row_start is None:
                    row_start = col
                continue
            if row_start is None:
                continue
            # Fill each horizontal run, then extend it with the runs of the
            # rows it touches - one stack pass per region instead of one
            # visit per cell of every neighbour lookup.
            group: list[tuple[int, int]] = []
            stack = [(row, row_start, col - 1)]
            seen[row, row_start:col] = True
            while stack:
                r, start, stop = stack.pop()
                group.extend((r, c) for c in range(start, stop + 1))
                for neighbour in (r - 1, r + 1):
                    if neighbour < 0 or neighbour >= height:
                        continue
                    c = start
                    while c <= stop:
                        if mask[neighbour, c] and not seen[neighbour, c]:
                            left = c
                            while (
                                left > 0
                                and mask[neighbour, left - 1]
                                and not seen[neighbour, left - 1]
                            ):
                                left -= 1
                            right = c
                            while (
                                right + 1 < width
                                and mask[neighbour, right + 1]
                                and not seen[neighbour, right + 1]
                            ):
                                right += 1
                            seen[neighbour, left : right + 1] = True
                            stack.append((neighbour, left, right))
                            c = right + 1
                        c += 1
            groups.append(group)
            row_start = None
    return groups


def detect_figures(
    image: np.ndarray,
    *,
    max_figures: int = 8,
    min_height_fraction: float = 0.03,
    max_height_fraction: float = 0.95,
    aspect_range: tuple[float, float] = (0.18, 1.10),
    min_fill: float = 0.45,
    sensitivity: float = 2.0,
    cell: int | None = None,
    ambient_radius: int | None = None,
    smoothing: int | None = None,
    enhanced: np.ndarray | None = None,
) -> list[FrameBox]:
    """Find the compact, upright, standing-out blobs in one frame.

    This is shape-and-contrast heuristics, not recognition: it looks for
    regions that differ from their wider surroundings, keeps the ones whose
    bounding box could be a body at a plausible distance, and scores them.
    It knows nothing about Roblox, so it boxes what stands out - a player,
    but also a pillar lit from one side - and every box carries the score
    it earned so the next layer can decide.

    Four steps, each of which exists because the obvious alternative
    failed on a dark capture:

    * **smoothing** - noise is saliency too, and in a low-light frame there
      is more of it than there is figure;
    * **saliency** - how far a pixel sits from the brightness of its
      neighbourhood. Edges were tried first and only light up a figure's
      outline, which fills too little of any box to look like a body;
    * **local normalisation** - the saliency a region needs to reach is a
      multiple of what is typical *around it*, not of what is typical in
      the frame, so a figure in a busy corner survives next to a bright
      wall;
    * **suppression** - comparing against a neighbourhood also lifts a
      ring around every figure, so the weaker box next to a stronger one
      is dropped instead of reported as a second contact.

    ``sensitivity`` is that multiple (higher means fussier). ``cell`` is
    the pooling size in pixels and defaults to a 160-cell-tall grid, which
    is where a torso stops being one cell. ``ambient_radius`` and
    ``smoothing`` default to a sixth and a hundred-twentieth of the frame
    height - roughly a torso and roughly the grain of sensor noise.
    """
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] < 3:
        raise FrameError(f"expected an (H, W, 3) image, got {array.shape}")
    height, width = array.shape[:2]
    if height < 8 or width < 8:
        return []

    # Detection runs on the lifted frame: a figure the operator cannot see
    # in the raw capture is exactly the case this is for.
    basis = np.asarray(enhanced) if enhanced is not None else lift_shadows(array)
    basis_lum = luminance(basis).astype(np.float32)
    if smoothing is None:
        smoothing = max(1, height // 120)
    if smoothing > 1:
        basis_lum = _box_mean(basis_lum, smoothing)
    edges = _edge_map(basis_lum)

    radius = int(ambient_radius) if ambient_radius else max(4, height // 6)
    saliency = np.abs(basis_lum - _box_mean(basis_lum, radius))
    # What is typical *here*, so the threshold follows the scene.
    ambient_saliency = _box_mean(saliency, max(3, radius * 3))
    relative = saliency / (ambient_saliency + 0.01)

    step = int(cell) if cell and cell > 0 else max(2, int(round(height / 160.0)))
    grid_h = max(1, height // step)
    grid_w = max(1, width // step)
    crop_h, crop_w = grid_h * step, grid_w * step
    grid = relative[:crop_h, :crop_w].reshape(grid_h, step, grid_w, step).mean(axis=(1, 3))
    mask = grid > max(1.05, float(sensitivity))
    if not mask.any():
        return []

    saliency_scale = float(np.percentile(saliency, 99.0)) or 1.0
    edge_scale = float(np.percentile(edges, 99.0)) or 1.0
    boxes: list[FrameBox] = []
    for group in _components(mask):
        rows = [row for row, _col in group]
        cols = [col for _row, col in group]
        top, bottom = min(rows), max(rows)
        left, right = min(cols), max(cols)
        fill = len(group) / float((bottom - top + 1) * (right - left + 1))
        if fill < min_fill:
            continue

        y0 = min(height - 1, top * step)
        y1 = min(height, (bottom + 1) * step)
        x0 = min(width - 1, left * step)
        x1 = min(width, (right + 1) * step)
        box_h = max(1, y1 - y0)
        box_w = max(1, x1 - x0)
        height_fraction = box_h / float(height)
        if height_fraction < min_height_fraction or height_fraction > max_height_fraction:
            continue
        aspect = box_w / float(box_h)
        low, high = aspect_range
        if aspect < low or aspect > high:
            continue

        # How far the shape is from "a body": upright and narrower than tall.
        shape_fit = float(np.exp(-((aspect - 0.45) ** 2) / (2 * 0.35**2)))
        stands_out = float(np.clip(saliency[y0:y1, x0:x1].mean() / saliency_scale, 0.0, 1.0))
        detail = float(np.clip(edges[y0:y1, x0:x1].mean() / edge_scale, 0.0, 1.0))
        contrast = _surround_contrast(basis_lum, x0, y0, x1, y1)
        score = float(
            (0.45 + 0.55 * detail)
            * (0.35 + 0.65 * stands_out)
            * shape_fit
            * (0.45 + 0.55 * contrast)
        )
        clipped = x0 <= 0 or y0 <= 0 or x1 >= width or y1 >= height
        boxes.append(
            FrameBox(
                x=int(x0),
                y=int(y0),
                width=int(box_w),
                height=int(box_h),
                score=score,
                contrast=contrast,
                clipped=clipped,
            )
        )

    return _suppress_nearby(boxes)[: max(1, int(max_figures))]


def _suppress_nearby(boxes: list[FrameBox], margin: float = 0.75) -> list[FrameBox]:
    """Drop the weaker box when it sits inside a stronger one's halo.

    Comparing a region with its neighbourhood also lifts a ring around
    every figure, and that ring passes every shape filter on its own. The
    ring is always the weaker of the two and always overlaps the figure's
    box once it is grown by ``margin`` of its own size, so a box whose
    centre lands there is the same contact reported twice.
    """
    kept: list[FrameBox] = []
    for box in sorted(boxes, key=lambda candidate: candidate.score, reverse=True):
        center_x, center_y = box.center
        for stronger in kept:
            pad_x = margin * stronger.width
            pad_y = margin * stronger.height
            inside_x = stronger.x - pad_x <= center_x <= stronger.x + stronger.width + pad_x
            inside_y = stronger.y - pad_y <= center_y <= stronger.y + stronger.height + pad_y
            if inside_x and inside_y:
                break
        else:
            kept.append(box)
    return sorted(kept, key=lambda candidate: candidate.score, reverse=True)


def _surround_contrast(basis: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> float:
    """How much a region stands out from the ring around it, in ``[0, 1]``.

    The detector's own answer to "can this be made out": a figure that is
    the same brightness as what surrounds it is not visible however many
    edges it has inside.
    """
    height, width = basis.shape
    pad_x = max(2, (x1 - x0) // 6)
    pad_y = max(2, (y1 - y0) // 6)
    inner = basis[y0:y1, x0:x1]
    if inner.size == 0:
        return 0.0
    outer_top = max(0, y0 - pad_y)
    outer_bottom = min(height, y1 + pad_y)
    outer_left = max(0, x0 - pad_x)
    outer_right = min(width, x1 + pad_x)
    ring = basis[outer_top:outer_bottom, outer_left:outer_right]
    if ring.size == 0:
        return 0.0
    difference = abs(float(inner.mean()) - float(ring.mean()))
    spread = float(basis.std()) or 1.0
    return float(np.clip(difference / (2.0 * spread), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Annotation and the one-call analysis
# ---------------------------------------------------------------------------


def annotate_frame(
    image: np.ndarray,
    boxes: list[FrameBox],
    *,
    color: tuple[int, int, int] = (255, 64, 64),
    thickness: int = 2,
) -> np.ndarray:
    """Draw every box onto a copy of the frame: outline plus corner ticks.

    Corner ticks rather than a plain rectangle because a player standing
    in front of a bright wall would otherwise hide the very outline that
    is meant to point at them.
    """
    array = np.asarray(image).copy()
    if array.ndim != 3 or array.shape[2] < 3:
        raise FrameError(f"expected an (H, W, 3) image, got {array.shape}")
    height, width = array.shape[:2]
    thickness = max(1, int(thickness))
    for box in boxes:
        x0 = max(0, min(width - 1, box.x))
        x1 = max(0, min(width, box.x + box.width))
        y0 = max(0, min(height - 1, box.y))
        y1 = max(0, min(height, box.y + box.height))
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        array[y0 : y0 + thickness, x0:x1] = color
        array[y1 - thickness : y1, x0:x1] = color
        array[y0:y1, x0 : x0 + thickness] = color
        array[y0:y1, x1 - thickness : x1] = color
        tick = max(4, min(x1 - x0, y1 - y0) // 4)
        for corner_y, step_y in ((y0, 1), (y1 - 1, -1)):
            rows = slice(
                min(corner_y, corner_y + step_y * tick),
                max(corner_y, corner_y + step_y * tick) + 1,
            )
            for corner_x, step_x in ((x0, 1), (x1 - 1, -1)):
                cols = slice(
                    min(corner_x, corner_x + step_x * tick),
                    max(corner_x, corner_x + step_x * tick) + 1,
                )
                array[rows, corner_x : corner_x + thickness] = color
                array[corner_y : corner_y + thickness, cols] = color
    return array


def figure_rows(boxes: list[FrameBox]) -> list[dict[str, Any]]:
    """Render boxes as the rows a table shows, ranked as the detector ranks them.

    ``on_screen`` is always true and is stated anyway: the question the
    operator asks is "which enemies are in view", and a column that can
    only ever say yes is still the column that answers it - a box is
    inside the frame by construction, and ``clipped`` next to it tells
    whether the figure is fully in view or runs off an edge.
    """
    rows: list[dict[str, Any]] = []
    # Ranked here rather than assumed: the rank column is only meaningful
    # if row 1 is the strongest box, and a caller that hands the boxes over
    # in detection order should still get a ranked table.
    for rank, box in enumerate(sorted(boxes, key=lambda item: item.score, reverse=True), start=1):
        row = {"rank": rank, "on_screen": True}
        row.update(box.as_dict())
        rows.append(row)
    return rows


def visibility_view(image: np.ndarray, enhanced: np.ndarray | None = None) -> dict[str, Any]:
    """What the frame's own light looks like, before and after the lift.

    The two numbers that decide whether a capture can be read at all:
    how dark the frame is, and how much of it is too dark to make out.
    Reported for the raw capture and for the lifted one, so "the lift
    helped" is a measurement and not a claim.
    """
    array = np.asarray(image)
    lifted = np.asarray(enhanced) if enhanced is not None else lift_shadows(array)
    raw_lum = luminance(array)
    lifted_lum = luminance(lifted)
    return {
        "mean_luminance": round(float(raw_lum.mean()), 4),
        "median_luminance": round(float(np.median(raw_lum)), 4),
        "shadow_fraction": round(float((raw_lum < 0.12).mean()), 4),
        "enhanced_mean_luminance": round(float(lifted_lum.mean()), 4),
        "enhanced_shadow_fraction": round(float((lifted_lum < 0.12).mean()), 4),
        "contrast_spread": round(float(raw_lum.std()), 4),
    }


def analyze_frame(
    image: np.ndarray,
    *,
    max_figures: int = 8,
    sensitivity: float = 2.0,
) -> dict[str, Any]:
    """Read one frame: lift its shadows, box the figures, and say both plainly.

    Returns the enhanced frame and the annotated frame as arrays - the
    caller decides what to do with them - plus the numbers behind the
    answer. ``contacts`` is empty when nothing stood out, which is a
    result and not a failure; a caller that reports "no contact found"
    is reporting the truth about a frame it could not read.
    """
    array = np.asarray(image)
    enhanced = lift_shadows(array)
    boxes = detect_figures(
        array, max_figures=max_figures, sensitivity=sensitivity, enhanced=enhanced
    )
    annotated = annotate_frame(enhanced, boxes)
    height, width = array.shape[:2]
    return {
        "width": int(width),
        "height": int(height),
        "enhanced": enhanced,
        "annotated": annotated,
        "contacts": boxes,
        "contact_rows": figure_rows(boxes),
        "contact_count": len(boxes),
        "visibility": visibility_view(array, enhanced),
    }
