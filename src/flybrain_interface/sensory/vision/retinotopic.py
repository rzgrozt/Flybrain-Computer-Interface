"""Validated measured viewing directions for MaleCNS optic columns."""

from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from flybrain_interface.connectome_data.optic_columns import OpticColumnRecord

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class MeasuredColumnDirection:
    """One measured visual direction in a forward/right/up coordinate frame."""

    side: str
    column: str
    p: int
    q: int
    forward: float
    right: float
    up: float
    elevation_deg: float
    azimuth_deg: float
    source_raw_p: int
    source_raw_q: int

    def __post_init__(self) -> None:
        if self.side not in {"L", "R"}:
            raise ValueError("direction side must be L or R")
        expected_column = column_name_from_pq(self.side, self.p, self.q)
        if self.column != expected_column:
            raise ValueError(
                f"column name does not match p/q coordinates: "
                f"{self.column} != {expected_column}"
            )
        values = (
            self.forward,
            self.right,
            self.up,
            self.elevation_deg,
            self.azimuth_deg,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("measured direction values must be finite")
        norm = math.sqrt(
            self.forward * self.forward
            + self.right * self.right
            + self.up * self.up
        )
        if not math.isclose(norm, 1.0, abs_tol=1e-8, rel_tol=0.0):
            raise ValueError("measured direction vector must be unit length")
        expected_elevation = math.degrees(math.asin(self.up))
        expected_azimuth = math.degrees(math.atan2(self.right, self.forward))
        if not math.isclose(
            self.elevation_deg,
            expected_elevation,
            abs_tol=1e-7,
            rel_tol=0.0,
        ):
            raise ValueError("elevation does not match direction vector")
        if not math.isclose(
            self.azimuth_deg,
            expected_azimuth,
            abs_tol=1e-7,
            rel_tol=0.0,
        ):
            raise ValueError("azimuth does not match direction vector")


def column_pq(column: OpticColumnRecord) -> tuple[int, int]:
    """Convert MaleCNS hex1/hex2 ROI labels to centered medulla p/q."""

    return column.grid_column - 19, column.grid_row - 18


def column_name_from_pq(side: str, p: int, q: int) -> str:
    """Convert centered medulla p/q back to the official ROI naming convention."""

    if side not in {"L", "R"}:
        raise ValueError("column side must be L or R")
    hex1 = q + 18
    hex2 = p + 19
    if hex1 < 0 or hex2 < 0:
        raise ValueError("p/q coordinates cannot be represented as column labels")
    return f"ME_{side}_col_{hex1:02d}_{hex2:02d}"


def load_measured_column_directions(
    path: Path,
) -> tuple[MeasuredColumnDirection, ...]:
    """Load and validate a generated measured-direction CSV."""

    if not path.is_file():
        raise FileNotFoundError(path)
    expected_fields = {
        "side",
        "column",
        "p",
        "q",
        "forward",
        "right",
        "up",
        "elevation_deg",
        "azimuth_deg",
        "source_raw_p",
        "source_raw_q",
    }
    records: list[MeasuredColumnDirection] = []
    seen_columns: set[str] = set()
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        missing = expected_fields - fields
        if missing:
            raise ValueError(
                f"measured-direction CSV missing fields: {sorted(missing)}"
            )
        for row_number, row in enumerate(reader, start=2):
            record = MeasuredColumnDirection(
                side=str(row["side"]),
                column=str(row["column"]),
                p=int(row["p"]),
                q=int(row["q"]),
                forward=float(row["forward"]),
                right=float(row["right"]),
                up=float(row["up"]),
                elevation_deg=float(row["elevation_deg"]),
                azimuth_deg=float(row["azimuth_deg"]),
                source_raw_p=int(row["source_raw_p"]),
                source_raw_q=int(row["source_raw_q"]),
            )
            if record.column in seen_columns:
                raise ValueError(
                    f"duplicate measured column at CSV row {row_number}: "
                    f"{record.column}"
                )
            seen_columns.add(record.column)
            records.append(record)

    if not records:
        raise ValueError("measured-direction CSV contains no records")
    records.sort(key=lambda record: (record.side, record.q, record.p))
    return tuple(records)

@dataclass(frozen=True, slots=True)
class VirtualScreenConfig:
    """Perspective virtual monitor in the fly forward/right/up frame."""

    horizontal_fov_deg: float
    vertical_fov_deg: float

    def __post_init__(self) -> None:
        for name, value in (
            ("horizontal_fov_deg", self.horizontal_fov_deg),
            ("vertical_fov_deg", self.vertical_fov_deg),
        ):
            if not math.isfinite(value) or not 0.0 < value < 180.0:
                raise ValueError(f"{name} must be finite and inside (0, 180)")


@dataclass(frozen=True, slots=True)
class ScreenProjection:
    """Projected viewing direction on normalized screen coordinates."""

    column: str
    visible: bool
    u: float | None
    v: float | None


def project_direction_to_screen(
    direction: MeasuredColumnDirection,
    config: VirtualScreenConfig,
) -> ScreenProjection:
    """Project a measured viewing ray onto a centered virtual screen.

    u and v are normalized to [0, 1], with u increasing to screen right
    and v increasing downwards. Directions behind the forward plane or outside
    the configured perspective frustum are marked invisible.
    """

    if direction.forward <= 0.0:
        return ScreenProjection(
            column=direction.column,
            visible=False,
            u=None,
            v=None,
        )

    horizontal_scale = math.tan(math.radians(config.horizontal_fov_deg) / 2.0)
    vertical_scale = math.tan(math.radians(config.vertical_fov_deg) / 2.0)
    normalized_x = (direction.right / direction.forward) / horizontal_scale
    normalized_y = (direction.up / direction.forward) / vertical_scale
    if abs(normalized_x) > 1.0 or abs(normalized_y) > 1.0:
        return ScreenProjection(
            column=direction.column,
            visible=False,
            u=None,
            v=None,
        )

    return ScreenProjection(
        column=direction.column,
        visible=True,
        u=(normalized_x + 1.0) / 2.0,
        v=(1.0 - normalized_y) / 2.0,
    )


def project_directions_to_screen(
    directions: tuple[MeasuredColumnDirection, ...],
    config: VirtualScreenConfig,
) -> tuple[ScreenProjection, ...]:
    """Project all measured columns while preserving record order."""

    return tuple(
        project_direction_to_screen(direction, config) for direction in directions
    )


def sample_screen_intensity(
    frame: npt.ArrayLike,
    projection: ScreenProjection,
) -> float | None:
    """Sample normalized screen intensity using bilinear interpolation.

    Two-dimensional frames are interpreted directly as normalized intensity.
    Three-dimensional frames use the arithmetic mean across channels as an explicit
    engineering proxy; no fly-specific spectral sensitivity is implied.
    """

    if not projection.visible:
        return None
    assert projection.u is not None
    assert projection.v is not None

    image = np.asarray(frame, dtype=np.float64)
    if image.ndim not in (2, 3):
        raise ValueError("screen frame must have shape (height, width[, channels])")
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError("screen frame dimensions must be positive")
    if image.ndim == 3 and image.shape[2] <= 0:
        raise ValueError("screen frame channel dimension must be positive")
    if not np.isfinite(image).all() or np.any((image < 0.0) | (image > 1.0)):
        raise ValueError(
            "screen frame values must be finite and normalized to [0, 1]"
        )

    intensity = image if image.ndim == 2 else np.mean(image, axis=2)
    x = projection.u * (intensity.shape[1] - 1)
    y = projection.v * (intensity.shape[0] - 1)
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(x0 + 1, intensity.shape[1] - 1)
    y1 = min(y0 + 1, intensity.shape[0] - 1)
    tx = x - x0
    ty = y - y0

    top = (1.0 - tx) * intensity[y0, x0] + tx * intensity[y0, x1]
    bottom = (1.0 - tx) * intensity[y1, x0] + tx * intensity[y1, x1]
    return float((1.0 - ty) * top + ty * bottom)


def sample_screen_columns(
    frame: npt.ArrayLike,
    projections: tuple[ScreenProjection, ...],
) -> dict[str, float]:
    """Return sampled intensity for each visible projected column."""

    samples: dict[str, float] = {}
    for projection in projections:
        value = sample_screen_intensity(frame, projection)
        if value is not None:
            samples[projection.column] = value
    return samples

@dataclass(frozen=True, slots=True)
class RetinotopicScreenEncoding:
    """Spatial screen samples on measured optic-column viewing directions."""

    columns: tuple[str, ...]
    projections: tuple[ScreenProjection, ...]
    frame_timestamps_s: tuple[float, ...]
    intensity_by_frame: FloatArray
    frame_sha256: str

    def __post_init__(self) -> None:
        if len(self.columns) != len(self.projections):
            raise ValueError("columns and projections must align")
        if self.intensity_by_frame.ndim != 2:
            raise ValueError("retinotopic intensity must be two-dimensional")
        if self.intensity_by_frame.shape != (
            len(self.frame_timestamps_s),
            len(self.columns),
        ):
            raise ValueError("retinotopic intensity shape does not match metadata")
        if not np.isfinite(self.intensity_by_frame).all():
            raise ValueError("retinotopic intensity must be finite")
        if np.any((self.intensity_by_frame < 0.0) | (self.intensity_by_frame > 1.0)):
            raise ValueError("retinotopic intensity must be normalized to [0, 1]")


@dataclass(frozen=True, slots=True)
class RetinotopicNeuronEncoding:
    """Screen intensities expanded from optic columns to mapped neurons."""

    neuron_indices: tuple[int, ...]
    neuron_columns: tuple[str, ...]
    frame_timestamps_s: tuple[float, ...]
    intensity_by_frame: FloatArray

    def __post_init__(self) -> None:
        if len(self.neuron_indices) != len(self.neuron_columns):
            raise ValueError("neuron indices and columns must align")
        if len(set(self.neuron_indices)) != len(self.neuron_indices):
            raise ValueError("retinotopic neuron indices must be unique")
        if self.intensity_by_frame.shape != (
            len(self.frame_timestamps_s),
            len(self.neuron_indices),
        ):
            raise ValueError(
                "retinotopic neuron intensity shape does not match metadata"
            )
        if not np.isfinite(self.intensity_by_frame).all():
            raise ValueError("retinotopic neuron intensity must be finite")


def encode_screen_sequence(
    frames: npt.ArrayLike,
    directions: tuple[MeasuredColumnDirection, ...],
    config: VirtualScreenConfig,
    *,
    frame_rate_hz: float,
    start_s: float = 0.0,
) -> RetinotopicScreenEncoding:
    """Sample a frame sequence at every visible measured optic-column direction."""

    if not math.isfinite(frame_rate_hz) or frame_rate_hz <= 0.0:
        raise ValueError("frame_rate_hz must be finite and positive")
    if not math.isfinite(start_s) or start_s < 0.0:
        raise ValueError("start_s must be finite and non-negative")

    image = np.asarray(frames, dtype=np.float64)
    if image.ndim not in (3, 4) or image.shape[0] <= 0:
        raise ValueError(
            "frames must have shape (time, height, width[, channels])"
        )
    if image.shape[1] <= 0 or image.shape[2] <= 0:
        raise ValueError("frame spatial dimensions must be positive")
    if image.ndim == 4 and image.shape[3] <= 0:
        raise ValueError("frame channel dimension must be positive")
    if not np.isfinite(image).all() or np.any((image < 0.0) | (image > 1.0)):
        raise ValueError("frame values must be finite and normalized to [0, 1]")

    projected = project_directions_to_screen(directions, config)
    visible = tuple(projection for projection in projected if projection.visible)
    columns = tuple(projection.column for projection in visible)
    samples = np.empty((image.shape[0], len(visible)), dtype=np.float64)
    for frame_index in range(image.shape[0]):
        for column_index, projection in enumerate(visible):
            value = sample_screen_intensity(image[frame_index], projection)
            assert value is not None
            samples[frame_index, column_index] = value

    timestamps = tuple(
        start_s + frame_index / frame_rate_hz
        for frame_index in range(image.shape[0])
    )
    canonical = np.asarray(image, dtype="<f8", order="C")
    digest = hashlib.sha256()
    digest.update(str(canonical.shape).encode("ascii"))
    digest.update(canonical.tobytes(order="C"))
    samples.flags.writeable = False
    return RetinotopicScreenEncoding(
        columns=columns,
        projections=visible,
        frame_timestamps_s=timestamps,
        intensity_by_frame=samples,
        frame_sha256=digest.hexdigest(),
    )


def expand_screen_encoding_to_neurons(
    encoding: RetinotopicScreenEncoding,
    neuron_columns: tuple[tuple[int, str], ...],
    *,
    background_intensity: float = 0.5,
) -> RetinotopicNeuronEncoding:
    """Expand column samples to neurons, using adapted background off-screen."""

    if (
        not math.isfinite(background_intensity)
        or not 0.0 <= background_intensity <= 1.0
    ):
        raise ValueError("background_intensity must be finite and in [0, 1]")
    ordered = tuple(sorted(neuron_columns, key=lambda item: item[0]))
    if len({index for index, _ in ordered}) != len(ordered):
        raise ValueError("neuron_columns contains duplicate neuron indices")
    column_lookup = {column: index for index, column in enumerate(encoding.columns)}
    intensity = np.full(
        (len(encoding.frame_timestamps_s), len(ordered)),
        background_intensity,
        dtype=np.float64,
    )
    for neuron_column, (_, column) in enumerate(ordered):
        source_column = column_lookup.get(column)
        if source_column is not None:
            intensity[:, neuron_column] = encoding.intensity_by_frame[:, source_column]
    intensity.flags.writeable = False
    return RetinotopicNeuronEncoding(
        neuron_indices=tuple(index for index, _ in ordered),
        neuron_columns=tuple(column for _, column in ordered),
        frame_timestamps_s=encoding.frame_timestamps_s,
        intensity_by_frame=intensity,
    )
