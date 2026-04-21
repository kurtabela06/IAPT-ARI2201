"""
vision/video_config.py
======================
Per-video "sidecar" calibration loader.

For every video clip the pipeline can optionally read a YAML file with the
same stem alongside the video:

    data/raw/video/regional_10am.MP4
    data/raw/video/regional_10am.yaml   ← sidecar (optional)

Any value present in the sidecar overrides the global default from config.py.
Values not present in the sidecar fall back to the global segment values so
you only need to write down what actually changed for this specific clip.

Loaded calibration is returned as a VideoCalibration dataclass so the rest
of the pipeline can access fields directly (no dict key typos).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from vision.config import (
    CALIBRATION,
    CALIBRATION_VIEWS,
    GATE_Y_RATIOS,
    NATIVE_FRAME_WIDTH,
    NATIVE_FRAME_HEIGHT,
    ROI_POLYGONS,
)


# ---------------------------------------------------------------------------
# DATACLASS
# ---------------------------------------------------------------------------

@dataclass
class VideoCalibration:
    """
    All calibration values needed to run the pipeline for one video clip.

    Units / coordinate convention
    ------------------------------
    roi_polygon       : native-frame pixel coordinates (1080 × 1920).
                        Already scaled from the calibration view — do not
                        scale again.
    gate_entry_ratio  : Y fraction of native frame height (0.0 = top).
    gate_exit_ratio   : Y fraction of native frame height (0.0 = top).
    gate_distance_m   : real-world metres between the two gate lines.
    pixels_per_metre  : NATIVE-frame pixels per metre (already scaled from
                        the calibration-view PPM).
    calibration_view  : (width, height) px of the window where the sidecar
                        coords were measured.  Stored for transparency in logs.
    source            : "sidecar" | "global" — for log messages.
    """
    segment:           str
    slot:              str
    pixels_per_metre:  float
    lanes:             int
    gate_entry_ratio:  float
    gate_exit_ratio:   float
    gate_distance_m:   float
    roi_polygon:       Optional[list[tuple[int, int]]]
    calibration_view:  tuple[int, int]
    source:            str = "global"
    notes:             str = ""


# ---------------------------------------------------------------------------
# SCALING HELPERS (sidecar-local, so we don't import config's private ones)
# ---------------------------------------------------------------------------

def _scale_roi(
    points: list[list[int]],
    view_w: int,
    view_h: int,
) -> list[tuple[int, int]]:
    """Scale ROI corners from a calibration view to the 1080 × 1920 native frame."""
    sx = NATIVE_FRAME_WIDTH  / view_w
    sy = NATIVE_FRAME_HEIGHT / view_h
    return [(int(round(x * sx)), int(round(y * sy))) for x, y in points]


# ---------------------------------------------------------------------------
# LOADER
# ---------------------------------------------------------------------------

def _sidecar_path(video_path: Path) -> Path:
    """Return the expected .yaml sidecar path for a given video file."""
    return video_path.with_suffix(".yaml")


def load_video_calibration(video_path: Path, segment: str) -> VideoCalibration:
    """
    Load calibration for one video clip.

    Looks for <video_stem>.yaml in the same directory as the video.
    If found, sidecar values override the global config.py defaults.
    If not found, global defaults are used unchanged.

    Parameters
    ----------
    video_path : path to the video file (e.g. data/raw/video/regional_10am.MP4)
    segment    : road segment name — "regional" | "valley"

    Returns
    -------
    VideoCalibration populated with the final (merged) values.
    """
    # --- Global defaults --------------------------------------------------
    calib_global  = CALIBRATION[segment]
    gate_global   = GATE_Y_RATIOS[segment]
    view_w, view_h = CALIBRATION_VIEWS[segment]

    raw_ppm   = calib_global["pixels_per_metre"]
    sx        = NATIVE_FRAME_WIDTH  / view_w
    scaled_ppm = raw_ppm * sx

    roi_polygon    = ROI_POLYGONS[segment] if ROI_POLYGONS and segment in ROI_POLYGONS else None
    gate_entry     = gate_global["gate_entry"]
    gate_exit      = gate_global["gate_exit"]
    gate_distance  = gate_global["gate_distance_m"]
    lanes          = calib_global["lanes"]
    notes          = ""
    source         = "global"
    effective_view = (view_w, view_h)

    # --- Sidecar override -------------------------------------------------
    sidecar = _sidecar_path(video_path)
    if sidecar.exists():
        with sidecar.open("r") as fh:
            raw: dict = yaml.safe_load(fh) or {}

        source = "sidecar"

        # Calibration-view dimensions (sidecar can declare its own window size)
        sv_w = int(raw.get("calibration_view_width",  view_w))
        sv_h = int(raw.get("calibration_view_height", view_h))
        effective_view = (sv_w, sv_h)

        # PPM — if the sidecar supplies it, it's measured on sv_w / sv_h
        if "pixels_per_metre" in raw:
            raw_ppm    = float(raw["pixels_per_metre"])
            sx         = NATIVE_FRAME_WIDTH / sv_w
            scaled_ppm = raw_ppm * sx

        # ROI polygon — sidecar coords are measured on sv_w × sv_h
        if "roi_polygon" in raw:
            roi_polygon = _scale_roi(raw["roi_polygon"], sv_w, sv_h)

        # Gate ratios and distance
        if "gate_entry_y" in raw:
            gate_entry = float(raw["gate_entry_y"])
        if "gate_exit_y" in raw:
            gate_exit = float(raw["gate_exit_y"])
        if "real_world_distance_meters" in raw:
            gate_distance = float(raw["real_world_distance_meters"])

        if "lanes" in raw:
            lanes = int(raw["lanes"])
        if "notes" in raw:
            notes = str(raw["notes"])

        print(f"  [calib] Sidecar loaded: {sidecar.name}")
    else:
        print(f"  [calib] No sidecar found — using global config.py defaults")
        print(f"          (Create {sidecar.name} to override per-clip)")

    # Derive slot from video stem if not deterministic (caller also passes it)
    slot = video_path.stem.split("_", 1)[-1] if "_" in video_path.stem else video_path.stem

    return VideoCalibration(
        segment          = segment,
        slot             = slot,
        pixels_per_metre = scaled_ppm,
        lanes            = lanes,
        gate_entry_ratio = gate_entry,
        gate_exit_ratio  = gate_exit,
        gate_distance_m  = gate_distance,
        roi_polygon      = roi_polygon,
        calibration_view = effective_view,
        source           = source,
        notes            = notes,
    )


def describe(vc: VideoCalibration) -> None:
    """Print a human-readable summary of a VideoCalibration to stdout."""
    view_w, view_h = vc.calibration_view
    print(f"  Calibration source  : {vc.source.upper()}")
    print(f"  Calibration view    : {view_w} × {view_h} px")
    print(f"  PPM (native)        : {vc.pixels_per_metre:.2f} px/m")
    print(f"  Lanes               : {vc.lanes}")
    print(f"  Gate entry          : {vc.gate_entry_ratio:.3f}  (Y fraction of frame height)")
    print(f"  Gate exit           : {vc.gate_exit_ratio:.3f}")
    print(f"  Gate distance       : {vc.gate_distance_m:.1f} m")
    if vc.roi_polygon:
        print(f"  ROI polygon         : {len(vc.roi_polygon)} points (native px)")
    else:
        print(f"  ROI polygon         : DISABLED")
    if vc.notes:
        print(f"  Notes               : {vc.notes}")
