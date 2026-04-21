"""
vision/detect.py
================
⚠️  LEGACY — NO LONGER PART OF THE ACTIVE PIPELINE  ⚠️

Phase 1 was pivoted to manual ground-truth extraction because of perspective
and tracking inconsistencies in the YOLO + ByteTrack output. This file is
retained only as reference for the CV approach that was attempted. The
active entry point is `main.py`, which drives
`data_processing/manual_to_sumo.py` against CSVs in
`data/raw/manual_counts/`.

Original module docstring follows.

Phase 1 detection & tracking pipeline for a single video clip.

Produces TWO output files per clip:

1. Per-frame CSV  (data/processed/yolo_output/<segment>_<slot>_frames.csv)
   Columns: frame, id, class, speed_kph, entry_time
   One row per detected vehicle per frame — the raw ground truth.

2. Per-minute summary CSV  (data/processed/yolo_output/<segment>_<slot>.csv)
   Aggregated 1-minute bins used downstream by extract_metrics.py
   and SUMO calibration.

Key calibration parameters live in vision/config.py — do NOT edit constants
here; change them there instead.

QUICK START (beginners)
-----------------------
Step 1: Drop your video into   data/raw/video/regional_10am.mp4
        (naming: <segment>_<slot>.mp4   where segment = regional | valley)

Step 2: Run TEST MODE to check the ROI and gate look right on YOUR footage:
    cd malta-road-recovery
    python vision/detect.py --video data/raw/video/regional_10am.mp4 \
        --segment regional --slot 10am --test

    A window will pop up showing the first frame with the polygon and gate
    drawn.  If they look wrong, edit vision/config.py and try again.

Step 3: Once the overlay looks good, run the full clip (no --test flag):
    python vision/detect.py --video data/raw/video/regional_10am.mp4 \
        --segment regional --slot 10am

    Or use main.py to process all clips at once:
    python main.py

All other options
-----------------
    --model    yolov8n.pt | yolov8s.pt | yolov8m.pt  (default: from config)
    --gate     0.55   Y position as fraction of frame height (overrides config)
    --preview         open a live OpenCV window during the full run
    --no-roi          disable ROI masking (useful for debugging)
    --test            process first 60 s only + show first-frame overlay
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from vision.config import (          # noqa: E402
    CALIBRATION,
    VEHICLE_CLASSES,
    DEFAULT_MODEL,
    TRACKER_CFG,
    SPEED_SMOOTH_FRAMES,
    CONFIDENCE_THRESHOLD,
    BIN_SECONDS,
    PER_FRAME_COLUMNS,
    PER_VEHICLE_COLUMNS,
    INFERENCE_DEVICE,
    INFERENCE_IMGSZ,
    FRAME_STRIDE,
    TEST_DURATION_S,
)
from vision.video_config import load_video_calibration, describe as describe_calib  # noqa: E402

# Test-mode preview window size (width, height) — keeps the 1080×1920 frame
# at a readable size on screen while we draw ROI/Gate on the native frame.
TEST_DISPLAY_SIZE = (540, 960)
from vision.utils import (           # noqa: E402
    build_roi_mask,
    apply_roi_mask,
    centroid_in_roi,
    draw_roi,
)
from vision.virtual_gates import (   # noqa: E402
    VirtualGate,
    SpeedTrap,
    TrapCrossing,
    make_speed_trap,
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "yolo_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_DIR = PROJECT_ROOT / "data" / "raw" / "video"


# ---------------------------------------------------------------------------
# ORIENTATION HELPERS
# ---------------------------------------------------------------------------

def _needs_portrait_rotation(frame_w: int, frame_h: int) -> bool:
    """True if the decoded frame arrives in landscape and must be rotated to portrait."""
    return frame_h < frame_w


def _to_portrait(frame: np.ndarray) -> np.ndarray:
    """Rotate a landscape frame 90° clockwise into portrait orientation."""
    return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)


# ---------------------------------------------------------------------------
# SPEED ESTIMATOR
# ---------------------------------------------------------------------------

class SpeedEstimator:
    """
    Converts frame-to-frame pixel displacement into km/h.

    Because we skip frames with FRAME_STRIDE, the time between two
    consecutive observations is  FRAME_STRIDE / fps  seconds, not  1 / fps.
    Pass effective_fps = fps / FRAME_STRIDE to account for this.
    """

    def __init__(self, pixels_per_metre: float, effective_fps: float) -> None:
        self.ppm = pixels_per_metre
        self.fps = effective_fps
        self._prev: dict[int, tuple[float, float]] = {}
        self._buffer: dict[int, list[float]] = defaultdict(list)

    def update(self, track_id: int, cx: float, cy: float) -> float | None:
        """Returns smoothed speed in km/h, or None on first observation."""
        raw = None
        if track_id in self._prev:
            px, py = self._prev[track_id]
            pixel_dist = float(np.hypot(cx - px, cy - py))
            metres     = pixel_dist / self.ppm
            speed_ms   = metres * self.fps
            raw        = speed_ms * 3.6             # → km/h

            # Discard physically impossible spikes (tracker glitch)
            if raw > 180:
                raw = None

        self._prev[track_id] = (cx, cy)

        if raw is not None:
            buf = self._buffer[track_id]
            buf.append(raw)
            if len(buf) > SPEED_SMOOTH_FRAMES:
                buf.pop(0)

        buf = self._buffer.get(track_id, [])
        return float(np.mean(buf)) if buf else None

    def forget(self, track_id: int) -> None:
        self._prev.pop(track_id, None)
        self._buffer.pop(track_id, None)


# ---------------------------------------------------------------------------
# BIN AGGREGATOR  (1-minute summary rows)
# ---------------------------------------------------------------------------

class BinAggregator:
    """Accumulates gate crossings and flushes them into 1-minute summary rows."""

    def __init__(self, bin_seconds: int = BIN_SECONDS) -> None:
        self.bin_seconds   = bin_seconds
        self._current_bin  = -1
        self._counts: dict[str, int] = defaultdict(int)
        self._speeds: list[float]    = []
        self.rows: list[dict]        = []

    def _bin_index(self, ts: float) -> int:
        return int(ts // self.bin_seconds)

    def add(self, ts: float, class_name: str, speed_kph: float | None) -> None:
        idx = self._bin_index(ts)
        if idx != self._current_bin and self._current_bin >= 0:
            self._flush(self._current_bin)
        self._current_bin = idx
        self._counts[class_name] += 1
        if speed_kph is not None and speed_kph > 0:
            self._speeds.append(speed_kph)

    def touch(self, ts: float) -> None:
        """Ensure at least one bin exists for the observed time range.

        Called when we see the video advance past a bin boundary with no
        vehicle counts.  Prevents an empty summary CSV when a short clip
        has no gate crossings.
        """
        idx = self._bin_index(ts)
        if self._current_bin < 0:
            self._current_bin = idx
        elif idx != self._current_bin:
            self._flush(self._current_bin)
            self._current_bin = idx

    def _flush(self, bin_idx: int) -> None:
        ts    = bin_idx * self.bin_seconds
        total = sum(self._counts.values())
        avg   = float(np.mean(self._speeds)) if self._speeds else 0.0
        self.rows.append({
            "timestamp_s":   ts,
            "timestamp_min": round(ts / 60, 2),
            "vehicle_count": total,
            "avg_speed_kmh": round(avg, 2),
            **{f"count_{k}": self._counts.get(k, 0) for k in VEHICLE_CLASSES.values()},
        })
        self._counts = defaultdict(int)
        self._speeds = []

    def finalise(self, last_ts: float) -> None:
        if self._current_bin >= 0:
            self._flush(self._current_bin)


# ---------------------------------------------------------------------------
# TEST-MODE FIRST-FRAME POPUP
# ---------------------------------------------------------------------------

def _show_test_overlay(
    cap: cv2.VideoCapture,
    trap: SpeedTrap,
    roi_polygon: list | None,
    segment: str,
    slot: str,
    rotate_to_portrait: bool = False,
) -> None:
    """
    Grab frame 0, draw the ROI polygon and gate line on the NATIVE-resolution
    frame, then pop up a down-scaled window so the user can visually verify
    their calibration coordinates.  Pressing any key closes the window.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    if not ret:
        print("[test] WARNING: Could not read first frame for overlay preview.")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return

    if rotate_to_portrait:
        frame = _to_portrait(frame)

    # Draw overlays on the native 1080×1920 frame so scaled ROI coords land
    # in the correct place relative to the footage.
    if roi_polygon is not None:
        draw_roi(frame, roi_polygon, label="ROI (edit in config.py)")
    trap.draw(frame)

    h, w = frame.shape[:2]
    cv2.putText(
        frame,
        f"TEST MODE — {segment} / {slot}   Press any key to start processing...",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2,
    )
    cv2.putText(
        frame,
        f"Native frame: {w}x{h}  |  Edit ROI/gate in vision/config.py",
        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
    )

    # Down-scale for display only — overlays stay faithful to the native frame.
    display_frame = cv2.resize(frame, TEST_DISPLAY_SIZE, interpolation=cv2.INTER_AREA)

    print("\n" + "="*60)
    print("  TEST MODE — First-frame overlay")
    print(f"  Native frame : {w} x {h}   Display : {TEST_DISPLAY_SIZE[0]} x {TEST_DISPLAY_SIZE[1]}")
    print(f"  Segment      : {segment}  |  Slot : {slot}")
    print("  The window shows your ROI polygon and counting gate.")
    print("  If they look wrong, close this window, edit vision/config.py,")
    print("  and re-run with --test.")
    print("  Press any key in the window to continue...")
    print("="*60 + "\n")

    cv2.imshow("TEST MODE — ROI & Gate Check (press any key)", display_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Rewind so the main loop starts from frame 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def run_pipeline(
    video_path:   Path,
    segment:      str,
    slot:         str,
    model_name:   str   = DEFAULT_MODEL,
    gate_ratio:   float | None = None,
    show_preview: bool  = False,
    use_roi:      bool  = True,
    test_mode:    bool  = False,
) -> tuple[Path, Path]:
    """
    Run the full Phase 1 pipeline for one video clip.

    Parameters
    ----------
    video_path   : absolute path to the .mp4 / .avi file
    segment      : "regional" | "valley"
    slot         : time-slot label — "10am", "1pm", "5pm"
    model_name   : YOLO weights file name (auto-downloaded if absent)
    gate_ratio   : override the gate Y-fraction from config (None = use config)
    show_preview : open an OpenCV display window during the full run
    use_roi      : apply the ROI polygon mask (disable for debugging)
    test_mode    : process only first 60 s + show first-frame overlay popup

    Returns
    -------
    (per_frame_csv_path, summary_csv_path)
    """
    # --- Validate inputs ---------------------------------------------------
    if not video_path.exists():
        sys.exit(
            f"CRITICAL: No video samples found in data/raw/video. "
            f"Please add your .mp4 clips (e.g., regional_10am.mp4) to begin."
        )

    if segment not in CALIBRATION:
        sys.exit(
            f"ERROR: Unknown segment '{segment}'. "
            f"Expected one of: {list(CALIBRATION.keys())}"
        )

    # --- Load per-video calibration (sidecar .yaml or global config.py) ----
    vc = load_video_calibration(video_path, segment)
    ppm     = vc.pixels_per_metre
    n_lanes = vc.lanes

    print(f"\n{'='*60}")
    print(f"  Malta Road Recovery — Phase 1 Vision Pipeline")
    print(f"{'='*60}")
    print(f"  Segment  : {segment.upper()}  ({n_lanes}-lane road)")
    print(f"  Slot     : {slot}")
    print(f"  Video    : {video_path.name}")
    describe_calib(vc)
    if test_mode:
        print(f"  Mode     : TEST — first {TEST_DURATION_S}s only")
    else:
        print(f"  Mode     : FULL RUN")
    print(f"{'='*60}\n")

    # --- Open video --------------------------------------------------------
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"CRITICAL: Cannot open video: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    raw_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Some DJI clips decode in landscape even though the footage is portrait.
    # Detect that here and rotate every frame 90° clockwise on read so the rest
    # of the pipeline (ROI, gate, PPM) sees the 1080×1920 portrait orientation
    # its coordinates were calibrated for.
    rotate_to_portrait = _needs_portrait_rotation(raw_w, raw_h)
    if rotate_to_portrait:
        frame_w, frame_h = raw_h, raw_w
        print(f"  Orientation : landscape {raw_w}×{raw_h} detected — "
              "auto-rotating 90° CW to portrait")
    else:
        frame_w, frame_h = raw_w, raw_h

    # In test mode, cap the number of frames to process
    max_frames = int(TEST_DURATION_S * fps) if test_mode else total_frames

    # Effective FPS after frame striding — used for speed calculation
    effective_fps = fps / FRAME_STRIDE

    print(f"  Resolution  : {frame_w} x {frame_h}  (portrait: width x height)")
    print(f"  FPS         : {fps:.1f}  |  Total frames : {total_frames}")
    print(f"  Frame stride: every {FRAME_STRIDE} frames  "
          f"({effective_fps:.1f} effective fps)")
    print(f"  Device      : {INFERENCE_DEVICE.upper()}  |  Inference size : {INFERENCE_IMGSZ}px")
    if test_mode:
        print(f"  Processing  : first {max_frames} frames (~{TEST_DURATION_S}s)")
    print()

    # --- Build ROI ---------------------------------------------------------
    roi_polygon = None
    roi_mask    = None

    if use_roi and vc.roi_polygon is not None:
        roi_polygon = vc.roi_polygon
        roi_mask    = build_roi_mask((frame_h, frame_w), roi_polygon)
        src = vc.source
        print(f"  ROI : {len(roi_polygon)}-point polygon active  [{src}]")
    else:
        print("  ROI : DISABLED")

    # --- Build dual-gate speed trap ---------------------------------------
    # Sidecar / global values are applied via keyword overrides;
    # --gate CLI flag further overrides the entry line if supplied.
    entry_ratio = gate_ratio if gate_ratio is not None else vc.gate_entry_ratio
    trap = make_speed_trap(
        segment, frame_h, frame_w,
        gate_entry_ratio = entry_ratio,
        gate_exit_ratio  = vc.gate_exit_ratio,
        gate_distance_m  = vc.gate_distance_m,
        roi_polygon      = roi_polygon,
    )

    print(f"  Gate entry : {trap.entry.y}px  ({trap.entry.y/frame_h:.0%})   "
          f"Gate exit : {trap.exit.y}px  ({trap.exit.y/frame_h:.0%})")
    print(f"  Gate dist  : {trap.gate_distance_m:.1f} m  (real-world)\n")

    # --- TEST MODE: show first-frame overlay before processing -------------
    if test_mode:
        _show_test_overlay(cap, trap, roi_polygon, segment, slot, rotate_to_portrait)

    # --- Load YOLO model ---------------------------------------------------
    print(f"  Loading YOLO model '{model_name}' on {INFERENCE_DEVICE.upper()}...")
    model = YOLO(model_name)
    print("  Model ready.\n")

    # --- Initialise pipeline components ------------------------------------
    # SpeedEstimator remains available for legacy per-frame speed logging,
    # but authoritative speed comes from the dual-gate trap below.
    estimator  = SpeedEstimator(ppm, effective_fps)
    aggregator = BinAggregator()

    per_frame_rows: list[dict] = []
    entry_logged:   set[int]   = set()

    # Per-vehicle state — one entry per unique track ID.
    # Populated on first sight; gate_speed_kph / exit_time filled in when the
    # SpeedTrap fires.  Flushed to a CSV at the end of the run.
    vehicles: dict[int, dict] = {}

    frame_idx     = 0   # absolute frame counter
    processed     = 0   # frames actually sent to YOLO

    # --- Frame loop --------------------------------------------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if rotate_to_portrait:
            frame = _to_portrait(frame)

        # Test mode: stop after the configured duration
        if test_mode and frame_idx >= max_frames:
            print(f"\n  [test] Reached {TEST_DURATION_S}s limit — stopping early.")
            break

        timestamp_s = frame_idx / fps

        # FRAME STRIDE — skip frames to save GPU time
        if frame_idx % FRAME_STRIDE != 0:
            frame_idx += 1
            continue

        processed += 1

        # Apply ROI mask to frame before YOLO inference
        inference_frame = (
            apply_roi_mask(frame, roi_mask) if roi_mask is not None else frame
        )

        # YOLO + ByteTrack — MPS GPU, fixed 640px inference size
        results = model.track(
            inference_frame,
            persist          = True,
            classes          = list(VEHICLE_CLASSES.keys()),
            tracker          = TRACKER_CFG,
            conf             = CONFIDENCE_THRESHOLD,
            imgsz            = INFERENCE_IMGSZ,
            device           = INFERENCE_DEVICE,
            verbose          = False,
        )

        if results and results[0].boxes is not None:
            boxes = results[0].boxes

            for box in boxes:
                if box.id is None:
                    continue

                track_id   = int(box.id.item())
                cls_id     = int(box.cls.item())
                class_name = VEHICLE_CLASSES.get(cls_id, "unknown")

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                # Secondary ROI check on the centroid
                if roi_polygon and not centroid_in_roi(x1, y1, x2, y2, roi_polygon):
                    continue

                speed = estimator.update(track_id, cx, cy)

                # Per-vehicle bookkeeping — counts every car once, whether
                # or not it ever crosses both gates.
                veh = vehicles.get(track_id)
                if veh is None:
                    veh = {
                        "id":             track_id,
                        "class":          class_name,
                        "entry_time_s":   round(timestamp_s, 3),
                        "exit_time_s":    None,
                        "gate_speed_kph": None,
                        "speed_samples":  [],
                        "frames_seen":    0,
                        "direction":      "",
                    }
                    vehicles[track_id] = veh
                    aggregator.add(timestamp_s, class_name, speed)
                else:
                    aggregator.touch(timestamp_s)

                veh["frames_seen"] += 1
                if speed is not None and speed > 0:
                    veh["speed_samples"].append(float(speed))

                # Populate speed_kph on EVERY per-frame row with the smoothed
                # per-frame estimate.  The gate-trap speed overrides this on
                # the frame it fires (below).
                row_speed = "" if speed is None else round(float(speed), 2)
                per_frame_rows.append({
                    "frame":      frame_idx,
                    "id":         track_id,
                    "class":      class_name,
                    "speed_kph":  row_speed,
                    "entry_time": round(timestamp_s, 3),
                })

                # Dual-gate speed-trap update (entry latch on top line,
                # speed computed on exit-line crossing).
                trap_event: TrapCrossing | None = trap.update(
                    track_id    = track_id,
                    cx          = cx,
                    cy          = cy,
                    timestamp_s = timestamp_s,
                    class_name  = class_name,
                    frame_idx   = frame_idx,
                )

                if trap_event is not None:
                    trap_speed = trap_event.speed_kph  # None if no entry seen
                    # Stamp the (more accurate) trap speed onto this frame's row.
                    if trap_speed is not None:
                        per_frame_rows[-1]["speed_kph"] = trap_speed
                    veh["exit_time_s"]    = round(trap_event.exit_time_s, 3)
                    veh["gate_speed_kph"] = trap_speed
                    veh["direction"]      = trap_event.direction
                    if trap_event.entry_time_s is not None:
                        veh["entry_time_s"] = round(trap_event.entry_time_s, 3)

                    if track_id not in entry_logged:
                        entry_logged.add(track_id)
                        speed_str = (
                            "N/A (no entry)" if trap_speed is None
                            else f"{trap_speed:>5.1f} km/h"
                        )
                        print(
                            f"  [trap] t={timestamp_s:>7.1f}s  "
                            f"ID={track_id:<4}  {class_name:<12}  "
                            f"{speed_str}  dir={trap_event.direction}"
                        )

                # Preview annotation (only shown during --preview or --test)
                if show_preview or test_mode:
                    cv2.rectangle(
                        frame,
                        (int(x1), int(y1)), (int(x2), int(y2)),
                        (255, 165, 0), 1,
                    )
                    lbl = f"{class_name} #{track_id}"
                    if speed:
                        lbl += f" {speed:.0f}km/h"
                    cv2.putText(
                        frame, lbl, (int(x1), max(int(y1) - 6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 165, 0), 1,
                    )

        # Live preview window (headless by default — only with --preview/--test)
        if show_preview or test_mode:
            trap.draw(frame)
            cv2.putText(
                frame,
                f"t={timestamp_s:.1f}s  seg={segment}  slot={slot}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1,
            )
            # Frame is already in native portrait orientation (rotated on read
            # if the source was landscape); just down-scale for a readable window.
            display_frame = cv2.resize(frame, TEST_DISPLAY_SIZE, interpolation=cv2.INTER_AREA)
            cv2.imshow("Malta Road Recovery — Phase 1 Preview", display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("  [detect] Preview closed by user.")
                break

        frame_idx += 1

        # Progress update every 500 processed frames
        if processed % 500 == 0:
            pct = 100 * frame_idx / max(max_frames, 1)
            elapsed_s = frame_idx / fps
            print(f"  ... {frame_idx}/{max_frames} frames  "
                  f"({pct:.0f}%)  t={elapsed_s:.0f}s")

    cap.release()
    if show_preview or test_mode:
        cv2.destroyAllWindows()

    aggregator.finalise(frame_idx / fps)

    # --- Write per-frame CSV (all rows buffered in memory, one write) ------
    frames_path = OUTPUT_DIR / f"{segment}_{slot}_frames.csv"
    df_frames   = pd.DataFrame(per_frame_rows, columns=PER_FRAME_COLUMNS)
    df_frames.to_csv(frames_path, index=False)

    print(f"\n  Per-frame CSV : {frames_path}")
    print(f"  {len(df_frames)} rows  |  {df_frames['id'].nunique()} unique track IDs")

    # --- Write per-vehicle CSV (1 row per detected car) -------------------
    vehicle_rows: list[dict] = []
    for veh in vehicles.values():
        samples = veh["speed_samples"]
        avg_spd = round(float(np.mean(samples)), 2) if samples else ""
        max_spd = round(float(np.max(samples)),  2) if samples else ""
        exit_t  = veh["exit_time_s"]
        duration = (
            round(exit_t - veh["entry_time_s"], 3)
            if exit_t is not None else ""
        )
        vehicle_rows.append({
            "id":              veh["id"],
            "class":           veh["class"],
            "entry_time_s":    veh["entry_time_s"],
            "exit_time_s":     exit_t if exit_t is not None else "",
            "duration_s":      duration,
            "gate_speed_kph":  veh["gate_speed_kph"] if veh["gate_speed_kph"] is not None else "",
            "avg_speed_kph":   avg_spd,
            "max_speed_kph":   max_spd,
            "frames_seen":     veh["frames_seen"],
            "direction":       veh["direction"],
        })

    vehicles_path = OUTPUT_DIR / f"{segment}_{slot}_vehicles.csv"
    df_vehicles   = pd.DataFrame(vehicle_rows, columns=PER_VEHICLE_COLUMNS)
    df_vehicles   = df_vehicles.sort_values("entry_time_s", kind="stable")
    df_vehicles.to_csv(vehicles_path, index=False)
    print(f"  Vehicles CSV  : {vehicles_path}")
    print(f"  {len(df_vehicles)} unique vehicle(s)  "
          f"|  {int((df_vehicles['gate_speed_kph'] != '').sum())} with gate speed")

    # --- Write summary CSV (one write at the end) --------------------------
    summary_path = OUTPUT_DIR / f"{segment}_{slot}.csv"
    df_summary   = pd.DataFrame(aggregator.rows)

    if not df_summary.empty:
        df_summary.insert(0, "segment",  segment)
        df_summary.insert(1, "slot",     slot)
        df_summary.insert(2, "n_lanes",  n_lanes)

    df_summary.to_csv(summary_path, index=False)
    print(f"  Summary CSV   : {summary_path}")
    print(f"  {len(df_summary)} 1-minute bin(s)\n")

    if not df_summary.empty:
        print(f"  Peak gate flow : {df_summary['vehicle_count'].max()} veh/min")
        print(f"  Mean speed     : {df_summary['avg_speed_kmh'].mean():.1f} km/h")
        print(f"  Total counted  : {df_summary['vehicle_count'].sum()} vehicles")

    return frames_path, summary_path


# ---------------------------------------------------------------------------
# CLI — single-clip entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Malta Road Recovery — Phase 1: detect & track one video clip.\n\n"
            "BEGINNERS: start with --test to check your ROI and gate position\n"
            "before running the full 10-minute clip."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check ROI/gate on first frame (recommended first step):
  python vision/detect.py --video data/raw/video/regional_10am.mp4 \\
      --segment regional --slot 10am --test

  # Full run (no window, fast):
  python vision/detect.py --video data/raw/video/regional_10am.mp4 \\
      --segment regional --slot 10am

  # Full run with live preview (slower, good for verification):
  python vision/detect.py --video data/raw/video/regional_10am.mp4 \\
      --segment regional --slot 10am --preview
        """,
    )
    p.add_argument(
        "--video", required=True, type=Path,
        help="Path to your video file, e.g. data/raw/video/regional_10am.mp4",
    )
    p.add_argument(
        "--segment", required=True, choices=list(CALIBRATION.keys()),
        help="Which road: regional | valley",
    )
    p.add_argument(
        "--slot", required=True,
        help="Time label for this clip: 10am | 1pm | 5pm (or any text)",
    )
    p.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"YOLO weights to use (default: {DEFAULT_MODEL}). "
             "Try yolov8s.pt for better accuracy at a speed cost.",
    )
    p.add_argument(
        "--gate", default=None, type=float,
        help="Gate Y as a fraction of frame height (0.0–1.0). "
             "Overrides the value in config.py.  Example: 0.55",
    )
    p.add_argument(
        "--preview", action="store_true",
        help="Show a live preview window during the full run "
             "(slower — only use for verification).",
    )
    p.add_argument(
        "--no-roi", action="store_true",
        help="Disable ROI masking.  Useful if you want to see all detections "
             "before deciding where to draw your polygon.",
    )
    p.add_argument(
        "--test", action="store_true",
        help="TEST MODE: process only the first 60 seconds and pop up a "
             "window showing the ROI + gate on the first frame.  "
             "Use this to check your config.py coordinates quickly.",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    run_pipeline(
        video_path   = args.video,
        segment      = args.segment,
        slot         = args.slot,
        model_name   = args.model,
        gate_ratio   = args.gate,
        show_preview = args.preview,
        use_roi      = not args.no_roi,
        test_mode    = args.test,
    )


if __name__ == "__main__":
    main()
