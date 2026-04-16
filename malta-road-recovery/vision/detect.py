"""
Malta Road Recovery Predictor — Phase 1: YOLO Detection & Tracking
===================================================================
Segment-aware vehicle detection, tracking, and speed estimation.
Outputs: data/processed/yolo_output/<segment>_<slot>.csv

Requirements:
    pip install ultralytics opencv-python-headless pandas numpy

Usage:
    python vision/detect.py --video data/raw/video/regional_10am.mp4 \
                             --segment regional \
                             --slot 10am
"""

import argparse
import csv
import sys
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# PROJECT ROOT  (vision/detect.py → project root is one level up)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR    = PROJECT_ROOT / "data" / "raw" / "video"
OUTPUT_DIR   = PROJECT_ROOT / "data" / "processed" / "yolo_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# PIXEL-TO-METRE CALIBRATION CONSTANTS
# ---------------------------------------------------------------------------
# Replace these with values measured from your actual camera setup.
# Method: place a known-length reference object (e.g. a 4 m road marking)
# in the video frame and count the pixels it spans.
#
#   PIXELS_PER_METRE = pixel_length_of_reference / real_length_in_metres
#
# These are *placeholder* values — one per road type to account for
# different camera heights / angles.

CALIBRATION = {
    "regional": {
        "pixels_per_metre": 8.5,   # PLACEHOLDER — calibrate from footage
        "lanes": 2,
        "approx_road_width_m": 7.0,
    },
    "valley": {
        "pixels_per_metre": 10.2,  # PLACEHOLDER — calibrate from footage
        "lanes": 1,
        "approx_road_width_m": 3.5,
    },
}

# YOLO class IDs we care about (COCO dataset)
VEHICLE_CLASSES = {
    2:  "car",
    3:  "motorcycle",
    5:  "bus",
    7:  "truck",
}

# Aggregation window in seconds
BIN_SECONDS = 60


# ---------------------------------------------------------------------------
# VIRTUAL GATE
# ---------------------------------------------------------------------------

class VirtualGate:
    """
    A horizontal line across the frame. A vehicle is counted when its
    tracked centroid crosses from one side of the line to the other.

    Args:
        y_position  : pixel row of the gate line
        frame_width : width of the video frame in pixels
        label       : human-readable name (e.g. "entry" or "exit")
    """

    def __init__(self, y_position: int, frame_width: int, label: str = "gate"):
        self.y = y_position
        self.width = frame_width
        self.label = label
        self._prev_y: dict[int, float] = {}   # track_id → last centroid y

    def update(self, track_id: int, cx: float, cy: float) -> bool:
        """
        Returns True if the vehicle with `track_id` just crossed this gate.
        """
        crossed = False
        if track_id in self._prev_y:
            prev = self._prev_y[track_id]
            # Upward crossing (entering frame from bottom)
            if prev > self.y >= cy:
                crossed = True
            # Downward crossing (entering frame from top)
            elif prev < self.y <= cy:
                crossed = True
        self._prev_y[track_id] = cy
        return crossed

    def draw(self, frame: np.ndarray) -> np.ndarray:
        cv2.line(frame, (0, self.y), (self.width, self.y), (0, 255, 0), 2)
        cv2.putText(frame, self.label, (10, self.y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
        return frame


# ---------------------------------------------------------------------------
# SPEED ESTIMATOR
# ---------------------------------------------------------------------------

class SpeedEstimator:
    """
    Estimates instantaneous speed in km/h for a tracked vehicle.

    Uses two consecutive centroid positions and the frame interval to
    derive pixel displacement, then converts to metres via calibration.

    Args:
        pixels_per_metre : calibration constant for this road segment
        fps              : video frames per second
    """

    def __init__(self, pixels_per_metre: float, fps: float):
        self.ppm = pixels_per_metre
        self.fps = fps
        self._prev: dict[int, tuple[float, float]] = {}   # id → (cx, cy)

    def update(self, track_id: int, cx: float, cy: float) -> float | None:
        """
        Returns speed in km/h, or None if this is the first observation.
        """
        speed_kmh = None
        if track_id in self._prev:
            px, py = self._prev[track_id]
            pixel_dist = np.hypot(cx - px, cy - py)
            metres     = pixel_dist / self.ppm
            seconds    = 1.0 / self.fps
            speed_kmh  = (metres / seconds) * 3.6   # m/s → km/h
        self._prev[track_id] = (cx, cy)
        return speed_kmh


# ---------------------------------------------------------------------------
# BIN AGGREGATOR
# ---------------------------------------------------------------------------

class BinAggregator:
    """
    Accumulates per-frame detections and flushes them into 1-minute CSV rows.
    """

    def __init__(self, bin_seconds: int = BIN_SECONDS):
        self.bin_seconds = bin_seconds
        self._current_bin: int = -1
        self._counts: dict[str, int] = defaultdict(int)
        self._speeds: list[float] = []
        self.rows: list[dict] = []

    def _bin_index(self, timestamp_s: float) -> int:
        return int(timestamp_s // self.bin_seconds)

    def add(self, timestamp_s: float, class_name: str, speed_kmh: float | None):
        bin_idx = self._bin_index(timestamp_s)

        if bin_idx != self._current_bin and self._current_bin >= 0:
            self._flush(self._current_bin)

        self._current_bin = bin_idx
        self._counts[class_name] += 1
        if speed_kmh is not None and speed_kmh > 0:
            self._speeds.append(speed_kmh)

    def _flush(self, bin_idx: int):
        ts = bin_idx * self.bin_seconds
        total = sum(self._counts.values())
        avg_spd = float(np.mean(self._speeds)) if self._speeds else 0.0

        self.rows.append({
            "timestamp_s":    ts,
            "timestamp_min":  round(ts / 60, 2),
            "vehicle_count":  total,
            "avg_speed_kmh":  round(avg_spd, 2),
            **{f"count_{k}": self._counts.get(k, 0) for k in VEHICLE_CLASSES.values()},
        })
        self._counts = defaultdict(int)
        self._speeds = []

    def finalise(self, last_timestamp_s: float):
        if self._current_bin >= 0:
            self._flush(self._current_bin)


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def validate_video_path(video_path: Path) -> None:
    """
    Raises SystemExit with a descriptive CRITICAL error if the video is missing.
    """
    if not video_path.exists():
        candidates = list(VIDEO_DIR.glob("*.mp4")) + list(VIDEO_DIR.glob("*.avi"))
        if not candidates:
            sys.exit(
                "CRITICAL: Video samples not found in data/raw/video. "
                "Please provide 15-minute clips to proceed with ground-truth calibration."
            )
        else:
            sys.exit(
                f"CRITICAL: Requested file '{video_path.name}' not found in "
                f"data/raw/video. Available files: "
                + ", ".join(p.name for p in candidates)
            )


def run_pipeline(
    video_path: Path,
    segment: str,
    slot: str,
    model_name: str = "yolov8n.pt",
    gate_ratio: float = 0.55,
    show_preview: bool = False,
) -> Path:
    """
    Full Phase 1 pipeline for one video clip.

    Args:
        video_path  : path to the .mp4 file
        segment     : "regional" or "valley"
        slot        : time-slot label, e.g. "10am", "1pm", "5pm"
        model_name  : YOLO weights to use
        gate_ratio  : vertical position of gate as fraction of frame height
        show_preview: open a live preview window (useful for gate calibration)

    Returns:
        Path to the output CSV.
    """
    validate_video_path(video_path)

    if segment not in CALIBRATION:
        sys.exit(f"ERROR: Unknown segment '{segment}'. Choose: {list(CALIBRATION)}")

    calib    = CALIBRATION[segment]
    ppm      = calib["pixels_per_metre"]
    n_lanes  = calib["lanes"]

    print(f"\n[Malta YOLO] Segment : {segment.upper()} ({n_lanes}-lane)")
    print(f"[Malta YOLO] Slot    : {slot}")
    print(f"[Malta YOLO] Video   : {video_path.name}")
    print(f"[Malta YOLO] Cal PPM : {ppm} px/m  (placeholder — update after filming)\n")

    # --- Video capture ---
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"CRITICAL: Cannot open video file: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[Malta YOLO] Resolution : {frame_w}×{frame_h}  |  FPS : {fps:.1f}  |  Frames : {total_frames}")

    # --- Components ---
    model      = YOLO(model_name)
    gate_y     = int(frame_h * gate_ratio)
    gate       = VirtualGate(gate_y, frame_w, label=f"{segment} gate")
    estimator  = SpeedEstimator(ppm, fps)
    aggregator = BinAggregator()

    # Per-track speed buffer (rolling mean over last N frames)
    track_speeds: dict[int, list[float]] = defaultdict(list)
    SPEED_BUFFER = 10

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_s = frame_idx / fps

        # --- YOLO inference with ByteTrack ---
        results = model.track(
            frame,
            persist=True,
            classes=list(VEHICLE_CLASSES.keys()),
            tracker="bytetrack.yaml",
            verbose=False,
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
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                # Speed estimation
                raw_speed = estimator.update(track_id, cx, cy)
                if raw_speed is not None:
                    track_speeds[track_id].append(raw_speed)
                    if len(track_speeds[track_id]) > SPEED_BUFFER:
                        track_speeds[track_id].pop(0)

                smoothed_speed = (
                    float(np.mean(track_speeds[track_id]))
                    if track_speeds[track_id] else None
                )

                # Gate crossing → count + log
                if gate.update(track_id, cx, cy):
                    aggregator.add(timestamp_s, class_name, smoothed_speed)

                # Annotate frame
                if show_preview:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                  (255, 165, 0), 1)
                    label = f"{class_name} #{track_id}"
                    if smoothed_speed:
                        label += f" {smoothed_speed:.0f}km/h"
                    cv2.putText(frame, label, (int(x1), int(y1) - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 165, 0), 1)

        if show_preview:
            gate.draw(frame)
            cv2.putText(frame, f"t={timestamp_s:.1f}s  seg={segment}  slot={slot}",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            cv2.imshow("Malta YOLO Preview", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[Malta YOLO] Preview closed by user.")
                break

        frame_idx += 1
        if frame_idx % 500 == 0:
            pct = 100 * frame_idx / max(total_frames, 1)
            print(f"  ... {frame_idx}/{total_frames} frames ({pct:.0f}%)")

    cap.release()
    if show_preview:
        cv2.destroyAllWindows()

    aggregator.finalise(frame_idx / fps)

    # --- Export CSV ---
    output_path = OUTPUT_DIR / f"{segment}_{slot}.csv"
    df = pd.DataFrame(aggregator.rows)

    # Add metadata columns useful for SUMO calibration
    df.insert(0, "segment",  segment)
    df.insert(1, "slot",     slot)
    df.insert(2, "n_lanes",  n_lanes)

    df.to_csv(output_path, index=False)
    print(f"\n[Malta YOLO] Done. {len(df)} 1-minute bins written to:\n  {output_path}\n")

    # Quick summary
    if not df.empty:
        print(f"  Peak flow    : {df['vehicle_count'].max()} veh/min")
        print(f"  Mean speed   : {df['avg_speed_kmh'].mean():.1f} km/h")
        print(f"  Total vehicles counted: {df['vehicle_count'].sum()}")

    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Malta Road Recovery — Phase 1 YOLO pipeline"
    )
    parser.add_argument(
        "--video",    required=True,  type=Path,
        help="Path to input video, e.g. data/raw/video/regional_10am.mp4"
    )
    parser.add_argument(
        "--segment",  required=True,  choices=["regional", "valley"],
        help="Road segment type"
    )
    parser.add_argument(
        "--slot",     required=True,
        help="Time slot label: 10am | 1pm | 5pm"
    )
    parser.add_argument(
        "--model",    default="yolov8n.pt",
        help="YOLO weights file (default: yolov8n.pt)"
    )
    parser.add_argument(
        "--gate",     default=0.55, type=float,
        help="Gate Y position as fraction of frame height (default: 0.55)"
    )
    parser.add_argument(
        "--preview",  action="store_true",
        help="Show live OpenCV preview window"
    )

    args = parser.parse_args()
    run_pipeline(
        video_path   = args.video,
        segment      = args.segment,
        slot         = args.slot,
        model_name   = args.model,
        gate_ratio   = args.gate,
        show_preview = args.preview,
    )


if __name__ == "__main__":
    main()