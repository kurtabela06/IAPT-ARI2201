"""
vision/virtual_gates.py
=======================
Virtual gate (counting line) logic for the Malta Road Recovery pipeline.

A VirtualGate is a horizontal line at a configurable Y position.
When a tracked vehicle centroid crosses from one side to the other,
the gate fires and returns a GateCrossing event.

The gate is ROI-aware: it only fires if the centroid is also inside
the configured ROI polygon (prevents detections leaking from adjacent lanes).

Public API
----------
VirtualGate
    .update(track_id, cx, cy)  → GateCrossing | None
    .draw(frame)               → frame

GateCrossing  (named-tuple)
    .track_id, .timestamp_s, .class_name, .speed_kph, .direction

Usage — calibration preview
----------------------------
    python vision/virtual_gates.py \
        --frame  calibration_still.jpg \
        --segment regional
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Import from sibling modules — works whether run as a script or imported
try:
    from vision.config import GATE_Y_RATIOS, ROI_POLYGONS
    from vision.utils import point_in_polygon, draw_roi
except ImportError:
    from config import GATE_Y_RATIOS, ROI_POLYGONS        # type: ignore
    from utils import point_in_polygon, draw_roi           # type: ignore


# ---------------------------------------------------------------------------
# DATA CLASS — gate crossing event
# ---------------------------------------------------------------------------

@dataclass
class GateCrossing:
    """
    Emitted once per vehicle per gate crossing.

    Fields
    ------
    track_id    : ByteTrack persistent ID
    timestamp_s : video time in seconds at the moment of crossing
    class_name  : "car" | "bus" | "truck" | "motorcycle"
    speed_kph   : smoothed speed estimate at crossing (None if unavailable)
    direction   : "downward" (top→bottom) | "upward" (bottom→top)
    frame_idx   : absolute frame number
    """
    track_id:    int
    timestamp_s: float
    class_name:  str
    speed_kph:   Optional[float]
    direction:   str
    frame_idx:   int


# ---------------------------------------------------------------------------
# VIRTUAL GATE
# ---------------------------------------------------------------------------

class VirtualGate:
    """
    Horizontal counting line.

    Parameters
    ----------
    y_position  : absolute pixel row for the gate line
    frame_width : used only for drawing — does not affect detection
    label       : human-readable name shown on preview frames
    colour      : BGR colour for the drawn line
    roi_polygon : if provided, only crossings with centroid inside the polygon
                  are reported; pass None to disable ROI filtering
    """

    def __init__(
        self,
        y_position: int,
        frame_width: int,
        label: str = "gate",
        colour: tuple[int, int, int] = (0, 255, 0),
        roi_polygon: list[tuple[int, int]] | None = None,
    ) -> None:
        self.y           = y_position
        self.width       = frame_width
        self.label       = label
        self.colour      = colour
        self.roi_polygon = roi_polygon

        # track_id → previous centroid Y
        self._prev_y: dict[int, float] = {}

    # ------------------------------------------------------------------
    # Core detection
    # ------------------------------------------------------------------

    def update(
        self,
        track_id:    int,
        cx:          float,
        cy:          float,
        timestamp_s: float = 0.0,
        class_name:  str   = "car",
        speed_kph:   float | None = None,
        frame_idx:   int   = 0,
    ) -> GateCrossing | None:
        """
        Call once per tracked box per frame.

        Returns a GateCrossing if this update constitutes a crossing,
        otherwise returns None.

        Parameters
        ----------
        track_id    : persistent ByteTrack ID
        cx, cy      : centroid of the bounding box in pixel coordinates
        timestamp_s : current video timestamp (seconds)
        class_name  : YOLO class label string
        speed_kph   : smoothed speed at this frame (may be None)
        frame_idx   : current frame number (for the CSV)
        """
        # ROI filter — ignore centroids outside the road polygon
        if self.roi_polygon is not None:
            if not point_in_polygon(cx, cy, self.roi_polygon):
                self._prev_y[track_id] = cy   # keep tracking position
                return None

        crossing = None

        if track_id in self._prev_y:
            prev = self._prev_y[track_id]

            # Downward crossing  (prev above gate, now at or below)
            if prev < self.y <= cy:
                crossing = GateCrossing(
                    track_id    = track_id,
                    timestamp_s = timestamp_s,
                    class_name  = class_name,
                    speed_kph   = speed_kph,
                    direction   = "downward",
                    frame_idx   = frame_idx,
                )
            # Upward crossing  (prev below gate, now at or above)
            elif prev > self.y >= cy:
                crossing = GateCrossing(
                    track_id    = track_id,
                    timestamp_s = timestamp_s,
                    class_name  = class_name,
                    speed_kph   = speed_kph,
                    direction   = "upward",
                    frame_idx   = frame_idx,
                )

        self._prev_y[track_id] = cy
        return crossing

    def forget(self, track_id: int) -> None:
        """Remove stale track IDs to keep the internal dict lean."""
        self._prev_y.pop(track_id, None)

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """
        Overlay the gate line and label on `frame` (in-place).
        Also draws the ROI polygon outline if one is configured.
        """
        h, w = frame.shape[:2]

        # Gate line — spans full frame width
        cv2.line(frame, (0, self.y), (w, self.y), self.colour, 2)
        cv2.putText(
            frame, self.label,
            (12, self.y - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.colour, 1, cv2.LINE_AA,
        )

        # ROI polygon outline
        if self.roi_polygon:
            draw_roi(frame, self.roi_polygon, colour=(0, 200, 255), label="ROI")

        return frame


# ---------------------------------------------------------------------------
# FACTORY — build gates from config.py values
# ---------------------------------------------------------------------------

# BGR colours shared by the preview and speed-trap drawing code
GATE_COLOURS: dict[str, tuple[int, int, int]] = {
    "gate_entry": (255, 255, 0),   # CYAN (BGR)
    "gate_exit":  (0, 255, 0),     # GREEN (BGR)
}


def make_gate(
    segment: str,
    frame_height: int,
    frame_width: int,
    gate_name: str = "gate_entry",
) -> VirtualGate:
    """
    Construct a single VirtualGate from config.py.

    Gate Y ratios are expressed against the NATIVE post-rotation frame height,
    so no SCALING_FACTOR_Y is applied here — the caller passes frame_height
    for the already-oriented 1080×1920 frame.

    Parameters
    ----------
    segment      : "regional" | "valley"
    frame_height : pixel height of the video frame
    frame_width  : pixel width of the video frame
    gate_name    : "gate_entry" | "gate_exit"
    """
    seg_gates = GATE_Y_RATIOS.get(segment)
    if seg_gates is None:
        raise ValueError(
            f"No gate config for segment '{segment}'. "
            f"Available: {list(GATE_Y_RATIOS.keys())}"
        )
    ratio = seg_gates.get(gate_name)
    if ratio is None:
        raise ValueError(
            f"No '{gate_name}' gate for segment '{segment}'. "
            f"Available: {list(seg_gates.keys())}"
        )

    y_px = int(frame_height * ratio)

    roi = None
    if ROI_POLYGONS is not None:
        roi = ROI_POLYGONS.get(segment)

    return VirtualGate(
        y_position  = y_px,
        frame_width = frame_width,
        label       = f"{segment} — {gate_name}",
        colour      = GATE_COLOURS.get(gate_name, (255, 255, 255)),
        roi_polygon = roi,
    )


# ---------------------------------------------------------------------------
# DUAL-GATE SPEED TRAP
# ---------------------------------------------------------------------------

@dataclass
class TrapCrossing:
    """Emitted when a tracked vehicle completes an entry → exit traversal.

    speed_kph is None when the vehicle entered before tracking began (no
    recorded entry_time), or when the measured dt is non-positive — we
    still emit the event so the vehicle is counted, but skip the speed.
    """
    track_id:       int
    entry_time_s:   Optional[float]
    exit_time_s:    float
    class_name:     str
    speed_kph:      Optional[float]
    direction:      str
    frame_idx:      int


class SpeedTrap:
    """
    Pair of virtual gates that together measure time-of-flight speed.

    A vehicle's entry_time_s is recorded when it first crosses the entry
    gate.  When the same track_id crosses the exit gate, the speed is
    computed from the measured real-world gate distance and the elapsed
    time.  If the exit crossing arrives with no recorded entry (vehicle
    appeared mid-frame past the entry gate), speed is reported as None
    to prevent infinite/invalid speed readings.
    """

    def __init__(
        self,
        entry_gate: VirtualGate,
        exit_gate:  VirtualGate,
        gate_distance_m: float,
    ) -> None:
        if gate_distance_m <= 0:
            raise ValueError("gate_distance_m must be > 0")
        self.entry = entry_gate
        self.exit  = exit_gate
        self.gate_distance_m = gate_distance_m
        # track_id → entry timestamp (seconds).  Missing key = no entry seen.
        self._entry_times: dict[int, float] = {}

    def update(
        self,
        track_id:    int,
        cx:          float,
        cy:          float,
        timestamp_s: float,
        class_name:  str,
        frame_idx:   int,
    ) -> TrapCrossing | None:
        """
        Call once per tracked box per frame.  Returns a TrapCrossing only
        when the vehicle crosses the EXIT gate; None otherwise (including
        when it merely crosses the entry gate — that just latches the
        entry timestamp).
        """
        # Entry-gate crossing: latch the timestamp, no event emitted.
        entry_cross = self.entry.update(
            track_id=track_id, cx=cx, cy=cy,
            timestamp_s=timestamp_s, class_name=class_name,
            frame_idx=frame_idx,
        )
        if entry_cross is not None and track_id not in self._entry_times:
            self._entry_times[track_id] = timestamp_s

        # Exit-gate crossing: compute speed if we saw the entry.
        exit_cross = self.exit.update(
            track_id=track_id, cx=cx, cy=cy,
            timestamp_s=timestamp_s, class_name=class_name,
            frame_idx=frame_idx,
        )
        if exit_cross is None:
            return None

        entry_ts = self._entry_times.pop(track_id, None)
        speed_kph: float | None = None
        if entry_ts is not None:
            dt = timestamp_s - entry_ts
            if dt > 0:
                speed_mps = self.gate_distance_m / dt
                speed_kph = round(speed_mps * 3.6, 2)

        return TrapCrossing(
            track_id     = track_id,
            entry_time_s = entry_ts,
            exit_time_s  = timestamp_s,
            class_name   = class_name,
            speed_kph    = speed_kph,
            direction    = exit_cross.direction,
            frame_idx    = frame_idx,
        )

    def forget(self, track_id: int) -> None:
        self.entry.forget(track_id)
        self.exit.forget(track_id)
        self._entry_times.pop(track_id, None)

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Draw entry (cyan), exit (green), and the shared ROI polygon."""
        self.entry.draw(frame)
        # exit.draw() would re-draw the ROI; call just the gate parts
        w = frame.shape[1]
        cv2.line(frame, (0, self.exit.y), (w, self.exit.y), self.exit.colour, 2)
        cv2.putText(
            frame, self.exit.label,
            (12, self.exit.y - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.exit.colour, 1, cv2.LINE_AA,
        )
        return frame


def make_speed_trap(
    segment: str,
    frame_height: int,
    frame_width: int,
    *,
    gate_entry_ratio:  float | None = None,
    gate_exit_ratio:   float | None = None,
    gate_distance_m:   float | None = None,
    roi_polygon:       list[tuple[int, int]] | None = None,
) -> SpeedTrap:
    """
    Build a dual-gate speed trap.

    Uses config.py defaults unless keyword overrides are supplied (e.g. from
    a per-video sidecar loaded by video_config.load_video_calibration).

    Parameters
    ----------
    segment           : "regional" | "valley"
    frame_height      : native frame height in px
    frame_width       : native frame width in px
    gate_entry_ratio  : Y fraction override for the entry line (0.0–1.0)
    gate_exit_ratio   : Y fraction override for the exit line (0.0–1.0)
    gate_distance_m   : real-world distance override (metres)
    roi_polygon       : native-frame ROI polygon override; if None the
                        segment default from config.py is used
    """
    seg_gates = GATE_Y_RATIOS.get(segment)
    if seg_gates is None:
        raise ValueError(f"No gate config for segment '{segment}'")

    entry_ratio = gate_entry_ratio if gate_entry_ratio is not None else seg_gates["gate_entry"]
    exit_ratio  = gate_exit_ratio  if gate_exit_ratio  is not None else seg_gates["gate_exit"]
    distance    = gate_distance_m  if gate_distance_m  is not None else seg_gates.get("gate_distance_m")
    if distance is None:
        raise ValueError(f"Segment '{segment}' is missing gate_distance_m in config.py")

    # ROI: use the supplied override, then config default, then None
    if roi_polygon is None and ROI_POLYGONS is not None:
        roi_polygon = ROI_POLYGONS.get(segment)

    entry_gate = VirtualGate(
        y_position  = int(frame_height * entry_ratio),
        frame_width = frame_width,
        label       = f"{segment} — gate_entry",
        colour      = GATE_COLOURS["gate_entry"],
        roi_polygon = roi_polygon,
    )
    exit_gate = VirtualGate(
        y_position  = int(frame_height * exit_ratio),
        frame_width = frame_width,
        label       = f"{segment} — gate_exit",
        colour      = GATE_COLOURS["gate_exit"],
        roi_polygon = roi_polygon,
    )
    return SpeedTrap(entry_gate, exit_gate, float(distance))


# ---------------------------------------------------------------------------
# CLI — calibration preview
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Preview gate and ROI positions on a still frame.\n\n"
            "If --video is given, the matching sidecar .yaml is loaded and its\n"
            "calibration values are used instead of the config.py defaults."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--frame",   required=True,  type=Path,
                        help="Path to a still JPEG/PNG captured from your video")
    parser.add_argument("--segment", required=True,
                        choices=list(GATE_Y_RATIOS.keys()),
                        help="Road segment: regional | valley")
    parser.add_argument("--video",   default=None, type=Path,
                        help="Path to the video this still came from — enables "
                             "sidecar .yaml loading (e.g. data/raw/video/regional_10am.MP4)")
    parser.add_argument("--output",  type=Path, default=None,
                        help="Save annotated image to this path (optional)")
    args = parser.parse_args()

    if not args.frame.exists():
        sys.exit(f"File not found: {args.frame}")

    img = cv2.imread(str(args.frame))
    if img is None:
        sys.exit(f"Cannot read image: {args.frame}")

    h, w = img.shape[:2]
    print(f"Frame size: {w}×{h}")

    # --- Load calibration (sidecar if available, else global defaults) ------
    if args.video is not None:
        # Import here to avoid a circular import at module level
        try:
            from vision.video_config import load_video_calibration, describe
        except ImportError:
            from video_config import load_video_calibration, describe   # type: ignore

        vc = load_video_calibration(args.video, args.segment)
        describe(vc)
        trap = make_speed_trap(
            args.segment, h, w,
            gate_entry_ratio = vc.gate_entry_ratio,
            gate_exit_ratio  = vc.gate_exit_ratio,
            gate_distance_m  = vc.gate_distance_m,
            roi_polygon      = vc.roi_polygon,
        )
        roi_polygon = vc.roi_polygon
    else:
        print("  (No --video supplied; using global config.py defaults)")
        trap = make_speed_trap(args.segment, h, w)
        roi_polygon = ROI_POLYGONS.get(args.segment) if ROI_POLYGONS else None

    # --- Draw ROI and both gate lines --------------------------------------
    if roi_polygon:
        try:
            from vision.utils import draw_roi
        except ImportError:
            from utils import draw_roi                                   # type: ignore
        draw_roi(img, roi_polygon, label="ROI")

    trap.draw(img)
    print(f"  Gate entry : y={trap.entry.y}px  ({trap.entry.y/h:.0%})")
    print(f"  Gate exit  : y={trap.exit.y}px   ({trap.exit.y/h:.0%})")
    print(f"  Distance   : {trap.gate_distance_m:.1f} m")

    out = args.output or args.frame.parent / f"{args.frame.stem}_annotated{args.frame.suffix}"
    cv2.imwrite(str(out), img)
    print(f"\nAnnotated image saved → {out}")

    cv2.imshow("Gate + ROI calibration  (press any key to close)", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
