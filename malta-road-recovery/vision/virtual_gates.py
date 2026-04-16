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

def make_gate(
    segment: str,
    frame_height: int,
    frame_width: int,
    gate_name: str = "primary",
) -> VirtualGate:
    """
    Construct a VirtualGate using the calibration values in config.py.

    Parameters
    ----------
    segment      : "regional" | "valley"
    frame_height : pixel height of the video frame
    frame_width  : pixel width of the video frame
    gate_name    : "primary" or "secondary"

    Returns
    -------
    VirtualGate instance, ROI-aware if ROI_POLYGONS is not None.

    Raises
    ------
    ValueError if segment or gate_name are not found in config.
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

    colours = {
        "primary":   (0, 255, 0),
        "secondary": (0, 180, 255),
    }

    return VirtualGate(
        y_position  = y_px,
        frame_width = frame_width,
        label       = f"{segment} — {gate_name}",
        colour      = colours.get(gate_name, (255, 255, 255)),
        roi_polygon = roi,
    )


# ---------------------------------------------------------------------------
# CLI — calibration preview
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preview gate and ROI positions on a still frame"
    )
    parser.add_argument("--frame",   required=True, type=Path,
                        help="Path to a still JPEG/PNG captured from your video")
    parser.add_argument("--segment", required=True,
                        choices=list(GATE_Y_RATIOS.keys()),
                        help="Road segment: regional | valley")
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

    for gate_name in GATE_Y_RATIOS.get(args.segment, {}):
        gate = make_gate(args.segment, h, w, gate_name)
        gate.draw(img)
        y_frac = GATE_Y_RATIOS[args.segment][gate_name]
        print(f"  Gate '{gate_name}': y={gate.y}px  ({y_frac:.0%} of frame height)")

    if ROI_POLYGONS and args.segment in ROI_POLYGONS:
        print(f"  ROI polygon: {ROI_POLYGONS[args.segment]}")
    else:
        print("  ROI polygon: DISABLED (set ROI_POLYGONS in config.py)")

    out = args.output or args.frame.parent / f"{args.frame.stem}_annotated{args.frame.suffix}"
    cv2.imwrite(str(out), img)
    print(f"\nAnnotated image saved → {out}")

    cv2.imshow("Gate + ROI calibration  (press any key to close)", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
