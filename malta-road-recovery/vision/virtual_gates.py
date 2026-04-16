"""
virtual_gates.py
================
Standalone gate configuration for Regional Road and Valley Road.

Provides recommended gate Y-positions (as frame-height fractions) and
helper functions to visualise gate placement on a still frame — useful
for calibrating gate position before a full pipeline run.

Usage (calibration check):
    python vision/virtual_gates.py --frame path/to/still.jpg --segment regional
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# RECOMMENDED GATE CONFIGURATIONS
# ---------------------------------------------------------------------------
# Each segment can define multiple gates (e.g. entry + exit) for future
# density / travel-time estimation. Phase 1 uses only the primary gate.

GATE_CONFIGS = {
    "regional": {
        "primary": {
            "y_ratio": 0.55,          # 55 % down the frame
            "label":   "Regional Rd — primary gate",
            "colour":  (0, 255, 0),   # green
        },
        "secondary": {
            "y_ratio": 0.75,
            "label":   "Regional Rd — secondary gate",
            "colour":  (0, 180, 255), # amber
        },
    },
    "valley": {
        "primary": {
            "y_ratio": 0.50,
            "label":   "Valley Rd — primary gate",
            "colour":  (0, 255, 180),
        },
    },
}


def get_gate_y(segment: str, gate: str = "primary", frame_height: int = 720) -> int:
    """Return absolute pixel Y for the named gate on this segment."""
    cfg = GATE_CONFIGS.get(segment, {}).get(gate)
    if cfg is None:
        raise ValueError(f"No gate '{gate}' defined for segment '{segment}'")
    return int(frame_height * cfg["y_ratio"])


def draw_gates_on_frame(frame: np.ndarray, segment: str) -> np.ndarray:
    """Overlay all configured gates on a still frame (for calibration)."""
    h, w = frame.shape[:2]
    for gate_name, cfg in GATE_CONFIGS.get(segment, {}).items():
        y = int(h * cfg["y_ratio"])
        cv2.line(frame, (0, y), (w, y), cfg["colour"], 2)
        cv2.putText(frame, cfg["label"], (12, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, cfg["colour"], 1)
    return frame


# ---------------------------------------------------------------------------
# CLI — calibration preview
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualise gate positions on a still frame"
    )
    parser.add_argument("--frame",   required=True, type=Path,
                        help="Path to a still JPEG/PNG from your video")
    parser.add_argument("--segment", required=True,
                        choices=list(GATE_CONFIGS.keys()))
    args = parser.parse_args()

    if not args.frame.exists():
        sys.exit(f"File not found: {args.frame}")

    img = cv2.imread(str(args.frame))
    if img is None:
        sys.exit(f"Cannot read image: {args.frame}")

    img = draw_gates_on_frame(img, args.segment)
    out = args.frame.parent / f"{args.frame.stem}_gates{args.frame.suffix}"
    cv2.imwrite(str(out), img)
    print(f"Gate preview saved to: {out}")

    cv2.imshow("Gate calibration", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()