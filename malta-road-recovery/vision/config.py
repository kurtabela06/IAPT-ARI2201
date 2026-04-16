"""
vision/config.py
================
Single source-of-truth for every calibration constant in Phase 1.

HOW TO CALIBRATE BEFORE YOUR FIRST RUN
---------------------------------------
Step 1 — Capture a still frame from each clip:
    python -c "
    import cv2, sys
    cap = cv2.VideoCapture(sys.argv[1])
    cap.set(cv2.CAP_PROP_POS_FRAMES, 300)   # frame 300 ≈ 10 s in
    _, f = cap.read()
    cv2.imwrite('calibration_still.jpg', f)
    cap.release()
    " data/raw/video/regional_10am.mp4

Step 2 — Open calibration_still.jpg in any image viewer that reports pixel
    coordinates (e.g. GIMP, Preview, Paint).  Find a road feature whose
    real-world length you know (lane width ≈ 3.5 m, road marking ≈ 3 m, etc.)
    and note the pixel distance between its two ends.

    PIXELS_PER_METRE = <pixel_distance> / <real_distance_in_metres>

    Update the value for the relevant segment below.

Step 3 — ROI Polygon (Region of Interest)
    In the same still image, click the four (or more) corners that enclose
    the road you want to analyse.  Note each (x, y) pixel coordinate.
    Replace the placeholder tuples in ROI_POLYGONS below.

    The polygon must be in ORDER (clockwise or anti-clockwise) and
    must fully enclose the road lanes you care about.

Step 4 — Virtual Gate
    Pick a horizontal Y pixel position that lies across the lane(s) inside
    your ROI.  Vehicles are counted each time their tracked centroid crosses
    this line.  Update GATE_Y_RATIOS below (value is 0.0-1.0 fraction of the
    frame height, so it scales automatically to any resolution).

Step 5 — Re-run main.py with --preview to visually confirm.
"""

# ---------------------------------------------------------------------------
# SPEED CALIBRATION
# ---------------------------------------------------------------------------
# One entry per road segment.
# pixels_per_metre : MEASURED from footage (see Step 2 above).
#                    The placeholder values below are deliberately conservative
#                    estimates — replace them with your measurements.

CALIBRATION: dict[str, dict] = {
    "regional": {
        # Regional Road — Twin-Carriageway / Tunnel Entrance section
        # Approx. 2 lanes, lane width ~3.5 m each.
        # PLACEHOLDER: measure from footage and update.
        "pixels_per_metre": 8.5,   # <-- REPLACE after Step 2
        "lanes": 2,
        "approx_road_width_m": 7.0,
    },
    "valley": {
        # Valley Road — Single-Lane section
        # Approx. 1 lane, road width ~3.5 m.
        # PLACEHOLDER: measure from footage and update.
        "pixels_per_metre": 10.2,  # <-- REPLACE after Step 2
        "lanes": 1,
        "approx_road_width_m": 3.5,
    },
}

# Convenience alias used throughout the pipeline
PIXELS_PER_METRE = {seg: v["pixels_per_metre"] for seg, v in CALIBRATION.items()}

# ---------------------------------------------------------------------------
# REGION OF INTEREST (ROI) POLYGONS
# ---------------------------------------------------------------------------
# Each polygon is a list of (x, y) pixel tuples.
# These are PLACEHOLDER coordinates — replace after Step 3.
#
# How to read the placeholders:
#   "regional" : a wide trapezoid covering both carriageways of the
#                Regional Road tunnel entrance.
#   "valley"   : a narrow trapezoid covering the single lane on Valley Road.
#
# IMPORTANT: These coordinates assume a 1920×1080 source frame.
#            If your clips are a different resolution, either:
#              (a) re-measure from your actual frame, OR
#              (b) set ROI_POLYGONS = None to disable masking entirely
#                  and re-enable once you have measured coordinates.
#
# To disable ROI masking temporarily:   ROI_POLYGONS = None

ROI_POLYGONS: dict[str, list[tuple[int, int]]] | None = {
    "regional": [
        # ┌─────────────────────────────────────────────────────────────────┐
        # │  Regional Road — Double Lane — Tunnel Entrance                  │
        # │  PLACEHOLDER  (1920×1080 frame assumed)                         │
        # │  Replace with your actual (x, y) corner clicks from the still.  │
        # └─────────────────────────────────────────────────────────────────┘
        (430, 350),   # top-left  corner of road area
        (1490, 350),  # top-right corner
        (1750, 980),  # bottom-right corner
        (170, 980),   # bottom-left corner
    ],
    "valley": [
        # ┌─────────────────────────────────────────────────────────────────┐
        # │  Valley Road — Single Lane                                      │
        # │  PLACEHOLDER  (1920×1080 frame assumed)                         │
        # └─────────────────────────────────────────────────────────────────┘
        (700, 380),   # top-left
        (1220, 380),  # top-right
        (1400, 980),  # bottom-right
        (520, 980),   # bottom-left
    ],
}

# ---------------------------------------------------------------------------
# VIRTUAL GATE POSITIONS
# ---------------------------------------------------------------------------
# Gate Y position expressed as a fraction of frame height (0.0 = top, 1.0 = bottom).
# The gate spans the full width of the ROI polygon (not the full frame).
#
# Rule of thumb: place the gate at roughly mid-frame so vehicles have had
# time to stabilise their track ID before they cross.
#
# GATE_Y_RATIOS["regional"]["primary"]  = 0.55  means 55% down the frame.

GATE_Y_RATIOS: dict[str, dict[str, float]] = {
    "regional": {
        "primary":   0.55,   # main counting gate  — ADJUST if needed
        "secondary": 0.75,   # optional second gate (unused in Phase 1 CSV)
    },
    "valley": {
        "primary":   0.50,   # ADJUST if needed
    },
}

# ---------------------------------------------------------------------------
# YOLO / TRACKING SETTINGS
# ---------------------------------------------------------------------------

# COCO class IDs that are treated as vehicles
VEHICLE_CLASSES: dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Default YOLOv8 weights — "yolov8n.pt" (nano) is fastest for a laptop.
# Upgrade to "yolov8s.pt" or "yolov8m.pt" for better accuracy at a speed cost.
DEFAULT_MODEL = "yolov8n.pt"

# Tracker config bundled with ultralytics
TRACKER_CFG = "bytetrack.yaml"

# How many consecutive frames to smooth speed over (rolling mean)
SPEED_SMOOTH_FRAMES = 10

# Minimum confidence threshold (0–1) — detections below this are discarded
CONFIDENCE_THRESHOLD = 0.35

# ---------------------------------------------------------------------------
# AGGREGATION
# ---------------------------------------------------------------------------
BIN_SECONDS = 60   # 1-minute bins for the summary CSV

# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------
# Per-frame CSV columns (produced by detect.py)
PER_FRAME_COLUMNS = ["frame", "id", "class", "speed_kph", "entry_time"]
