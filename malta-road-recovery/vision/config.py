"""
vision/config.py
================
Single source-of-truth for every calibration constant in Phase 1.

VIDEO FORMAT
------------
Videos are recorded at 1080p, 30fps, PORTRAIT orientation.
  Frame size: 1080 wide x 1920 tall  (width x height)
  All (x, y) coordinates in this file use that convention.

HOW TO CALIBRATE BEFORE YOUR FIRST RUN
---------------------------------------
Step 1 — Use --test mode to grab a preview of the first frame:
    cd malta-road-recovery
    python vision/detect.py --video data/raw/video/regional_10am.mp4 \
        --segment regional --slot 10am --test

    The window that pops up shows frame 0 with the current ROI polygon
    and gate line drawn on it.  Use this to check your coordinates before
    committing to a full 10-minute run.

Step 2 — Measure PIXELS_PER_METRE from that still frame.
    Open the still in any image viewer that reports pixel coordinates
    (e.g. GIMP, macOS Preview, Paint).  Find a road feature whose real-world
    length you know (lane width ≈ 3.5 m, road marking ≈ 3 m, etc.) and note
    the pixel distance between its two ends.

    PIXELS_PER_METRE = <pixel_distance> / <real_distance_in_metres>

    Update the value for the relevant segment below.

Step 3 — ROI Polygon (Region of Interest)
    In the same still, click the corners that enclose the road you want to
    analyse.  Note each (x, y) pixel coordinate and replace the placeholder
    tuples in ROI_POLYGONS below.

    The polygon must be in ORDER (clockwise or anti-clockwise) and must
    fully enclose the road lanes you care about.

    Remember: portrait frame = 1080 wide x 1920 tall.
    x runs left→right (0-1080), y runs top→bottom (0-1920).

Step 4 — Virtual Gate
    Pick a Y pixel position that lies across the lane(s) inside your ROI.
    Vehicles are counted each time their centroid crosses this line.
    Update GATE_Y_RATIOS below (value is 0.0-1.0 fraction of frame height,
    so it scales to any resolution automatically).

Step 5 — Re-run with --test to visually confirm your edits, then run the
    full batch: python main.py
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
        "pixels_per_metre": 25,
        "lanes": 2,
        "approx_road_width_m": 7,  # includes some shoulder area, not just lane width
    },
    "valley": {
        # Valley Road — Single-Lane section
        # Approx. 1 lane, road width ~3.5 m.
        # PLACEHOLDER: measure from your footage and update (see Step 2 above).
        "pixels_per_metre": 10.2,  # <-- REPLACE after measuring
        "lanes": 1,
        "approx_road_width_m": 3.0,  # single lane with no shoulder
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
# COORDINATE INPUT CONVENTION
# ---------------------------
# Native (post-rotation) frame is 1080 wide × 1920 tall (portrait).
# Calibration measurements are taken on a 605 × 1080 display window, so
# every coordinate below is entered in that smaller frame and scaled up
# at load time:
#     native_x = measured_x * SCALING_FACTOR_X   (605  → 1080)
#     native_y = measured_y * SCALING_FACTOR_Y   (1080 → 1920)
#
# The same SCALING_FACTOR_X is also applied to pixels-per-metre at runtime
# (see detect.py) so a PPM measured on the 605px view produces correct
# speeds on the 1080px native frame.
#
# To disable ROI masking temporarily:   ROI_POLYGONS = None

CALIBRATION_VIEW_WIDTH  = 605    # px — width of the window where coords were measured
CALIBRATION_VIEW_HEIGHT = 1080   # px — height of the window where coords were measured
NATIVE_FRAME_WIDTH      = 1080   # px — width of the raw portrait frame (post-rotation)
NATIVE_FRAME_HEIGHT     = 1920   # px — height of the raw portrait frame (post-rotation)

SCALING_FACTOR_X = NATIVE_FRAME_WIDTH  / CALIBRATION_VIEW_WIDTH    # ≈ 1.7851
SCALING_FACTOR_Y = NATIVE_FRAME_HEIGHT / CALIBRATION_VIEW_HEIGHT   # ≈ 1.7778


def _scale_polygon(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Scale measured (x, y) from the 605×1080 calibration view to the 1080×1920 native frame."""
    return [
        (int(round(x * SCALING_FACTOR_X)), int(round(y * SCALING_FACTOR_Y)))
        for x, y in points
    ]


# All coordinates below are measured on the 605 × 1080 calibration view.
_ROI_POLYGONS_RAW: dict[str, list[tuple[int, int]]] = {
    "regional": [
        # ┌─────────────────────────────────────────────────────────────────┐
        # │  Regional Road — Double Lane — Tunnel Entrance                  │
        # │  Coords measured on the 605 × 1080 calibration view.            │
        # └─────────────────────────────────────────────────────────────────┘
        (200, 585),   # top-left  corner
        (270, 585),   # top-right corner
        (480, 850),   # bottom-right corner
        (190, 850),   # bottom-left corner
    ],
    "valley": [
        # ┌─────────────────────────────────────────────────────────────────┐
        # │  Valley Road — Single Lane                                      │
        # │  Coords measured on the 605 × 1080 calibration view.            │
        # └─────────────────────────────────────────────────────────────────┘
        (250, 500),   # top-left
        (830, 500),   # top-right
        (830, 1700),  # bottom-right
        (250, 1700),  # bottom-left
    ],
}

ROI_POLYGONS: dict[str, list[tuple[int, int]]] | None = {
    seg: _scale_polygon(pts) for seg, pts in _ROI_POLYGONS_RAW.items()
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

# ---------------------------------------------------------------------------
# M3 MAC INFERENCE OPTIMISATIONS
# ---------------------------------------------------------------------------

# Metal Performance Shaders — uses the M3 GPU instead of CPU.
# Set to "cpu" if you hit any MPS-related crash (rare on macOS 14+).
INFERENCE_DEVICE = "mps"

# YOLO inference resolution.  Even though the video is 1080p, YOLO detects
# best at 640px and it saves significant GPU memory.  Do not change this.
INFERENCE_IMGSZ = 640

# Frame stride — only process every Nth frame.
# At 30fps, FRAME_STRIDE=5 means ~6fps effective detection rate.
# Traffic moves slowly enough that this is more than sufficient for counting,
# and it cuts total processing time by ~80%.
# Set to 1 to process every frame (accurate but ~5× slower).
FRAME_STRIDE = 5

# ---------------------------------------------------------------------------
# TEST MODE
# ---------------------------------------------------------------------------

# When TEST_MODE is True (or --test flag is passed), the pipeline:
#   1. Only processes the first TEST_DURATION_S seconds of each video.
#   2. Pops up a window showing frame 0 with the ROI polygon and gate
#      overlaid, so you can check your coordinates before committing to a
#      full 10-minute run.
# Leave this as False for production runs — use --test on the command line.
TEST_MODE = False
TEST_DURATION_S = 60   # seconds of video to process in test mode

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
