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
        "pixels_per_metre": 28,
        "lanes": 2,
        "approx_road_width_m": 7,
    },
    "valley": {
        # Valley Road — Single-Lane section
        "pixels_per_metre": 33,  # CHANGE
        "lanes": 1,
        "approx_road_width_m": 3.0,
    },
}

# Convenience alias used throughout the pipeline
PIXELS_PER_METRE = {seg: v["pixels_per_metre"] for seg, v in CALIBRATION.items()}

# ---------------------------------------------------------------------------
# REGION OF INTEREST (ROI) POLYGONS
# ---------------------------------------------------------------------------
# Native (post-rotation) frame is 1080 wide × 1920 tall (portrait) for both
# segments.  Calibration was done in different display windows per segment,
# so each segment carries its own view size and its own scaling factors:
#
#     native_x = measured_x * scale_x   (view_width  → 1080)
#     native_y = measured_y * scale_y   (view_height → 1920)
#
# The segment's scale_x is also applied to its pixels-per-metre at runtime
# (see detect.py), so a PPM measured on that view produces correct speeds
# on the 1080p native frame.
#
# To disable ROI masking temporarily:   ROI_POLYGONS = None

NATIVE_FRAME_WIDTH  = 1080   # px — width of the raw portrait frame (post-rotation)
NATIVE_FRAME_HEIGHT = 1920   # px — height of the raw portrait frame (post-rotation)

# Per-segment calibration-view dimensions (px).  Everything measured for
# that segment — ROI corners, PPM — was taken in this window size.
CALIBRATION_VIEWS: dict[str, tuple[int, int]] = {
    "regional": (605, 1080),   # Regional Road measurements taken here
    "valley":   (510,  910),   # Valley Road measurements taken here
}


def _segment_scales(segment: str) -> tuple[float, float]:
    """Return (scale_x, scale_y) that map this segment's calibration view to the native frame."""
    view_w, view_h = CALIBRATION_VIEWS[segment]
    return NATIVE_FRAME_WIDTH / view_w, NATIVE_FRAME_HEIGHT / view_h


# Convenience look-ups, populated once at import time
SCALING_FACTORS: dict[str, tuple[float, float]] = {
    seg: _segment_scales(seg) for seg in CALIBRATION_VIEWS
}


def scaling_factor_x(segment: str) -> float:
    """X-axis calibration-view → native-frame scale for a segment."""
    return SCALING_FACTORS[segment][0]


def scaling_factor_y(segment: str) -> float:
    """Y-axis calibration-view → native-frame scale for a segment."""
    return SCALING_FACTORS[segment][1]


# Backwards-compatible defaults (regional view, kept for any legacy imports)
SCALING_FACTOR_X = scaling_factor_x("regional")
SCALING_FACTOR_Y = scaling_factor_y("regional")


def _scale_polygon(
    segment: str,
    points: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Scale measured (x, y) from a segment's calibration view to the native frame."""
    sx, sy = SCALING_FACTORS[segment]
    return [(int(round(x * sx)), int(round(y * sy))) for x, y in points]


# ROI corners are ENTERED in each segment's own calibration view (see
# CALIBRATION_VIEWS above) and scaled to the 1080 × 1920 native frame at
# load time via _scale_polygon().
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
        # │  Coords measured on the 510 × 910 calibration view.             │
        # └─────────────────────────────────────────────────────────────────┘
        (290, 510),   # top-left
        (370, 530),   # top-right
        (200, 830),   # bottom-right
        (0,   730),   # bottom-left
    ],
}

ROI_POLYGONS: dict[str, list[tuple[int, int]]] | None = {
    seg: _scale_polygon(seg, pts) for seg, pts in _ROI_POLYGONS_RAW.items()
}

# ---------------------------------------------------------------------------
# DUAL-GATE SPEED TRAP
# ---------------------------------------------------------------------------
# Each segment defines TWO horizontal gate lines — an entry gate and an
# exit gate — separated by a known real-world distance on the road.
# A vehicle's speed is derived from the time it takes to travel between
# them (not from per-frame pixel motion):
#
#     speed_mps = gate_distance_m / (exit_time_s - entry_time_s)
#
# gate_entry / gate_exit : Y position as a fraction of NATIVE frame height
#                          (0.0 = top, 1.0 = bottom).  Resolution-independent,
#                          so these values do not need SCALING_FACTOR_Y.
# gate_distance_m        : measured real-world distance (metres) between the
#                          two gate lines on the road surface.
#
# Both lines must sit inside the ROI polygon and should be far enough apart
# for the tracker to stabilise (a few vehicle lengths is a good rule).

GATE_Y_RATIOS: dict[str, dict[str, float]] = {
    "regional": {
        "gate_entry":      0.6,
        "gate_exit":       0.75,
        "gate_distance_m": 15.0,
    },
    "valley": {
        "gate_entry":      0.595,
        "gate_exit":       0.785,
        "gate_distance_m": 8.5, # CHANGE
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

# Per-vehicle CSV columns — one row per tracked vehicle for easy reading
PER_VEHICLE_COLUMNS = [
    "id",
    "class",
    "entry_time_s",
    "exit_time_s",
    "duration_s",
    "gate_speed_kph",
    "avg_speed_kph",
    "max_speed_kph",
    "frames_seen",
    "direction",
]
