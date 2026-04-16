"""
vision/utils.py
===============
Shared geometry and drawing helpers for the Malta Road Recovery pipeline.

Public API
----------
build_roi_mask(frame_shape, polygon)
    → np.ndarray uint8 binary mask (255 inside ROI, 0 outside)

apply_roi_mask(frame, mask)
    → frame with everything outside the ROI blacked out

point_in_polygon(x, y, polygon)
    → bool — True if (x, y) falls inside the polygon

draw_roi(frame, polygon, colour, label)
    → frame annotated with the polygon outline and label

clip_bbox_to_roi(x1, y1, x2, y2, mask)
    → bool — True if the bbox centroid falls inside the ROI mask
"""

from __future__ import annotations

import numpy as np
import cv2


def build_roi_mask(
    frame_shape: tuple[int, int, int] | tuple[int, int],
    polygon: list[tuple[int, int]],
) -> np.ndarray:
    """
    Build a binary uint8 mask of the same H×W as the source frame.

    Parameters
    ----------
    frame_shape : (H, W, C) or (H, W) — shape of the video frame
    polygon     : ordered list of (x, y) corner pixels defining the ROI

    Returns
    -------
    mask : np.ndarray, dtype=uint8, shape=(H, W)
        255 inside the polygon, 0 everywhere else.

    Example
    -------
    >>> mask = build_roi_mask(frame.shape, [(100,200),(500,200),(500,600),(100,600)])
    """
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)
    return mask


def apply_roi_mask(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Zero out every pixel outside the ROI polygon.

    Parameters
    ----------
    frame : BGR image (H, W, 3)
    mask  : uint8 mask from build_roi_mask — same H×W

    Returns
    -------
    Masked copy of `frame`.  The original array is not modified.
    """
    masked = frame.copy()
    masked[mask == 0] = 0
    return masked


def point_in_polygon(x: float, y: float, polygon: list[tuple[int, int]]) -> bool:
    """
    Ray-casting test: returns True if the point (x, y) is inside the polygon.

    This is the authoritative check used by VirtualGate — faster than
    cv2.pointPolygonTest for repeated single-point queries.
    """
    n = len(polygon)
    inside = False
    px, py = float(x), float(y)
    j = n - 1
    for i in range(n):
        xi, yi = float(polygon[i][0]), float(polygon[i][1])
        xj, yj = float(polygon[j][0]), float(polygon[j][1])
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def centroid_in_roi(
    x1: float, y1: float, x2: float, y2: float,
    polygon: list[tuple[int, int]],
) -> bool:
    """
    Returns True if the centroid of bounding box (x1,y1,x2,y2) is inside
    the given polygon.  Used to filter out detections outside the road area.
    """
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return point_in_polygon(cx, cy, polygon)


def draw_roi(
    frame: np.ndarray,
    polygon: list[tuple[int, int]],
    colour: tuple[int, int, int] = (0, 200, 255),
    label: str = "ROI",
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw the ROI polygon outline and label onto a frame (in-place).

    Parameters
    ----------
    frame    : BGR image to annotate
    polygon  : list of (x, y) tuples
    colour   : BGR colour for the outline
    label    : text label placed at the first polygon vertex
    thickness: line thickness in pixels

    Returns
    -------
    Annotated frame (same object as input).
    """
    pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], isClosed=True, color=colour, thickness=thickness)
    if polygon:
        cv2.putText(frame, label, (polygon[0][0] + 4, polygon[0][1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    return frame


def extract_still_frame(video_path: str, frame_index: int = 300) -> np.ndarray | None:
    """
    Convenience helper: grab a single frame from a video file.

    Used by calibration scripts (e.g. python -c "from vision.utils import ...").

    Parameters
    ----------
    video_path  : path to the .mp4 / .avi file
    frame_index : which frame to grab (default 300 ≈ 10 s at 30 fps)

    Returns
    -------
    BGR image array, or None if the frame could not be read.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None
