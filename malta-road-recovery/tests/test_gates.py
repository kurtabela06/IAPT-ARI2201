"""
tests/test_gates.py
===================
Unit tests for the virtual gate and ROI utilities.

Run with:
    cd malta-road-recovery
    python -m pytest tests/test_gates.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Make sure the project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vision.utils import (
    build_roi_mask,
    point_in_polygon,
    centroid_in_roi,
    apply_roi_mask,
)
from vision.virtual_gates import VirtualGate, GateCrossing, make_gate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SQUARE_POLY = [(100, 100), (400, 100), (400, 400), (100, 400)]  # 300×300 square
GATE_Y      = 250   # horizontal line through the middle of the square


@pytest.fixture
def gate_no_roi() -> VirtualGate:
    """Gate with no ROI filter."""
    return VirtualGate(y_position=GATE_Y, frame_width=640, label="test-gate")


@pytest.fixture
def gate_with_roi() -> VirtualGate:
    """Gate with a 300×300 square ROI."""
    return VirtualGate(
        y_position  = GATE_Y,
        frame_width = 640,
        label       = "test-gate-roi",
        roi_polygon = SQUARE_POLY,
    )


# ---------------------------------------------------------------------------
# point_in_polygon
# ---------------------------------------------------------------------------

class TestPointInPolygon:
    def test_centre_inside(self):
        assert point_in_polygon(250, 250, SQUARE_POLY) is True

    def test_outside(self):
        assert point_in_polygon(50, 50, SQUARE_POLY) is False

    def test_on_edge_roughly_inside(self):
        # Just inside the left edge
        assert point_in_polygon(101, 250, SQUARE_POLY) is True

    def test_bottom_right_outside(self):
        assert point_in_polygon(450, 450, SQUARE_POLY) is False

    def test_triangle_polygon(self):
        tri = [(0, 0), (200, 0), (100, 200)]
        assert point_in_polygon(100, 50, tri) is True
        assert point_in_polygon(0, 200, tri) is False


# ---------------------------------------------------------------------------
# centroid_in_roi
# ---------------------------------------------------------------------------

class TestCentroidInROI:
    def test_bbox_centroid_inside(self):
        # bbox centred at (250, 250) — inside SQUARE_POLY
        assert centroid_in_roi(200, 200, 300, 300, SQUARE_POLY) is True

    def test_bbox_centroid_outside(self):
        # bbox centred at (50, 50) — outside
        assert centroid_in_roi(0, 0, 100, 100, SQUARE_POLY) is False


# ---------------------------------------------------------------------------
# build_roi_mask / apply_roi_mask
# ---------------------------------------------------------------------------

class TestROIMask:
    def test_mask_shape(self):
        mask = build_roi_mask((720, 1280, 3), SQUARE_POLY)
        assert mask.shape == (720, 1280)
        assert mask.dtype == np.uint8

    def test_mask_inside_is_255(self):
        mask = build_roi_mask((500, 500), SQUARE_POLY)
        # Centre of the polygon should be 255
        assert mask[250, 250] == 255

    def test_mask_outside_is_0(self):
        mask = build_roi_mask((500, 500), SQUARE_POLY)
        # Far corner should be 0
        assert mask[10, 10] == 0

    def test_apply_zeros_outside(self):
        frame = np.ones((500, 500, 3), dtype=np.uint8) * 200
        mask  = build_roi_mask((500, 500), SQUARE_POLY)
        out   = apply_roi_mask(frame, mask)
        # Inside should keep value
        assert out[250, 250, 0] == 200
        # Outside should be zeroed
        assert out[10, 10, 0] == 0

    def test_original_frame_unchanged(self):
        frame = np.ones((500, 500, 3), dtype=np.uint8) * 200
        mask  = build_roi_mask((500, 500), SQUARE_POLY)
        _     = apply_roi_mask(frame, mask)
        assert frame[10, 10, 0] == 200   # original not modified


# ---------------------------------------------------------------------------
# VirtualGate — no ROI
# ---------------------------------------------------------------------------

class TestVirtualGateNoROI:
    def test_no_crossing_first_frame(self, gate_no_roi):
        result = gate_no_roi.update(1, cx=200, cy=200, timestamp_s=0.0)
        assert result is None

    def test_downward_crossing(self, gate_no_roi):
        # Move from above to below the gate
        gate_no_roi.update(1, cx=200, cy=GATE_Y - 10, timestamp_s=0.0)
        result = gate_no_roi.update(1, cx=200, cy=GATE_Y + 10, timestamp_s=0.5)
        assert isinstance(result, GateCrossing)
        assert result.direction == "downward"
        assert result.track_id == 1

    def test_upward_crossing(self, gate_no_roi):
        gate_no_roi.update(2, cx=200, cy=GATE_Y + 10, timestamp_s=0.0)
        result = gate_no_roi.update(2, cx=200, cy=GATE_Y - 10, timestamp_s=0.5)
        assert isinstance(result, GateCrossing)
        assert result.direction == "upward"

    def test_no_crossing_same_side(self, gate_no_roi):
        gate_no_roi.update(3, cx=200, cy=GATE_Y - 50, timestamp_s=0.0)
        result = gate_no_roi.update(3, cx=200, cy=GATE_Y - 20, timestamp_s=0.5)
        assert result is None

    def test_speed_recorded_on_crossing(self, gate_no_roi):
        gate_no_roi.update(4, cx=200, cy=GATE_Y - 10, timestamp_s=0.0)
        result = gate_no_roi.update(
            4, cx=200, cy=GATE_Y + 10,
            timestamp_s=0.5, class_name="car", speed_kph=45.0
        )
        assert result is not None
        assert result.speed_kph == pytest.approx(45.0)
        assert result.class_name == "car"

    def test_crossing_at_exact_gate_y(self, gate_no_roi):
        # Centroid lands exactly on the gate line — counts as crossing
        gate_no_roi.update(5, cx=200, cy=GATE_Y - 1, timestamp_s=0.0)
        result = gate_no_roi.update(5, cx=200, cy=GATE_Y, timestamp_s=0.1)
        assert result is not None

    def test_independent_track_ids(self, gate_no_roi):
        gate_no_roi.update(10, cx=200, cy=GATE_Y - 10, timestamp_s=0.0)
        gate_no_roi.update(11, cx=200, cy=GATE_Y + 10, timestamp_s=0.0)
        r10 = gate_no_roi.update(10, cx=200, cy=GATE_Y + 10, timestamp_s=0.1)
        r11 = gate_no_roi.update(11, cx=200, cy=GATE_Y - 10, timestamp_s=0.1)
        assert r10 is not None
        assert r11 is not None
        assert r10.direction == "downward"
        assert r11.direction == "upward"

    def test_forget_clears_state(self, gate_no_roi):
        gate_no_roi.update(20, cx=200, cy=GATE_Y - 10, timestamp_s=0.0)
        gate_no_roi.forget(20)
        # After forget, the next update should be treated as first observation
        result = gate_no_roi.update(20, cx=200, cy=GATE_Y + 10, timestamp_s=0.5)
        assert result is None


# ---------------------------------------------------------------------------
# VirtualGate — with ROI
# ---------------------------------------------------------------------------

class TestVirtualGateWithROI:
    def test_crossing_inside_roi_detected(self, gate_with_roi):
        # Centroid at x=250 — inside the 100-400 x range
        gate_with_roi.update(1, cx=250, cy=GATE_Y - 10, timestamp_s=0.0)
        result = gate_with_roi.update(1, cx=250, cy=GATE_Y + 10, timestamp_s=0.5)
        assert result is not None

    def test_crossing_outside_roi_ignored(self, gate_with_roi):
        # Centroid at x=50 — outside the polygon
        gate_with_roi.update(2, cx=50, cy=GATE_Y - 10, timestamp_s=0.0)
        result = gate_with_roi.update(2, cx=50, cy=GATE_Y + 10, timestamp_s=0.5)
        assert result is None

    def test_crossing_exactly_on_roi_boundary(self, gate_with_roi):
        # x=101 is just inside the left edge
        gate_with_roi.update(3, cx=101, cy=GATE_Y - 5, timestamp_s=0.0)
        result = gate_with_roi.update(3, cx=101, cy=GATE_Y + 5, timestamp_s=0.1)
        assert result is not None


# ---------------------------------------------------------------------------
# make_gate factory
# ---------------------------------------------------------------------------

class TestMakeGate:
    def test_regional_primary(self):
        g = make_gate("regional", frame_height=1080, frame_width=1920)
        assert isinstance(g, VirtualGate)
        # Default ratio is 0.55 → y = 594
        assert g.y == int(1080 * 0.55)

    def test_valley_primary(self):
        g = make_gate("valley", frame_height=720, frame_width=1280)
        assert isinstance(g, VirtualGate)
        assert g.y == int(720 * 0.50)

    def test_unknown_segment_raises(self):
        with pytest.raises(ValueError, match="No gate config"):
            make_gate("motorway", frame_height=720, frame_width=1280)

    def test_unknown_gate_name_raises(self):
        with pytest.raises(ValueError, match="No 'tertiary'"):
            make_gate("regional", frame_height=720, frame_width=1280, gate_name="tertiary")

    def test_roi_attached_when_config_has_polygon(self):
        from vision.config import ROI_POLYGONS
        g = make_gate("regional", frame_height=1080, frame_width=1920)
        if ROI_POLYGONS is not None and "regional" in ROI_POLYGONS:
            assert g.roi_polygon is not None
        else:
            assert g.roi_polygon is None
