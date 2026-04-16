"""
vision/
=======
Phase 1 — Computer Vision pipeline for Malta Road Recovery.

Modules
-------
config          Calibration constants, ROI polygons, gate positions.
utils           Geometry helpers: ROI mask, point-in-polygon, drawing.
virtual_gates   VirtualGate class, GateCrossing dataclass, make_gate factory.
detect          Full single-clip pipeline: YOLO + ByteTrack + speed estimation.
extract_metrics Post-processing: per-frame CSV → calibration metrics CSV.

Typical import
--------------
    from vision.config import CALIBRATION, ROI_POLYGONS
    from vision.detect import run_pipeline
    from vision.extract_metrics import process_file
"""
