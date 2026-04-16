"""
main.py
=======
Malta Road Recovery — Phase 1 batch entry point.

Scans data/raw/video/ for all .mp4 / .avi / .mov clips matching the naming
convention  <segment>_<slot>.<ext>  and runs the full detection pipeline on
each one, producing:

  data/processed/yolo_output/<segment>_<slot>_frames.csv   (per-frame)
  data/processed/yolo_output/<segment>_<slot>.csv          (1-min summary)
  data/processed/calibration/<segment>_<slot>_metrics.csv  (SUMO-ready)

EMPTY FOLDER GUARD
------------------
If data/raw/video/ contains no video files the script exits immediately with:
  CRITICAL: No video samples found in data/raw/video.
  Please add 15-minute clips (e.g., regional_10am.mp4) to begin.

NAMING CONVENTION
-----------------
Videos must be named   <segment>_<slot>.<ext>
  segment : "regional"  (Regional Road — twin lane / tunnel entrance)
            "valley"    (Valley Road — single lane)
  slot    : "10am" | "1pm" | "5pm"   (or any free-text label)
  ext     : .mp4 | .avi | .mov

Examples:
    regional_10am.mp4
    regional_1pm.mp4
    valley_5pm.avi

USAGE
-----
    cd malta-road-recovery
    python main.py                          # default YOLOv8n, no preview
    python main.py --preview                # open live OpenCV windows
    python main.py --model yolov8s.pt       # better accuracy, slower
    python main.py --no-metrics             # skip extract_metrics step
    python main.py --no-roi                 # disable ROI masking (debug)

CALIBRATION WORKFLOW (read before first run)
--------------------------------------------
1. Add your 15-minute clips to  data/raw/video/
2. Run:  python main.py --preview --model yolov8n.pt
3. Watch the preview window.  Adjust the following in vision/config.py:
     - ROI_POLYGONS     (replace placeholder coordinates with pixel clicks)
     - GATE_Y_RATIOS    (move the gate line to the right position)
     - CALIBRATION[*]["pixels_per_metre"]  (measure from a road feature)
4. Re-run without --preview for the full batch run.
5. Find outputs in data/processed/yolo_output/ and data/processed/calibration/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# PATH SETUP — run from anywhere inside the project
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from vision.config import CALIBRATION, DEFAULT_MODEL   # noqa: E402
from vision.detect import run_pipeline                  # noqa: E402
from vision.extract_metrics import process_file         # noqa: E402

VIDEO_DIR  = PROJECT_ROOT / "data" / "raw" / "video"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "yolo_output"

KNOWN_SEGMENTS = set(CALIBRATION.keys())
VIDEO_EXTS     = ("*.mp4", "*.avi", "*.mov", "*.MP4", "*.AVI", "*.MOV")


# ---------------------------------------------------------------------------
# VIDEO DISCOVERY
# ---------------------------------------------------------------------------

def discover_videos() -> list[dict]:
    """
    Scan VIDEO_DIR for clips matching  <segment>_<slot>.<ext>.

    Returns a list of dicts:  {path, segment, slot}
    Clips with unrecognised names are listed but skipped.
    """
    if not VIDEO_DIR.exists():
        return []

    found  = []
    skipped = []

    for ext in VIDEO_EXTS:
        for p in sorted(VIDEO_DIR.glob(ext)):
            parts = p.stem.split("_", 1)
            if len(parts) != 2:
                skipped.append(p.name)
                continue
            segment, slot = parts
            if segment not in KNOWN_SEGMENTS:
                skipped.append(p.name)
                continue
            found.append({"path": p, "segment": segment, "slot": slot})

    if skipped:
        print("\n[main] The following files were skipped (unrecognised naming):")
        for name in skipped:
            print(f"         {name}")
        print(f"       Expected format: <segment>_<slot>.mp4  "
              f"(segment must be one of: {sorted(KNOWN_SEGMENTS)})\n")

    return found


# ---------------------------------------------------------------------------
# ARGUMENT PARSER
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Malta Road Recovery — Phase 1 batch runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py
  python main.py --preview
  python main.py --model yolov8s.pt --no-roi
        """,
    )
    p.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"YOLO weights file  (default: {DEFAULT_MODEL}). "
             "Use yolov8s.pt / yolov8m.pt for better accuracy.",
    )
    p.add_argument(
        "--gate", default=None, type=float,
        help="Gate Y as fraction of frame height — overrides config for ALL clips",
    )
    p.add_argument(
        "--preview", action="store_true",
        help="Open a live OpenCV preview window for each clip (gate/ROI verification)",
    )
    p.add_argument(
        "--no-roi", action="store_true",
        help="Disable ROI masking globally (useful for debugging YOLO detections)",
    )
    p.add_argument(
        "--no-metrics", action="store_true",
        help="Skip the extract_metrics step (faster, just produces CSVs)",
    )
    return p


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    args   = _build_parser().parse_args()
    videos = discover_videos()

    # -----------------------------------------------------------------------
    # EMPTY FOLDER GUARD — exact required message
    # -----------------------------------------------------------------------
    if not videos:
        print(
            "CRITICAL: No video samples found in data/raw/video. "
            "Please add 15-minute clips (e.g., regional_10am.mp4) to begin."
        )
        sys.exit(1)

    print(f"\n[main] Found {len(videos)} clip(s) to process.\n")
    for v in videos:
        print(f"  {v['segment']:<10}  {v['slot']:<8}  {v['path'].name}")
    print()

    results: list[dict] = []

    for v in videos:
        try:
            frames_csv, summary_csv = run_pipeline(
                video_path   = v["path"],
                segment      = v["segment"],
                slot         = v["slot"],
                model_name   = args.model,
                gate_ratio   = args.gate,
                show_preview = args.preview,
                use_roi      = not args.no_roi,
            )
            result = {
                "segment":     v["segment"],
                "slot":        v["slot"],
                "frames_csv":  frames_csv,
                "summary_csv": summary_csv,
                "metrics_csv": None,
                "status":      "ok",
            }

            # Optionally run extract_metrics on the per-frame CSV
            if not args.no_metrics:
                try:
                    metrics_csv = process_file(frames_csv)
                    result["metrics_csv"] = metrics_csv
                except Exception as exc:
                    print(f"  [metrics] WARNING: metrics extraction failed: {exc}")

            results.append(result)

        except SystemExit as exc:
            print(f"\n[main] ERROR processing {v['path'].name}: {exc}")
            results.append({
                "segment": v["segment"],
                "slot":    v["slot"],
                "status":  "failed",
            })
        except Exception as exc:
            print(f"\n[main] UNEXPECTED ERROR on {v['path'].name}: {exc}")
            results.append({
                "segment": v["segment"],
                "slot":    v["slot"],
                "status":  "failed",
            })

    # -----------------------------------------------------------------------
    # FINAL SUMMARY
    # -----------------------------------------------------------------------
    ok     = [r for r in results if r.get("status") == "ok"]
    failed = [r for r in results if r.get("status") == "failed"]

    print(f"\n{'='*60}")
    print(f"[main] Phase 1 complete.  {len(ok)}/{len(results)} clips processed.\n")

    if ok:
        print("  Output files:")
        for r in ok:
            print(f"\n  [{r['segment']} / {r['slot']}]")
            print(f"    Per-frame CSV : {r['frames_csv']}")
            print(f"    Summary CSV   : {r['summary_csv']}")
            if r.get("metrics_csv"):
                print(f"    Metrics CSV   : {r['metrics_csv']}")

    if failed:
        print(f"\n  Failed clips ({len(failed)}):")
        for r in failed:
            print(f"    {r['segment']}_{r['slot']}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
