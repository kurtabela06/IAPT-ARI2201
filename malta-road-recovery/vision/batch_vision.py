"""
batch_vision.py
===============
Convenience script — processes all available video clips in data/raw/video/
using the naming convention:  <segment>_<slot>.mp4

Example file names:
    regional_10am.mp4
    regional_1pm.mp4
    regional_5pm.mp4
    valley_10am.mp4
    valley_1pm.mp4
    valley_5pm.mp4

Usage:
    python vision/batch_vision.py [--model yolov8n.pt] [--preview]
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR    = PROJECT_ROOT / "data" / "raw" / "video"

# Import pipeline
sys.path.insert(0, str(PROJECT_ROOT))
from vision.detect import run_pipeline
from vision.extract_metrics import main as extract_main

KNOWN_SEGMENTS = {"regional", "valley"}
KNOWN_SLOTS    = {"10am", "1pm", "5pm"}


def discover_videos() -> list[dict]:
    """
    Scan VIDEO_DIR for .mp4 / .avi files matching <segment>_<slot>.<ext>.
    Returns a list of dicts with keys: path, segment, slot.
    """
    found = []
    for ext in ("*.mp4", "*.avi", "*.mov"):
        for p in sorted(VIDEO_DIR.glob(ext)):
            parts = p.stem.split("_", 1)
            if len(parts) != 2:
                print(f"  [skip] Cannot parse segment/slot from: {p.name}")
                continue
            segment, slot = parts
            if segment not in KNOWN_SEGMENTS:
                print(f"  [skip] Unknown segment '{segment}' in: {p.name}")
                continue
            found.append({"path": p, "segment": segment, "slot": slot})
    return found


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch-process all Malta road video clips"
    )
    parser.add_argument("--model",   default="yolov8n.pt")
    parser.add_argument("--gate",    default=0.55, type=float)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    videos = discover_videos()

    if not videos:
        sys.exit(
            "CRITICAL: Video samples not found in data/raw/video. "
            "Please provide 15-minute clips to proceed with ground-truth calibration."
        )

    print(f"\n[Batch] Found {len(videos)} video clip(s) to process.\n")

    output_csvs = []
    for v in videos:
        print(f"{'='*60}")
        print(f"  Processing: {v['path'].name}")
        out = run_pipeline(
            video_path   = v["path"],
            segment      = v["segment"],
            slot         = v["slot"],
            model_name   = args.model,
            gate_ratio   = args.gate,
            show_preview = args.preview,
        )
        output_csvs.append(out)

    print(f"\n{'='*60}")
    print(f"[Batch] All done. {len(output_csvs)} CSV(s) written:\n")
    for p in output_csvs:
        print(f"  {p}")
    print()


if __name__ == "__main__":
    main()