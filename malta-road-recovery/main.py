"""
main.py
=======
Malta Road Recovery — Phase 1 batch entry point.

Phase 1 is Manual Ground-Truth Extraction. The automated YOLO pipeline has
been archived (see the Archive/ folder at the repo root). A human watches
each 10-minute clip, fills one CSV per segment × direction × time slot, and
this script drives the rest of the pipeline.

Directions handled: regional_north, regional_south, valley_east, valley_west.

For each  data/raw/manual_counts/<direction>/<slot>.csv  this produces:

  data/processed/aggregated/<direction>/<slot>.csv          (validated + enriched bins)
  simulation/demand/<direction>_<slot>.rou.xml              (SUMO routes)
  simulation/demand/<direction>_<slot>.flow.xml             (SUMO flows)
  data/processed/calibration/<direction>_<slot>_metrics.csv (calibration)

Usage
-----
    python main.py                    # process every manual CSV
    python main.py --no-metrics       # skip extract_metrics step
    python main.py --input <file>     # process a single CSV
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_processing.manual_to_sumo import (    # noqa: E402
    DIRECTIONS, MANUAL_IN_DIR, discover_manual_csvs,
    process_manual_file, _parse_direction_and_slot,
)
from data_processing.extract_metrics import process_file  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Malta Road Recovery — Phase 1 (Manual Ground Truth) batch runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", type=Path, default=None,
                   help="Process a single manual CSV "
                        "(default: every CSV under data/raw/manual_counts/<direction>/)")
    p.add_argument("--no-metrics", action="store_true",
                   help="Skip the extract_metrics step "
                        "(just produces aggregated bins + SUMO demand)")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    if args.input is not None:
        if not args.input.exists():
            sys.exit(f"CRITICAL: Input file not found: {args.input}")
        targets = [args.input]
    else:
        targets = discover_manual_csvs()

    if not targets:
        print(
            f"CRITICAL: No manual CSVs found under {MANUAL_IN_DIR}/<direction>/. "
            f"Expected direction folders: {list(DIRECTIONS)}. "
            "See data/raw/manual_counts/README.md for the row-level schema."
        )
        sys.exit(1)

    print(f"\n[main] Found {len(targets)} manual CSV(s) to process.\n")
    for p in targets:
        direction, slot = _parse_direction_and_slot(p)
        print(f"  {direction:<16}  {slot:<8}  {p.name}")
    print()

    results: list[dict] = []
    for p in targets:
        try:
            direction, slot = _parse_direction_and_slot(p)
            outputs = process_manual_file(p)
            result = {
                "direction":      direction,
                "slot":           slot,
                "aggregated_csv": outputs["aggregated"],
                "rou_xml":        outputs["rou"],
                "flow_xml":       outputs["flow"],
                "metrics_csv":    None,
                "status":         "ok",
            }
            if not args.no_metrics:
                try:
                    result["metrics_csv"] = process_file(outputs["aggregated"])
                except Exception as exc:
                    print(f"  [metrics] WARNING: metrics extraction failed: {exc}")
            results.append(result)

        except Exception as exc:
            print(f"\n[main] ERROR processing {p.name}: {exc}")
            results.append({"direction": p.parent.name, "slot": p.stem, "status": "failed"})

    ok     = [r for r in results if r.get("status") == "ok"]
    failed = [r for r in results if r.get("status") == "failed"]

    print(f"\n{'='*60}")
    print(f"[main] Phase 1 complete.  {len(ok)}/{len(results)} clips processed.\n")

    if ok:
        print("  Output files:")
        for r in ok:
            print(f"\n  [{r['direction']} / {r['slot']}]")
            print(f"    Aggregated CSV : {r['aggregated_csv']}")
            print(f"    SUMO .rou.xml  : {r['rou_xml']}")
            print(f"    SUMO .flow.xml : {r['flow_xml']}")
            if r.get("metrics_csv"):
                print(f"    Metrics CSV    : {r['metrics_csv']}")

    if failed:
        print(f"\n  Failed clips ({len(failed)}):")
        for r in failed:
            print(f"    {r['direction']}/{r['slot']}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
