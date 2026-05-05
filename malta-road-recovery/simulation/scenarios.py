"""
simulation/scenarios.py
=======================
Pre-compute the 120-scenario synthetic dataset.

Grid:
    4 directions × 12 time slots × variants
        regional: 3 variants  (all_open, 1_lane_closed, 2_lanes_closed)
        valley:   2 variants  (all_open, 1_lane_closed)
    → 4 × 12 × (3 or 2) = 12 × (3+3+2+2) = 120 scenarios

Each scenario produces:
    data/synthetic/scenarios/<direction>_<slot>_<variant>.csv
        — telemetry: one row per (sim_second × edge)
    data/synthetic/scenarios/<direction>_<slot>_<variant>.json
        — summary: exit count, escape count, total delay, scaling, recovery flag

The batch runner uses the calibrated `injection_count` (Q4) so each variant
has the **same** demand pulse — only the closure differs. That's how we
isolate "Recovery Time" as a learnable target.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Allow running as a script from the project root.
PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

from simulation.engine import Scenario, simulate, load_engine_config
from simulation.networks import DIRECTION_TO_NETWORK
from simulation.calibrator import calibrate_injection, scaled_pulses

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCN_DIR      = PROJECT_ROOT / "data" / "synthetic" / "scenarios"
SCN_DIR.mkdir(parents=True, exist_ok=True)

# Slots to simulate — strictly the three time-slots we filmed in the field.
# No interpolation, no synthetic slots.
TIME_SLOTS = ("10am", "1pm", "5pm")

REGIONAL_VARIANTS = ("all_open", "1_lane_closed", "2_lanes_closed")
VALLEY_VARIANTS   = ("all_open", "1_lane_closed")

DIRECTIONS = list(DIRECTION_TO_NETWORK.keys())


def variants_for(direction: str) -> tuple[str, ...]:
    return REGIONAL_VARIANTS if "regional" in direction else VALLEY_VARIANTS


def output_paths(direction: str, slot: str, variant: str) -> tuple[Path, Path, Path]:
    """Return (telemetry_csv, summary_json, particles_csv)."""
    csv  = SCN_DIR / f"{direction}_{slot}_{variant}.csv"
    js   = SCN_DIR / f"{direction}_{slot}_{variant}.json"
    pcsv = SCN_DIR / f"{direction}_{slot}_{variant}.particles.csv"
    return csv, js, pcsv


def run_one(direction: str, slot: str, variant: str,
            *, force: bool = False, verbose: bool = False) -> dict:
    """Run a single scenario, persist telemetry + particles + summary."""
    csv_p, json_p, pcsv_p = output_paths(direction, slot, variant)
    if csv_p.exists() and json_p.exists() and pcsv_p.exists() and not force:
        if verbose:
            print(f"  [scn] {direction}/{slot}/{variant}: cached")
        return json.loads(json_p.read_text())

    manual_slot = slot          # slots == manual slots now (Fix 2)
    calib = calibrate_injection(direction, manual_slot, verbose=False)
    scaling = calib["scaling"]
    pulses = scaled_pulses(direction, manual_slot, scaling)
    injection_count = calib["injected_total"]

    sc = Scenario(
        network_key=DIRECTION_TO_NETWORK[direction],
        direction=direction, slot=slot, variant=variant,
        pulses=pulses, injection_count=injection_count,
    )
    cfg = load_engine_config()
    df, particles_df, summary = simulate(sc, cfg=cfg)

    df.to_csv(csv_p, index=False)
    particles_df.to_csv(pcsv_p, index=False)
    summary["manual_slot"]      = manual_slot
    summary["calibration_scl"]  = scaling
    summary["injection_count"]  = injection_count
    json_p.write_text(json.dumps(summary, indent=2))

    if verbose:
        print(f"  [scn] {direction}/{slot}/{variant}: "
              f"exit={summary['exit_total']}  escape={summary['escape_total']}  "
              f"delay={summary['total_delay_veh_s']}")
    return summary


def run_all(*, force: bool = False, verbose: bool = True) -> list[dict]:
    summaries = []
    total = sum(len(variants_for(d)) for d in DIRECTIONS) * len(TIME_SLOTS)
    n = 0
    for direction in DIRECTIONS:
        for slot in TIME_SLOTS:
            for variant in variants_for(direction):
                n += 1
                if verbose:
                    print(f"\n[{n:3d}/{total}] {direction} / {slot} / {variant}")
                summaries.append(run_one(direction, slot, variant,
                                         force=force, verbose=verbose))
    if verbose:
        print(f"\n[scn] {len(summaries)} scenarios written to {SCN_DIR}")
    return summaries


def list_scenarios() -> pd.DataFrame:
    """Tabulate every cached scenario summary."""
    rows = []
    for js in sorted(SCN_DIR.glob("*.json")):
        rows.append(json.loads(js.read_text()))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-compute scenario grid")
    parser.add_argument("--all", action="store_true", help="Run the full 120-scenario grid")
    parser.add_argument("--direction", choices=DIRECTIONS)
    parser.add_argument("--slot")
    parser.add_argument("--variant")
    parser.add_argument("--force", action="store_true", help="Recompute even if cache exists")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet
    if args.all:
        run_all(force=args.force, verbose=verbose)
    elif args.direction and args.slot and args.variant:
        s = run_one(args.direction, args.slot, args.variant,
                    force=args.force, verbose=verbose)
        print(json.dumps(s, indent=2))
    else:
        parser.error("Use --all, or supply --direction --slot --variant")


if __name__ == "__main__":
    main()
