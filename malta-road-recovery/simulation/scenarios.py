"""
simulation/scenarios.py
=======================
Pre-compute the synthetic scenario library.

The CLI hierarchy (Q4 of this round):

    Level 1: Direction         (regional_north / regional_south / valley_east / valley_west)
    Level 2: Time slot         (10am / 1pm / 5pm)
    Level 3: Base horizon H    (600 / 900 / 1200 s)
    Level 4: Mode              (closure_only / closure_plus_recovery)
    Level 5: Transition        (regional only — 2_to_1 / 2_to_0)

For mode = closure_only the run lasts H seconds. For mode = closure_plus_recovery
the run lasts 2H seconds: closure-from-the-start, then a lane-swap (the chosen
"transition") at t = H. Demand is looped from the manual CSV — so a 1800 s
recovery run cycles through the 10 manual bins ~3 times.

Cache layout (duration-specific subfolders):

    data/synthetic/scenarios/
        600s/<scenario>.{csv, json, particles.csv}
        900s/<scenario>.{csv, json, particles.csv}
        1200s/<scenario>.{csv, json, particles.csv}
        1800s/<scenario>.{csv, json, particles.csv}    (recovery from H=900)
        2400s/<scenario>.{csv, json, particles.csv}    (recovery from H=1200)
        1200s_recovery/<scenario>.{csv, …}             (recovery from H=600)

Scenario IDs use the canonical `<direction>_<slot>_<H>s_<mode>_<transition>` form,
e.g.  regional_south_5pm_600s_closure_only_1_lane_closed
      regional_south_5pm_600s_recovery_2_to_1
      valley_east_5pm_900s_recovery
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

from simulation.engine import Scenario, RecoveryPolicy, simulate, load_engine_config
from simulation.networks import DIRECTION_TO_NETWORK
from simulation.calibrator import calibrate_injection, scaled_pulses

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCN_ROOT     = PROJECT_ROOT / "data" / "synthetic" / "scenarios"
SCN_ROOT.mkdir(parents=True, exist_ok=True)

TIME_SLOTS    = ("10am", "1pm", "5pm")
BASE_HORIZONS = (600, 900, 1200)               # H values offered by the CLI
DIRECTIONS    = list(DIRECTION_TO_NETWORK.keys())

# Mode labels.
MODE_CLOSURE_ONLY = "closure_only"
MODE_RECOVERY     = "recovery"

# For Mode B (recovery), the transition encodes how many lanes are CLOSED
# before and after recovery — matching the user's mental model.
# Format: (closure_variant, new_lane_count_after_recovery)
#
# Regional roads are 2-lane Main Ways, so:
#   "2closed_to_1closed"  starts with both lanes shut (eff=0), recovery reopens 1 lane (eff=1)
#   "2closed_to_0closed"  starts with both lanes shut (eff=0), recovery reopens fully (eff=2)
#   "1closed_to_0closed"  starts with 1 lane shut (eff=1), recovery reopens fully (eff=2)
REGIONAL_TRANSITIONS = {
    "2closed_to_1closed":  ("2_lanes_closed", 1),
    "2closed_to_0closed":  ("2_lanes_closed", 2),
    "1closed_to_0closed":  ("1_lane_closed",  2),
}
# Valley roads are 1-lane; "1 closed → 0 closed" is the only sensible recovery.
VALLEY_TRANSITIONS = {
    "1closed_to_0closed":  ("1_lane_closed",  1),
}

# Closure variants for Mode A (closure-only previews):
REGIONAL_CLOSURE_VARIANTS = ("all_open", "1_lane_closed", "2_lanes_closed")
VALLEY_CLOSURE_VARIANTS   = ("all_open", "1_lane_closed")


def transitions_for(direction: str) -> dict[str, tuple[str, int]]:
    return REGIONAL_TRANSITIONS if "regional" in direction else VALLEY_TRANSITIONS


def closure_variants_for(direction: str) -> tuple[str, ...]:
    return REGIONAL_CLOSURE_VARIANTS if "regional" in direction else VALLEY_CLOSURE_VARIANTS


# ---------------------------------------------------------------------------
# Path helpers — duration-specific subfolders
# ---------------------------------------------------------------------------

def _scenario_id(direction: str, slot: str, mode: str,
                 variant_or_transition: str) -> str:
    return f"{direction}_{slot}_{variant_or_transition}_{mode}"


def output_paths(direction: str, slot: str, base_h: int, mode: str,
                 variant_or_transition: str) -> tuple[Path, Path, Path]:
    """
    Return (telemetry_csv, summary_json, particles_csv) inside the
    duration-specific subfolder. Subfolder name = total horizon used:
       closure_only        → <H>s/
       closure_plus_recovery → <2H>s/
    """
    total_h = base_h if mode == MODE_CLOSURE_ONLY else 2 * base_h
    sub = SCN_ROOT / f"{total_h}s"
    sub.mkdir(parents=True, exist_ok=True)
    sid = _scenario_id(direction, slot, mode, variant_or_transition)
    return sub / f"{sid}.csv", sub / f"{sid}.json", sub / f"{sid}.particles.csv"


# ---------------------------------------------------------------------------
# Scenario builder
# ---------------------------------------------------------------------------

def _build_scenario(direction: str, slot: str, base_h: int, mode: str,
                    variant_or_transition: str) -> Scenario:
    """Construct a Scenario object from the hierarchical selection."""
    calib = calibrate_injection(direction, slot, verbose=False)
    scaling = calib["scaling"]
    pulses = scaled_pulses(direction, slot, scaling)
    injection = calib["injected_total"]

    if mode == MODE_CLOSURE_ONLY:
        return Scenario(
            network_key=DIRECTION_TO_NETWORK[direction],
            direction=direction, slot=slot, variant=variant_or_transition,
            pulses=pulses, injection_count=injection,
            horizon_s=base_h,
            base_horizon_s=base_h,
        )

    # MODE_RECOVERY
    transitions = transitions_for(direction)
    if variant_or_transition not in transitions:
        raise ValueError(
            f"Invalid transition '{variant_or_transition}' for {direction}. "
            f"Available: {list(transitions)}"
        )
    closure_variant, recovery_lanes = transitions[variant_or_transition]
    return Scenario(
        network_key=DIRECTION_TO_NETWORK[direction],
        direction=direction, slot=slot, variant=closure_variant,
        pulses=pulses, injection_count=injection,
        horizon_s=2 * base_h,
        base_horizon_s=base_h,
        recovery=RecoveryPolicy(start_s=base_h, new_lane_count=recovery_lanes),
        transition_label=_transition_label(variant_or_transition, closure_variant, recovery_lanes),
    )


def _transition_label(transition_key: str, closure_variant: str, recovery_lanes: int) -> str:
    pretty = {
        "2closed_to_1closed":  "Recovery: 2 lanes closed → 1 lane closed",
        "2closed_to_0closed":  "Recovery: 2 lanes closed → fully reopened",
        "1closed_to_0closed":  "Recovery: 1 lane closed  → fully reopened",
    }
    return pretty.get(transition_key, f"Recovery → eff_lanes={recovery_lanes}")


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------

def run_one(direction: str, slot: str, base_h: int, mode: str,
            variant_or_transition: str,
            *, force: bool = False, verbose: bool = False) -> dict:
    """Run one (direction, slot, H, mode, variant_or_transition) scenario."""
    csv_p, json_p, pcsv_p = output_paths(direction, slot, base_h, mode, variant_or_transition)
    if csv_p.exists() and json_p.exists() and pcsv_p.exists() and not force:
        if verbose:
            print(f"  [scn] {csv_p.parent.name}/{csv_p.stem}: cached")
        return json.loads(json_p.read_text())

    sc = _build_scenario(direction, slot, base_h, mode, variant_or_transition)
    cfg = load_engine_config()
    df, particles_df, summary = simulate(sc, cfg=cfg)

    df.to_csv(csv_p, index=False)
    particles_df.to_csv(pcsv_p, index=False)
    summary["manual_slot"]      = slot
    summary["base_horizon_s"]   = base_h
    summary["mode"]             = mode
    summary["variant_or_transition"] = variant_or_transition
    # Calibrated injection per CSV cycle (H=600 baseline):
    summary["calibrated_inflow_per_cycle"] = sc.injection_count
    # Actual injected total = sum(injected per bin) over all bins consumed
    # by the looped inflow in this run's horizon.
    bin_secs = cfg["engine"]["bin_seconds"]
    bins_per_cycle = len(sc.pulses) if sc.pulses else 1
    horizon = sc.horizon_s or cfg["engine"]["horizon_s"]
    bins_injected = horizon // bin_secs       # whole bins, since t < horizon guard
    # Per-bin counts after scaling
    if sc.pulses and sc.injection_count:
        observed_per_cycle = sum(p["car"]+p["hgv"]+p["motorcycle"] for p in sc.pulses)
        scaling = (sc.injection_count / observed_per_cycle) if observed_per_cycle else 1.0
        # Sum the integer-rounded injection per bin (looped) over bins_injected.
        total_inj = 0
        for b in range(bins_injected):
            p = sc.pulses[b % bins_per_cycle]
            for cls in ("car", "hgv", "motorcycle"):
                total_inj += int(round(p.get(cls, 0) * scaling))
        summary["injection_count"] = total_inj
    else:
        summary["injection_count"] = sc.injection_count or 0
    summary["loop_cycles"]      = round(bins_injected / bins_per_cycle, 3)
    summary["calibration_scl"]  = (sc.injection_count / sum(p["car"]+p["hgv"]+p["motorcycle"]
                                                            for p in sc.pulses)) if sc.pulses else 1.0
    json_p.write_text(json.dumps(summary, indent=2))

    if verbose:
        rec = summary.get("recovery_time_s")
        rec_str = f"  recovery={rec}s" if rec is not None else ""
        print(f"  [scn] {csv_p.parent.name}/{csv_p.stem}: "
              f"exit={summary['exit_total']}  escape={summary['escape_total']}  "
              f"delay={summary['total_delay_veh_s']:.0f}{rec_str}")
    return summary


# ---------------------------------------------------------------------------
# Bulk runner — all combinations
# ---------------------------------------------------------------------------

def iterate_grid():
    """Yield every (direction, slot, base_h, mode, variant_or_transition) combo."""
    for direction in DIRECTIONS:
        for slot in TIME_SLOTS:
            for base_h in BASE_HORIZONS:
                # Mode A (closure_only) → all standard closure variants
                for v in closure_variants_for(direction):
                    yield direction, slot, base_h, MODE_CLOSURE_ONLY, v
                # Mode B (recovery) → every transition supported by this network
                for t_key in transitions_for(direction):
                    yield direction, slot, base_h, MODE_RECOVERY, t_key


def run_all(*, force: bool = False, verbose: bool = True,
            build_ml_features: bool = True) -> list[dict]:
    summaries = []
    grid = list(iterate_grid())
    total = len(grid)
    for i, (direction, slot, base_h, mode, vot) in enumerate(grid, 1):
        if verbose:
            print(f"\n[{i:3d}/{total}] {direction} / {slot} / H={base_h}s / {mode} / {vot}")
        summaries.append(run_one(direction, slot, base_h, mode, vot,
                                 force=force, verbose=verbose))
    if verbose:
        print(f"\n[scn] {len(summaries)} scenarios written under {SCN_ROOT}")

    # Prompt 2: aggregate ML snapshot features.
    if build_ml_features:
        try:
            from simulation.build_features import build_all as _build_features_all
            if verbose:
                print(f"\n[scn] Building ML feature snapshots → data/ml_training/recovery_stats.csv")
            _build_features_all()
        except Exception as exc:
            print(f"  [scn] ML feature build failed: {exc}")

    return summaries


def list_scenarios() -> pd.DataFrame:
    """Tabulate every cached scenario summary across all duration subfolders."""
    rows = []
    for js in sorted(SCN_ROOT.glob("*/*.json")):
        rows.append(json.loads(js.read_text()))
    return pd.DataFrame(rows)


def count_scenarios() -> int:
    return len(list(SCN_ROOT.glob("*/*.json")))


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-compute scenario grid")
    parser.add_argument("--all", action="store_true",
                        help="Run every combination in the hierarchy")
    parser.add_argument("--direction", choices=DIRECTIONS)
    parser.add_argument("--slot",      choices=TIME_SLOTS)
    parser.add_argument("--base-h",    type=int, choices=BASE_HORIZONS)
    parser.add_argument("--mode",      choices=(MODE_CLOSURE_ONLY, MODE_RECOVERY))
    parser.add_argument("--variant-or-transition",
                        help="Closure variant (Mode A) or transition key (Mode B)")
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if cache exists")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet
    if args.all:
        run_all(force=args.force, verbose=verbose)
        return
    if not (args.direction and args.slot and args.base_h
            and args.mode and args.variant_or_transition):
        parser.error("Use --all, or supply --direction --slot --base-h --mode "
                     "--variant-or-transition")
    s = run_one(args.direction, args.slot, args.base_h, args.mode,
                args.variant_or_transition, force=args.force, verbose=verbose)
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
