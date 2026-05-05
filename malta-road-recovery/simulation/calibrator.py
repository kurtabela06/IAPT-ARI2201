"""
simulation/calibrator.py
========================
Injection-count calibrator (Q4 — Calibration Rule for Demand Injection).

For each (direction × slot) we compute the **Injection Count** — the total
number of vehicles to spawn over the 600 s horizon under the All-Lanes-Open
variant — such that the simulated exit count matches the manual CSV total
within a small tolerance.

Once calibrated, the same injection count is reused by the 1-Lane-Closed
and Full-Closure variants. That way differences in exit flow between the
three variants are caused by the closure mechanics, not by demand changes.

The calibrator does a bisection search over a multiplicative scaling factor
applied uniformly to every minute's pulse:

    pulse'_i = round(pulse_i × scaling)

Bounds and tolerance come from config/simulation.yaml (`calibration:` block).

Cache file: data/processed/calibration/<direction>_<slot>_injection.json
   {
       "direction":        "regional_north",
       "slot":             "10am",
       "manual_total":     343,
       "scaling":          1.04,
       "injected_total":   358,
       "simulated_exit":   343,
       "iterations":       11,
       "match_pct":        0.0
   }
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

from simulation.engine import Scenario, simulate, load_engine_config
from simulation.networks import DIRECTION_TO_NETWORK, build_network
from data_processing.manual_to_pulse import load_pulses

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CALIB_DIR    = PROJECT_ROOT / "data" / "processed" / "calibration"
CALIB_DIR.mkdir(parents=True, exist_ok=True)


def cache_path(direction: str, slot: str) -> Path:
    return CALIB_DIR / f"{direction}_{slot}_injection.json"


def manual_total(pulses: list[dict]) -> int:
    return int(sum(p.get("car",0) + p.get("hgv",0) + p.get("motorcycle",0)
                   for p in pulses))


def calibrate_injection(direction: str, slot: str,
                        *, force: bool = False, verbose: bool = False) -> dict:
    """
    Bisect-search a scaling factor that drives simulated exit total to match
    the manual CSV total (All-Open variant). Cached after first run.
    """
    cp = cache_path(direction, slot)
    if cp.exists() and not force:
        return json.loads(cp.read_text())

    cfg = load_engine_config()
    cal = cfg["calibration"]
    tol_pct = cal["match_tolerance_pct"]
    max_it  = cal["max_iterations"]
    lo_f    = cal["search_low_factor"]
    hi_f    = cal["search_high_factor"]

    pulses_records = load_pulses(direction, slot)
    target = manual_total(pulses_records)
    if target == 0:
        result = {"direction": direction, "slot": slot, "manual_total": 0,
                  "scaling": 1.0, "injected_total": 0, "simulated_exit": 0,
                  "iterations": 0, "match_pct": 0.0}
        cp.write_text(json.dumps(result, indent=2))
        return result

    network_key = DIRECTION_TO_NETWORK[direction]

    def run_with(scaling: float) -> int:
        scaled = []
        for p in pulses_records:
            scaled.append({
                "car":              int(round(p["car"]        * scaling)),
                "hgv":              int(round(p["hgv"]        * scaling)),
                "motorcycle":       int(round(p["motorcycle"] * scaling)),
                "incident_status":  p["incident_status"],
            })
        sc = Scenario(network_key=network_key, direction=direction, slot=slot,
                      variant="all_open", pulses=scaled)
        _, _, summary = simulate(sc, cfg=cfg, emit_particles=False)
        return int(summary["exit_total"]) + int(summary["escape_total"])

    lo, hi = lo_f, hi_f
    best = None
    iterations = 0
    for it in range(max_it):
        iterations = it + 1
        mid = 0.5 * (lo + hi)
        sim_exit = run_with(mid)
        match_pct = abs(sim_exit - target) / max(target, 1) * 100.0
        if verbose:
            print(f"  [calib] {direction}/{slot} iter={it+1}  scaling={mid:.4f}  "
                  f"exit={sim_exit}  target={target}  Δ={match_pct:.2f}%")
        if best is None or match_pct < best[2]:
            best = (mid, sim_exit, match_pct)
        if match_pct <= tol_pct:
            break
        if sim_exit < target:
            lo = mid
        else:
            hi = mid

    scaling, sim_exit, match_pct = best
    injected = sum(int(round(p["car"]*scaling) + round(p["hgv"]*scaling) +
                       round(p["motorcycle"]*scaling))
                   for p in pulses_records)

    result = {
        "direction":      direction,
        "slot":           slot,
        "manual_total":   target,
        "scaling":        round(float(scaling), 4),
        "injected_total": int(injected),
        "simulated_exit": int(sim_exit),
        "iterations":     int(iterations),
        "match_pct":      round(float(match_pct), 3),
    }
    cp.write_text(json.dumps(result, indent=2))
    return result


def scaled_pulses(direction: str, slot: str, scaling: float) -> list[dict]:
    """Apply a scaling factor to the cached pulses for a slot."""
    pulses_records = load_pulses(direction, slot)
    return [
        {
            "car":              int(round(p["car"]        * scaling)),
            "hgv":              int(round(p["hgv"]        * scaling)),
            "motorcycle":       int(round(p["motorcycle"] * scaling)),
            "incident_status":  p["incident_status"],
        }
        for p in pulses_records
    ]


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Calibrate injection count for one slot")
    parser.add_argument("--direction", required=True, choices=list(DIRECTION_TO_NETWORK))
    parser.add_argument("--slot", required=True)
    parser.add_argument("--force", action="store_true",
                        help="Recalibrate even if cache exists")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    result = calibrate_injection(args.direction, args.slot,
                                 force=args.force, verbose=args.verbose)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
