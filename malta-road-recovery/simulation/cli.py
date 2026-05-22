"""
simulation/cli.py
=================
Persistent drill-down menu for the Phase 2 NetworkX engine.

Hierarchy (revised):

    Main menu:
      1. Run a single scenario (live)
      2. Pre-compute all scenarios (batch)  → also rebuilds the ML CSV
      3. View pre-computed scenario  → drill-down
      4. Rebuild ML feature snapshots (data/ml_training/recovery_stats.csv)
      5. Train ML model              [Phase 4 stub]
      6. Predict on a new scenario   [Phase 4 stub]
      7. Quit

    Options 1 and 3 use the same five-level drill-down:
      Level 1: Direction         (RNB / RSB / VEB / VWB)
      Level 2: Time slot         (10am / 1pm / 5pm)
      Level 3: Base horizon H    (600 / 900 / 1200 s)
      Level 4: Mode              (A: Closure only / B: Closure + Recovery)
      Level 5: Variant           Mode A → all_open / 1_lane_closed / 2_lanes_closed
                                Mode B (Regional) → 2_to_1, 2_to_0, 1_to_2
                                Mode B (Valley)   → 1_to_1

`while True` loop wraps the whole menu — closing a Matplotlib window
returns to the main menu, never exits Python.
"""

from __future__ import annotations

import json
from pathlib import Path

from simulation.networks    import DIRECTION_TO_NETWORK
from simulation.scenarios   import (
    TIME_SLOTS, BASE_HORIZONS, MODE_CLOSURE_ONLY, MODE_RECOVERY,
    REGIONAL_TRANSITIONS, VALLEY_TRANSITIONS, REGIONAL_CLOSURE_VARIANTS,
    VALLEY_CLOSURE_VARIANTS, transitions_for, closure_variants_for,
    run_one, run_all, output_paths, list_scenarios, count_scenarios,
    SCN_ROOT,
)
from simulation.visualizer  import visualise_scenario


DIRECTION_LABEL = {
    "regional_north": "Regional Road Northbound (RNB)",
    "regional_south": "Regional Road Southbound (RSB)",
    "valley_east":    "Valley Road Eastbound  (VEB)",
    "valley_west":    "Valley Road Westbound  (VWB)",
}

VARIANT_LABEL = {
    "all_open":        "All lanes open (baseline)",
    "1_lane_closed":   "1 lane closed",
    "2_lanes_closed":  "2 lanes closed",
    "full_closure":    "Full closure",
}

TRANSITION_LABEL = {
    "2closed_to_1closed":  "2 lanes closed → 1 lane closed",
    "2closed_to_0closed":  "2 lanes closed → 0 lanes closed (fully reopened)",
    "1closed_to_0closed":  "1 lane closed  → 0 lanes closed (fully reopened)",
}

SLOT_LABEL = {"10am": "10:00 AM", "1pm":  "1:00 PM", "5pm":  "5:00 PM"}

MODE_LABEL = {
    MODE_CLOSURE_ONLY: "(A) Closure preview only — runs to H",
    MODE_RECOVERY:     "(B) Closure + Recovery   — runs to 2 × H",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        return "q"


def _print_menu_header(title: str, subtitle: str | None = None) -> None:
    print()
    print("=" * 64)
    print(f"  {title}")
    if subtitle:
        print(f"  {subtitle}")
    print("=" * 64)


def _select_from(options: list[tuple[str, str]],
                 *, title: str, allow_back: bool = True) -> str | None:
    while True:
        _print_menu_header(title)
        for i, (_, label) in enumerate(options, 1):
            print(f"  {i}. {label}")
        if allow_back:
            print(f"  0. Back")
        choice = _ask("\n> ")
        if choice in ("0", "b", "back") and allow_back:
            return None
        try:
            idx = int(choice)
            if 1 <= idx <= len(options):
                return options[idx - 1][0]
        except ValueError:
            pass
        print("Invalid choice. Try again.")


def _drill_down(*, action_label: str, fixed_mode: str
                ) -> tuple[str, str, int, str, str] | None:
    """
    Walk the user through Direction → Slot → Horizon → (variant|transition).
    `fixed_mode` is set by the menu entry (closure-only or recovery), so the
    user is not asked to pick a mode — the sub-menu at the last level adapts
    to the road type automatically.
    """
    direction = _select_from(
        [(k, v) for k, v in DIRECTION_LABEL.items()],
        title=f"{action_label} — pick a road / direction",
    )
    if direction is None: return None

    slot = _select_from(
        [(s, SLOT_LABEL.get(s, s)) for s in TIME_SLOTS],
        title=f"{DIRECTION_LABEL[direction]} — pick a time slot",
    )
    if slot is None: return None

    base_h = _select_from(
        [(str(h), f"{h} s  ({h//60} min closure"
                  + (f" + {h//60} min recovery)" if fixed_mode == MODE_RECOVERY else ")"))
          for h in BASE_HORIZONS],
        title=f"{slot} — Select Simulation Horizon",
    )
    if base_h is None: return None
    base_h = int(base_h)

    if fixed_mode == MODE_CLOSURE_ONLY:
        variants = closure_variants_for(direction)
        vot = _select_from(
            [(v, VARIANT_LABEL.get(v, v)) for v in variants],
            title=f"H = {base_h}s · Static Closure — pick the closure variant",
        )
    else:   # MODE_RECOVERY — transitions filtered by road type
        transitions = transitions_for(direction)
        vot = _select_from(
            [(t_key, TRANSITION_LABEL.get(t_key, t_key)) for t_key in transitions],
            title=f"H = {base_h}s · Closure & Recovery — pick the lane transition",
        )
    if vot is None: return None

    return direction, slot, base_h, fixed_mode, vot


# ---------------------------------------------------------------------------
# Top-level menu actions
# ---------------------------------------------------------------------------

def _preview_single(fixed_mode: str, action_label: str) -> None:
    """Options [1]/[2] — single-scenario preview. Runs live (or reuses the
    cached playback files if present) and opens the GUI."""
    sel = _drill_down(action_label=action_label, fixed_mode=fixed_mode)
    if sel is None: return
    direction, slot, base_h, mode, vot = sel

    csv_p, json_p, _ = output_paths(direction, slot, base_h, mode, vot)
    if csv_p.exists() and json_p.exists():
        print(f"\n→ using cached playback for {direction}/{slot}/H={base_h}s/{vot}")
        summary = json.loads(json_p.read_text())
    else:
        print(f"\n→ running {direction} / {slot} / H={base_h}s / {mode} / {vot} live…")
        summary = run_one(direction, slot, base_h, mode, vot,
                          force=True, verbose=True)
    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    if _ask("\nOpen the visualisation now? [Y/n] ").lower() not in ("n", "no"):
        visualise_scenario(csv_p, direction=direction, slot=slot, variant=vot,
                           summary=summary)


def _action_standard_batch() -> None:
    print("\n  RUN STANDARD BATCH — 1× run, visual precomputations + 1 ML row set.")
    confirm = _ask("  Proceed? This (re)builds all playback files. [y/N] ")
    if not confirm.lower().startswith("y"):
        return
    force = _ask("  Force re-simulation of cached scenarios? [y/N] ").lower().startswith("y")
    from simulation.scenarios import run_standard_batch
    run_standard_batch(force=force, verbose=True)
    _ask("\nPress Enter to return to the main menu… ")


def _action_ml_factory_batch() -> None:
    print("\n  RUN ML FACTORY BATCH — 5× multi-seed runs.")
    print("  • Purges data/ml_training/ first (stale csv/parquet removed).")
    print("  • Runs the full matrix 5× with distinct seeds.")
    print("  • Runs 1–4 are headless; only the 5th writes JSON playback files.")
    print("  • Every run appends its T-snapshot with a run_id (1–5).")
    confirm = _ask("\n  This is heavy (5 × full matrix). Proceed? [y/N] ")
    if not confirm.lower().startswith("y"):
        return
    from simulation.scenarios import run_ml_factory_batch
    run_ml_factory_batch(verbose=True)
    _ask("\nPress Enter to return to the main menu… ")


def _action_train_ml() -> None:
    """[5] — run the full RandomForest training pipeline and show metrics."""
    from simulation.build_features import MASTER_CSV
    print("\n  TRAIN ML MODEL — RandomForest recovery-time regressor")
    if not MASTER_CSV.exists():
        print(f"  No training set at {MASTER_CSV}.")
        print("  Run Option [4] (ML Factory Batch) first to generate it.")
        _ask("\nPress Enter to return… ")
        return
    try:
        from ml.train_model import train
        result = train(verbose=True)
        print(f"\n  ✓ trained.  R²={result['r2']:.3f}  MAE={result['mae']:.1f}s  "
              f"→ {result['model_path']}")
    except Exception as exc:
        import traceback
        print(f"\n  Training failed: {exc}")
        traceback.print_exc()
    _ask("\nPress Enter to return… ")


def _action_predict() -> None:
    """[6] — Predictive Dashboard: pick params, run live inference."""
    from ml.predict import predict_recovery, MODEL_PATH

    if not MODEL_PATH.exists():
        print(f"\n  No trained model at {MODEL_PATH}.")
        print("  Run Option [5] (Train ML Model) first.")
        _ask("\nPress Enter to return… ")
        return

    print("\n  PREDICTIVE DASHBOARD — instantaneous recovery-time inference")
    direction = _select_from(
        [(k, v) for k, v in DIRECTION_LABEL.items()],
        title="Select Road & Direction",
    )
    if direction is None: return
    slot = _select_from(
        [(s, SLOT_LABEL.get(s, s)) for s in TIME_SLOTS],
        title="Select Time Slot",
    )
    if slot is None: return
    base_h = _select_from(
        [("600", "10 minutes (600 s)"), ("900", "15 minutes (900 s)"),
         ("1200", "20 minutes (1200 s)")],
        title="Select Incident Duration",
    )
    if base_h is None: return
    base_h = int(base_h)
    # Transitions valid for that road (recovery transitions only).
    transitions = transitions_for(direction)
    transition = _select_from(
        [(t, TRANSITION_LABEL.get(t, t)) for t in transitions],
        title="Select Transition Type",
    )
    if transition is None: return

    try:
        out = predict_recovery(direction, slot, base_h, transition)
    except Exception as exc:
        print(f"\n  Prediction failed: {exc}")
        _ask("\nPress Enter to return… ")
        return

    secs = out["predicted_seconds"]
    print()
    print("  " + "-" * 52)
    print(f"  [ML PREDICTION] Predicted Road Recovery Time: "
          f"{secs:.0f} seconds ({secs/60:.1f} minutes)")
    print("  " + "-" * 52)
    if out.get("baseline_source"):
        print(f"  baseline conditions from: {out['baseline_source']}")
    print(f"  inputs: occ_ratio={out['features']['spatial_occupancy_ratio']:.3f}  "
          f"hgv={out['features']['hgv_ratio']:.3f}  "
          f"backpressure={out['features']['backpressure_segment_count']}  "
          f"escape={out['features']['escape_flow_rate']:.2f}")
    _ask("\nPress Enter to return… ")


# ---------------------------------------------------------------------------
# Persistent menu loop
# ---------------------------------------------------------------------------

def run_menu() -> None:
    while True:
        print()
        print("=" * 57)
        print("      MALTA DIGITAL TWIN - BATCH SIMULATION SUITE        ")
        print("=" * 57)
        print(f"  cached playback scenarios: {count_scenarios()}")
        print("-" * 57)
        print("[1] View a Single Scenario (Static Closure Preview)")
        print("[2] View a Single Scenario (Closure & Opening Recovery Preview)")
        print("    --> sub-menu offers transitions (2->1, 2->0, 1->0) by road")
        print("[3] RUN STANDARD BATCH   (1x — visual precomputations only)")
        print("[4] RUN ML FACTORY BATCH (5x multi-seed — purges + 5x ML data)")
        print("[5] TRAIN ML MODEL       (RandomForest grid-search → .pkl)")
        print("[6] PREDICT RECOVERY     (live inference dashboard)")
        print("[7] Quit")
        print("=" * 57)
        choice = _ask("\n> ")

        if choice == "1":
            _preview_single(MODE_CLOSURE_ONLY,
                            "Static Closure Preview")
        elif choice == "2":
            _preview_single(MODE_RECOVERY,
                            "Closure & Opening Recovery Preview")
        elif choice == "3":
            _action_standard_batch()
        elif choice == "4":
            _action_ml_factory_batch()
        elif choice == "5":
            _action_train_ml()
        elif choice == "6":
            _action_predict()
        elif choice in ("7", "q", "quit", "exit"):
            print("\nBye.\n")
            return
        else:
            print("Invalid choice. Try again.")
