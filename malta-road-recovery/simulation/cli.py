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


def _drill_down(*, action_label: str) -> tuple[str, str, int, str, str] | None:
    """
    Walk the user through the 5-level drill-down. Returns
    (direction, slot, base_h, mode, variant_or_transition) or None if backed out.
    """
    direction = _select_from(
        [(k, v) for k, v in DIRECTION_LABEL.items()],
        title=f"{action_label} — Level 1: pick a direction",
    )
    if direction is None: return None

    slot = _select_from(
        [(s, SLOT_LABEL.get(s, s)) for s in TIME_SLOTS],
        title=f"{DIRECTION_LABEL[direction]} — Level 2: pick a time slot",
    )
    if slot is None: return None

    base_h = _select_from(
        [(str(h), f"H = {h} s  ({h//60} minutes)") for h in BASE_HORIZONS],
        title=f"{slot} — Level 3: pick the base horizon H",
    )
    if base_h is None: return None
    base_h = int(base_h)

    mode = _select_from(
        [(MODE_CLOSURE_ONLY, MODE_LABEL[MODE_CLOSURE_ONLY]),
         (MODE_RECOVERY,     MODE_LABEL[MODE_RECOVERY])],
        title=f"H = {base_h}s — Level 4: pick a mode",
    )
    if mode is None: return None

    if mode == MODE_CLOSURE_ONLY:
        variants = closure_variants_for(direction)
        vot = _select_from(
            [(v, VARIANT_LABEL.get(v, v)) for v in variants],
            title=f"H = {base_h}s · Closure only — Level 5: pick an incident variant",
        )
    else:   # MODE_RECOVERY
        transitions = transitions_for(direction)
        vot = _select_from(
            [(t_key, TRANSITION_LABEL.get(t_key, t_key)) for t_key in transitions],
            title=f"H = {base_h}s · Closure + Recovery — Level 5: pick a transition",
        )
    if vot is None: return None

    return direction, slot, base_h, mode, vot


# ---------------------------------------------------------------------------
# Top-level menu actions
# ---------------------------------------------------------------------------

def _action_single_run() -> None:
    sel = _drill_down(action_label="Single live run")
    if sel is None: return
    direction, slot, base_h, mode, vot = sel

    print(f"\n→ running {direction} / {slot} / H={base_h}s / {mode} / {vot} live (force=True)…")
    summary = run_one(direction, slot, base_h, mode, vot, force=True, verbose=True)
    print("\nSummary:")
    print(json.dumps(summary, indent=2))

    csv_p, _, _ = output_paths(direction, slot, base_h, mode, vot)
    if _ask("\nOpen the visualisation now? [y/N] ").lower().startswith("y"):
        visualise_scenario(csv_p, direction=direction, slot=slot, variant=vot,
                           summary=summary)


def _action_precompute_all() -> None:
    confirm = _ask(
        "Pre-compute the full scenario grid (many combinations)? [y/N] "
    )
    if not confirm.lower().startswith("y"):
        return
    force = _ask("Force recomputation of cached scenarios? [y/N] ").lower().startswith("y")
    summaries = run_all(force=force, verbose=True)
    print(f"\nDone. {len(summaries)} scenarios written under {SCN_ROOT}.")
    _ask("\nPress Enter to return to the main menu… ")


def _action_view_scenario() -> None:
    sel = _drill_down(action_label="View pre-computed scenario")
    if sel is None: return
    direction, slot, base_h, mode, vot = sel

    csv_p, json_p, _ = output_paths(direction, slot, base_h, mode, vot)
    if not csv_p.exists() or not json_p.exists():
        print(f"\n  Scenario data not found: {csv_p}")
        print(f"  Please run Option 2 to pre-compute or select Option 1 for a live run.")
        _ask("\nPress Enter to return… ")
        return

    summary = json.loads(json_p.read_text())
    print("\n  Summary:")
    print(json.dumps(summary, indent=2))
    visualise_scenario(csv_p, direction=direction, slot=slot, variant=vot,
                       summary=summary)


def _action_build_features() -> None:
    print("\n  Rebuilding ML feature snapshots from the scenario cache…")
    try:
        from simulation.build_features import build_all as _build_all
        out = _build_all()
        print(f"\n  Wrote → {out}")
    except Exception as exc:
        print(f"\n  Build failed: {exc}")
    _ask("\nPress Enter to return… ")


def _action_train_ml() -> None:
    print("\n  [Phase 4 stub] ML training not yet wired up.")
    print("  data/ml_training/recovery_stats.csv is ready for downstream "
          "XGBoost / RF / linear baselines.")
    _ask("\nPress Enter to return… ")


def _action_predict() -> None:
    print("\n  [Phase 4 stub] Predictor not yet wired up.")
    _ask("\nPress Enter to return… ")


# ---------------------------------------------------------------------------
# Persistent menu loop
# ---------------------------------------------------------------------------

def run_menu() -> None:
    while True:
        _print_menu_header(
            "Malta Road Recovery — Phase 2 Engine",
            f"Cached scenarios: {count_scenarios()}",
        )
        print("  1. Run a single scenario (live)")
        print("  2. Pre-compute all scenarios (batch)")
        print("  3. View pre-computed scenario")
        print("  4. Rebuild ML feature snapshots (recovery_stats.csv)")
        print("  5. Train ML model on scenarios   [Phase 4 stub]")
        print("  6. Predict on a new scenario     [Phase 4 stub]")
        print("  7. Quit")
        choice = _ask("\n> ")

        if choice == "1":
            _action_single_run()
        elif choice == "2":
            _action_precompute_all()
        elif choice == "3":
            _action_view_scenario()
        elif choice == "4":
            _action_build_features()
        elif choice == "5":
            _action_train_ml()
        elif choice == "6":
            _action_predict()
        elif choice in ("7", "q", "quit", "exit"):
            print("\nBye.\n")
            return
        else:
            print("Invalid choice. Try again.")
