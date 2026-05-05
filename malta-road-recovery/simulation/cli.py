"""
simulation/cli.py
=================
Persistent drill-down menu for the Phase 2 NetworkX engine.

Layout (Q7 spec):

    Main menu:
      1. Run a single scenario (live)
      2. Pre-compute all scenarios (batch)
      3. View pre-computed scenario  → drill-down
      4. Train ML model on scenarios  (Phase 4 stub)
      5. Predict on a new scenario    (Phase 4 stub)
      6. Quit

    Option 3 drill-down:
      Level 1: Direction      (Regional North / South, Valley East / West)
      Level 2: Time slot      (12 canonical slots)
      Level 3: Variant        (regional → 3 options, valley → 2)

`while True` loop wraps the whole menu — closing a Matplotlib window
returns to the main menu, never exits Python.
"""

from __future__ import annotations

import json
from pathlib import Path

from simulation.networks    import DIRECTION_TO_NETWORK
from simulation.scenarios   import (
    TIME_SLOTS, run_one, run_all, output_paths,
    REGIONAL_VARIANTS, VALLEY_VARIANTS, variants_for, list_scenarios, SCN_DIR,
)
from simulation.visualizer  import visualise_scenario


DIRECTION_LABEL = {
    "regional_north": "Regional Road Northbound (RNB)",
    "regional_south": "Regional Road Southbound (RSB)",
    "valley_east":    "Valley Road Eastbound (VEB)",
    "valley_west":    "Valley Road Westbound (VWB)",
}

VARIANT_LABEL = {
    "all_open":        "All lanes open (baseline)",
    "1_lane_closed":   "1 lane closed",
    "2_lanes_closed":  "2 lanes closed",
    "full_closure":    "Full closure",
}

SLOT_LABEL = {
    "10am": "10:00 AM",
    "1pm":  "1:00 PM",
    "5pm":  "5:00 PM",
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
    print("=" * 60)
    print(f"  {title}")
    if subtitle:
        print(f"  {subtitle}")
    print("=" * 60)


def _select_from(options: list[tuple[str, str]],
                 *, title: str, allow_back: bool = True) -> str | None:
    """Return the user's chosen key or None if they pick 'back'."""
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


# ---------------------------------------------------------------------------
# Top-level menu actions
# ---------------------------------------------------------------------------

def _action_single_run() -> None:
    direction = _select_from(
        [(k, v) for k, v in DIRECTION_LABEL.items()],
        title="Single live run — pick a direction",
    )
    if direction is None:
        return
    slot = _select_from(
        [(s, SLOT_LABEL.get(s, s)) for s in TIME_SLOTS],
        title=f"{DIRECTION_LABEL[direction]} — pick a time slot",
    )
    if slot is None:
        return
    variants = variants_for(direction)
    variant = _select_from(
        [(v, VARIANT_LABEL.get(v, v)) for v in variants],
        title=f"{slot} — pick an incident variant",
    )
    if variant is None:
        return

    print(f"\n→ running {direction} / {slot} / {variant} live (force=True)…")
    summary = run_one(direction, slot, variant, force=True, verbose=True)
    print("\nSummary:")
    print(json.dumps(summary, indent=2))

    csv_p, _, _ = output_paths(direction, slot, variant)
    if _ask("\nOpen the visualisation now? [y/N] ").lower().startswith("y"):
        visualise_scenario(csv_p, direction=direction, slot=slot, variant=variant)


def _action_precompute_all() -> None:
    confirm = _ask(
        "Pre-compute the full 120-scenario grid? [y/N] (use --force to recompute existing) "
    )
    if not confirm.lower().startswith("y"):
        return
    force = _ask("Force recomputation of cached scenarios? [y/N] ").lower().startswith("y")
    summaries = run_all(force=force, verbose=True)
    print(f"\nDone. {len(summaries)} scenarios available in {SCN_DIR}.")
    _ask("\nPress Enter to return to the main menu… ")


def _action_view_scenario() -> None:
    direction = _select_from(
        [(k, v) for k, v in DIRECTION_LABEL.items()],
        title="View pre-computed scenario — pick a direction",
    )
    if direction is None:
        return
    slot = _select_from(
        [(s, SLOT_LABEL.get(s, s)) for s in TIME_SLOTS],
        title=f"{DIRECTION_LABEL[direction]} — pick a time slot",
    )
    if slot is None:
        return
    variants = variants_for(direction)
    variant = _select_from(
        [(v, VARIANT_LABEL.get(v, v)) for v in variants],
        title=f"{slot} — pick an incident variant",
    )
    if variant is None:
        return

    csv_p, json_p, _ = output_paths(direction, slot, variant)
    if not csv_p.exists() or not json_p.exists():
        print(f"\n  Scenario data not found: {csv_p.name}")
        print(f"  Please run Option 2 to pre-compute or select Option 1 for a live run.")
        _ask("\nPress Enter to return… ")
        return

    summary = json.loads(json_p.read_text())
    print("\n  Summary:")
    print(json.dumps(summary, indent=2))
    visualise_scenario(csv_p, direction=direction, slot=slot, variant=variant)


def _action_train_ml() -> None:
    print("\n  [Phase 4 stub] ML training not yet wired up.")
    print("  This option will train an XGBoost model on the 120-scenario "
          "telemetry once Phase 4 is implemented.")
    _ask("\nPress Enter to return… ")


def _action_predict() -> None:
    print("\n  [Phase 4 stub] Predictor not yet wired up.")
    _ask("\nPress Enter to return… ")


# ---------------------------------------------------------------------------
# Persistent menu loop
# ---------------------------------------------------------------------------

def _scenario_count() -> int:
    return len(list(SCN_DIR.glob("*.json")))


def run_menu() -> None:
    while True:
        _print_menu_header(
            "Malta Road Recovery — Phase 2 Engine",
            f"Cached scenarios: {_scenario_count()}",
        )
        print("  1. Run a single scenario (live)")
        print("  2. Pre-compute all scenarios (batch)")
        print("  3. View pre-computed scenario")
        print("  4. Train ML model on scenarios   [Phase 4 stub]")
        print("  5. Predict on a new scenario     [Phase 4 stub]")
        print("  6. Quit")
        choice = _ask("\n> ")

        if choice == "1":
            _action_single_run()
        elif choice == "2":
            _action_precompute_all()
        elif choice == "3":
            _action_view_scenario()
        elif choice == "4":
            _action_train_ml()
        elif choice == "5":
            _action_predict()
        elif choice in ("6", "q", "quit", "exit"):
            print("\nBye.\n")
            return
        else:
            print("Invalid choice. Try again.")
