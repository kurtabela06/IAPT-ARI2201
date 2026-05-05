"""
data_processing/manual_to_pulse.py
==================================
Manual ground-truth CSV → 1-minute demand-pulse CSV.

Phase 2 (NetworkX engine) only needs raw counts per minute. PCE / density /
flow-rate calculations have moved into the engine, where they are computed
from spatial occupancy on each graph edge.

Input  CSV (one row per 1-minute bin):

    timestamp_bin, car_count, motorcycle_count, heavy_vehicle_count,
    avg_car_speed_kph, incident_status

Output pulse CSV (also one row per bin):

    timestamp_bin, car, hgv, motorcycle, incident_status, observed_speed_kph

`observed_speed_kph` is kept as informational only — it's used by the
calibrator to decide whether the run looks plausible, but never consumed by
the engine itself.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT   = Path(__file__).resolve().parent.parent
MANUAL_IN_DIR  = PROJECT_ROOT / "data" / "raw" / "manual_counts"
PULSES_OUT_DIR = PROJECT_ROOT / "data" / "processed" / "pulses"

PULSES_OUT_DIR.mkdir(parents=True, exist_ok=True)

DIRECTIONS = ("regional_north", "regional_south", "valley_east", "valley_west")
VALID_INCIDENT = {"clear", "slow", "blocked", "incident"}

REQUIRED = ("timestamp_bin", "car_count", "heavy_vehicle_count",
            "avg_car_speed_kph", "incident_status")


def parse_direction_and_slot(input_path: Path) -> tuple[str, str]:
    direction = input_path.parent.name
    if direction not in DIRECTIONS:
        raise ValueError(f"Parent folder '{direction}' is not a valid direction. "
                         f"Expected one of: {list(DIRECTIONS)}")
    return direction, input_path.stem


def to_pulses(df_raw: pd.DataFrame, *, direction: str, slot: str) -> pd.DataFrame:
    """Validate the raw CSV and reshape into a per-bin pulse DataFrame."""
    missing = set(REQUIRED) - set(df_raw.columns)
    if missing:
        raise ValueError(
            f"Manual CSV {direction}/{slot} is missing columns: {sorted(missing)}. "
            f"Required: {list(REQUIRED)}"
        )

    keep = list(REQUIRED) + ["motorcycle_count"]
    df = df_raw[[c for c in keep if c in df_raw.columns]].copy()
    if "motorcycle_count" not in df.columns:
        df["motorcycle_count"] = 0

    df = df.dropna(how="all").reset_index(drop=True)

    df["timestamp_bin"]       = pd.to_numeric(df["timestamp_bin"],       errors="coerce").astype("Int64")
    df["car_count"]           = pd.to_numeric(df["car_count"],           errors="coerce").fillna(0).astype(int)
    df["motorcycle_count"]    = pd.to_numeric(df["motorcycle_count"],    errors="coerce").fillna(0).astype(int)
    df["heavy_vehicle_count"] = pd.to_numeric(df["heavy_vehicle_count"], errors="coerce").fillna(0).astype(int)
    df["avg_car_speed_kph"]   = pd.to_numeric(df["avg_car_speed_kph"],   errors="coerce")
    df["incident_status"]     = df["incident_status"].astype(str).str.strip().str.lower()

    bad = df[df["timestamp_bin"].isna()]
    if not bad.empty:
        print(f"  [pulses] {direction}/{slot}: dropping {len(bad)} blank row(s) "
              f"(rows {bad.index.tolist()})")
        df = df.dropna(subset=["timestamp_bin"])

    bad_status = sorted(set(df["incident_status"]) - VALID_INCIDENT)
    if bad_status:
        raise ValueError(
            f"Manual CSV {direction}/{slot} has unknown incident_status: {bad_status}. "
            f"Allowed: {sorted(VALID_INCIDENT)}"
        )

    df = df.sort_values("timestamp_bin").reset_index(drop=True)

    return pd.DataFrame({
        "timestamp_bin":      df["timestamp_bin"].astype(int),
        "car":                df["car_count"],
        "hgv":                df["heavy_vehicle_count"],
        "motorcycle":         df["motorcycle_count"],
        "incident_status":    df["incident_status"],
        "observed_speed_kph": df["avg_car_speed_kph"],
    })


def pulses_path(direction: str, slot: str) -> Path:
    return PULSES_OUT_DIR / direction / f"{slot}.csv"


def load_pulses(direction: str, slot: str) -> list[dict]:
    """Read the pulse CSV and return list[dict] suitable for engine.Scenario."""
    p = pulses_path(direction, slot)
    if not p.exists():
        raise FileNotFoundError(f"Pulse CSV not found: {p}.  Run process_manual_file first.")
    df = pd.read_csv(p)
    return df.to_dict("records")


def process_manual_file(input_path: Path) -> Path:
    direction, slot = parse_direction_and_slot(input_path)
    df_raw = pd.read_csv(input_path)
    pulses = to_pulses(df_raw, direction=direction, slot=slot)

    out = pulses_path(direction, slot)
    out.parent.mkdir(parents=True, exist_ok=True)
    pulses.to_csv(out, index=False)

    total = int(pulses["car"].sum() + pulses["hgv"].sum() + pulses["motorcycle"].sum())
    print(f"\n[pulses] {direction} / {slot}")
    print(f"  Bins                : {len(pulses)}")
    print(f"  Cars / HGV / motos  : {int(pulses['car'].sum())} / {int(pulses['hgv'].sum())} / {int(pulses['motorcycle'].sum())}")
    print(f"  Total vehicles      : {total}")
    print(f"  Bins with incident  : {(pulses['incident_status'] != 'clear').sum()}")
    print(f"  → {out}")
    return out


def discover_manual_csvs() -> list[Path]:
    out = []
    for d in DIRECTIONS:
        out.extend(sorted((MANUAL_IN_DIR / d).glob("*.csv")))
    return out


def process_all() -> list[Path]:
    csvs = discover_manual_csvs()
    if not csvs:
        sys.exit(f"CRITICAL: No manual CSVs under {MANUAL_IN_DIR}/<direction>/.")
    return [process_manual_file(p) for p in csvs]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manual count CSV → 1-minute demand-pulse CSV"
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=Path, help="Single manual CSV to process")
    src.add_argument("--all",   action="store_true", help="Process every direction × slot")
    args = parser.parse_args()

    if args.all:
        process_all()
        return
    if not args.input.exists():
        sys.exit(f"CRITICAL: Input file not found: {args.input}")
    process_manual_file(args.input)


if __name__ == "__main__":
    main()
