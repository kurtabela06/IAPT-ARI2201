"""
data_processing/extract_metrics.py
==================================
Post-processes the aggregated per-bin CSVs produced by `manual_to_sumo.py`
into calibration-ready traffic metrics for downstream SUMO scripts.

Input  : data/processed/aggregated/<direction>/<slot>.csv
         (one row per 1-minute bin, already validated)

Output : data/processed/calibration/<direction>_<slot>_metrics.csv
         (per-bin calibration columns — space-mean speed, flow, density,
          heavy-vehicle ratio, incident flag — ready for SUMO calibration)

No OpenCV, no YOLO, no video decoding — pure tabular work on the
manually-validated ground truth.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT   = Path(__file__).resolve().parent.parent
AGGREGATED_DIR = PROJECT_ROOT / "data" / "processed" / "aggregated"
CALIB_DIR      = PROJECT_ROOT / "data" / "processed" / "calibration"
CALIB_DIR.mkdir(parents=True, exist_ok=True)

DIRECTIONS = ("regional_north", "regional_south", "valley_east", "valley_west")

REQUIRED_COLUMNS = (
    "direction", "slot", "timestamp_bin", "timestamp_s",
    "car_count", "heavy_vehicle_count", "vehicle_count",
    "avg_speed_kph", "flow_rate_veh_hr", "density_veh_km",
    "incident_status", "incident_flag",
)


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def harmonic_mean(speeds: pd.Series) -> float:
    """Space-mean speed = harmonic mean of per-bin average speeds (>0 only)."""
    valid = speeds[(speeds > 0) & speeds.notna()]
    if valid.empty:
        return 0.0
    return float(len(valid) / (1.0 / valid).sum())


def extract_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich the aggregated DataFrame with per-bin calibration metrics.
    Assumes `df` came from `manual_to_sumo.validate_and_enrich`.
    """
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Aggregated CSV is missing columns: {sorted(missing)}. "
            "Did you run data_processing/manual_to_sumo.py first?"
        )

    df = df.copy()
    total = df["vehicle_count"].replace(0, np.nan)
    df["ratio_heavy"] = (df["heavy_vehicle_count"] / total).fillna(0).round(4)
    df["ratio_car"]   = (df["car_count"]           / total).fillna(0).round(4)
    df = df.rename(columns={"avg_speed_kph": "time_mean_speed_kph"})
    return df


def summary_stats(df: pd.DataFrame) -> dict:
    return {
        "direction":             df["direction"].iloc[0],
        "slot":                  df["slot"].iloc[0],
        "total_bins":            len(df),
        "total_vehicles":        int(df["vehicle_count"].sum()),
        "total_cars":            int(df["car_count"].sum()),
        "total_heavy":           int(df["heavy_vehicle_count"].sum()),
        "peak_flow_veh_hr":      int(df["flow_rate_veh_hr"].max()),
        "mean_flow_veh_hr":      round(float(df["flow_rate_veh_hr"].mean()), 1),
        "space_mean_speed_kph":  round(harmonic_mean(df["time_mean_speed_kph"]), 2),
        "mean_density_veh_km":   round(float(df["density_veh_km"].mean(skipna=True)), 3),
        "incident_bins":         int(df["incident_flag"].sum()),
    }


# ---------------------------------------------------------------------------
# FILE PROCESSING
# ---------------------------------------------------------------------------

def process_file(input_path: Path, output_path: Path | None = None) -> Path:
    df_in  = pd.read_csv(input_path)
    df_out = extract_metrics(df_in)
    stats  = summary_stats(df_out)

    direction = df_out["direction"].iloc[0]
    slot      = df_out["slot"].iloc[0]
    out = output_path or (CALIB_DIR / f"{direction}_{slot}_metrics.csv")
    df_out.to_csv(out, index=False)

    print("\n[metrics] Calibration summary")
    print("─" * 48)
    for k, v in stats.items():
        print(f"  {k:<24} {v}")
    print(f"\n  Output → {out}\n")

    return out


def process_aggregated() -> list[Path]:
    """Run `process_file` over every aggregated CSV in every direction folder."""
    outs = []
    for direction in DIRECTIONS:
        for csv_path in sorted((AGGREGATED_DIR / direction).glob("*.csv")):
            outs.append(process_file(csv_path))
    return outs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract calibration metrics from an aggregated per-bin CSV"
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=Path,
                     help="Aggregated CSV (data/processed/aggregated/<direction>/<slot>.csv)")
    src.add_argument("--all", action="store_true",
                     help="Process every aggregated CSV in every direction")

    parser.add_argument("--output", type=Path, default=None,
                        help="Output path (auto-named if omitted)")
    args = parser.parse_args()

    if args.all:
        outs = process_aggregated()
        if not outs:
            sys.exit(f"CRITICAL: No aggregated CSVs in {AGGREGATED_DIR}")
        return

    if not args.input.exists():
        sys.exit(f"CRITICAL: Input file not found: {args.input}")
    process_file(args.input, args.output)


if __name__ == "__main__":
    main()
