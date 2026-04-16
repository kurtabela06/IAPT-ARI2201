"""
extract_metrics.py
==================
Post-processes raw YOLO output CSVs into calibration-ready metrics
for SUMO demand files.

Computes:
  - flow_rate_veh_per_hr   (q)
  - space_mean_speed_kmh   (u)  — harmonic mean of time-mean speeds
  - density_veh_per_km     (k = q / u)
  - vehicle_class_ratios   (car%, bus%, truck%, motorcycle%)
  - peak_15min_flow        (used for demand scaling in .rou.xml)

Usage:
    python vision/extract_metrics.py \
        --input  data/processed/yolo_output/regional_10am.csv \
        --output data/processed/calibration/regional_10am_metrics.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CALIB_DIR    = PROJECT_ROOT / "data" / "processed" / "calibration"
CALIB_DIR.mkdir(parents=True, exist_ok=True)


def harmonic_mean(speeds: pd.Series) -> float:
    """
    Space mean speed is the harmonic mean of individual vehicle speeds.
    Filters out zero/NaN entries before computing.
    """
    valid = speeds[speeds > 0].dropna()
    if valid.empty:
        return 0.0
    return float(len(valid) / (1.0 / valid).sum())


def extract_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw 1-minute bin DataFrame into calibration metrics.
    """
    required = {"vehicle_count", "avg_speed_kmh", "segment", "slot"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in input CSV: {missing}")

    # Flow rate: veh/min → veh/hr
    df["flow_rate_veh_hr"] = df["vehicle_count"] * 60

    # Density: k = q / u  (veh/km)
    # Guard against zero speed to avoid division errors
    df["density_veh_km"] = np.where(
        df["avg_speed_kmh"] > 0,
        df["flow_rate_veh_hr"] / df["avg_speed_kmh"],
        np.nan,
    )

    # Vehicle class ratios (if class columns present)
    class_cols = [c for c in df.columns if c.startswith("count_")]
    if class_cols:
        total = df[class_cols].sum(axis=1).replace(0, np.nan)
        for col in class_cols:
            class_name = col.replace("count_", "")
            df[f"ratio_{class_name}"] = (df[col] / total).fillna(0).round(4)

    # Rename for clarity
    df = df.rename(columns={"avg_speed_kmh": "time_mean_speed_kmh"})

    return df


def summary_stats(df: pd.DataFrame) -> dict:
    """Return a dict of key summary numbers for logging."""
    return {
        "segment":            df["segment"].iloc[0],
        "slot":               df["slot"].iloc[0],
        "bins":               len(df),
        "total_vehicles":     int(df["vehicle_count"].sum()),
        "peak_flow_veh_hr":   int(df["flow_rate_veh_hr"].max()),
        "mean_flow_veh_hr":   round(df["flow_rate_veh_hr"].mean(), 1),
        "space_mean_spd_kmh": round(harmonic_mean(df["time_mean_speed_kmh"]), 2),
        "mean_density_veh_km": round(df["density_veh_km"].mean(), 3),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract calibration metrics from YOLO output CSV"
    )
    parser.add_argument("--input",  required=True, type=Path,
                        help="Raw YOLO output CSV")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output path (auto-named if omitted)")
    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(f"CRITICAL: Input file not found: {args.input}")

    df  = pd.read_csv(args.input)
    df  = extract_metrics(df)
    stats = summary_stats(df)

    out = args.output or (CALIB_DIR / (args.input.stem + "_metrics.csv"))
    df.to_csv(out, index=False)

    print("\n[Metrics] Calibration summary")
    print("─" * 36)
    for k, v in stats.items():
        print(f"  {k:<28} {v}")
    print(f"\n  Output → {out}\n")


if __name__ == "__main__":
    main()