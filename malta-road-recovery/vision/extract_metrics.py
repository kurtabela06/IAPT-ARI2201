"""
vision/extract_metrics.py
=========================
Post-processes YOLO output CSVs into calibration-ready traffic metrics
for downstream SUMO demand files.

Accepts EITHER:
  (a) Per-frame CSV  (*_frames.csv)  — raw row per detected vehicle per frame
  (b) Summary CSV    (*_<slot>.csv)  — pre-binned 1-minute aggregates

Output: a calibration metrics CSV with:
  - flow_rate_veh_hr       (q)
  - space_mean_speed_kmh   (u)  — harmonic mean of sampled speeds
  - density_veh_per_km     (k = q / u)
  - vehicle class ratios
  - peak_15min_flow        for SUMO demand scaling

Usage
-----
    # From a per-frame CSV:
    python vision/extract_metrics.py \
        --input  data/processed/yolo_output/regional_10am_frames.csv \
        --output data/processed/calibration/regional_10am_metrics.csv

    # From a summary CSV:
    python vision/extract_metrics.py \
        --input  data/processed/yolo_output/regional_10am.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CALIB_DIR    = PROJECT_ROOT / "data" / "processed" / "calibration"
CALIB_DIR.mkdir(parents=True, exist_ok=True)

# Class names we track ratios for
TRACKED_CLASSES = ("car", "motorcycle", "bus", "truck")


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def harmonic_mean(speeds: pd.Series) -> float:
    """
    Space mean speed = harmonic mean of time-mean speeds.
    Filters out zero / NaN values before computing.
    """
    valid = speeds[(speeds > 0) & speeds.notna()]
    if valid.empty:
        return 0.0
    return float(len(valid) / (1.0 / valid).sum())


# ---------------------------------------------------------------------------
# CONVERT PER-FRAME CSV → SUMMARY BINS
# ---------------------------------------------------------------------------

def aggregate_frames(df: pd.DataFrame, bin_seconds: int = 60) -> pd.DataFrame:
    """
    Take a per-frame DataFrame (frame, id, class, speed_kph, entry_time)
    and reduce it to 1-minute bins, counting only gate-crossing events
    (first appearance of each track ID in a given bin).

    Returns a summary DataFrame compatible with extract_metrics().
    """
    required = {"frame", "id", "class", "speed_kph", "entry_time"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Per-frame CSV is missing columns: {missing}")

    df = df.copy()
    df["speed_kph"] = pd.to_numeric(df["speed_kph"], errors="coerce")

    # Bin index per row
    df["bin_idx"] = (df["entry_time"] // bin_seconds).astype(int)

    # For counting, take the FIRST frame a track ID appears in each bin
    first = df.sort_values("frame").drop_duplicates(subset=["bin_idx", "id"])

    rows = []
    for bin_idx, grp in first.groupby("bin_idx"):
        ts    = int(bin_idx) * bin_seconds
        total = len(grp)
        avg_s = grp["speed_kph"].mean()
        row   = {
            "timestamp_s":   ts,
            "timestamp_min": round(ts / 60, 2),
            "vehicle_count": total,
            "avg_speed_kmh": round(avg_s, 2) if not np.isnan(avg_s) else 0.0,
        }
        for cls in TRACKED_CLASSES:
            row[f"count_{cls}"] = int((grp["class"] == cls).sum())
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# METRIC EXTRACTION
# ---------------------------------------------------------------------------

def extract_metrics(df: pd.DataFrame, segment: str = "", slot: str = "") -> pd.DataFrame:
    """
    Transform a 1-minute bin DataFrame into calibration metrics.

    Adds columns:
      flow_rate_veh_hr, density_veh_km, ratio_<class> for each class
    Renames avg_speed_kmh → time_mean_speed_kmh.

    Parameters
    ----------
    df      : summary DataFrame (from aggregate_frames or directly from summary CSV)
    segment : label to insert (can be inferred from the df if already present)
    slot    : label to insert
    """
    df = df.copy()

    # Ensure mandatory columns exist
    if "segment" not in df.columns:
        df.insert(0, "segment", segment)
    if "slot" not in df.columns:
        df.insert(1, "slot", slot)

    required = {"vehicle_count", "avg_speed_kmh"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Summary DataFrame is missing columns: {missing}")

    # Flow: veh/min → veh/hr
    df["flow_rate_veh_hr"] = df["vehicle_count"] * 60

    # Density: k = q / u  (veh/km)
    df["density_veh_km"] = np.where(
        df["avg_speed_kmh"] > 0,
        df["flow_rate_veh_hr"] / df["avg_speed_kmh"],
        np.nan,
    ).round(4)

    # Vehicle class ratios
    class_cols = [c for c in df.columns if c.startswith("count_")]
    if class_cols:
        total = df[class_cols].sum(axis=1).replace(0, np.nan)
        for col in class_cols:
            cls = col.replace("count_", "")
            df[f"ratio_{cls}"] = (df[col] / total).fillna(0).round(4)

    df = df.rename(columns={"avg_speed_kmh": "time_mean_speed_kmh"})

    return df


# ---------------------------------------------------------------------------
# SUMMARY STATS (for console print)
# ---------------------------------------------------------------------------

def summary_stats(df: pd.DataFrame) -> dict:
    seg  = df["segment"].iloc[0] if "segment" in df.columns else "?"
    slot = df["slot"].iloc[0]    if "slot"    in df.columns else "?"
    spd_col = "time_mean_speed_kmh" if "time_mean_speed_kmh" in df.columns else "avg_speed_kmh"

    return {
        "segment":             seg,
        "slot":                slot,
        "total_bins":          len(df),
        "total_vehicles":      int(df["vehicle_count"].sum()),
        "peak_flow_veh_hr":    int(df["flow_rate_veh_hr"].max()),
        "mean_flow_veh_hr":    round(float(df["flow_rate_veh_hr"].mean()), 1),
        "space_mean_spd_kmh":  round(harmonic_mean(df[spd_col]), 2),
        "mean_density_veh_km": round(float(df["density_veh_km"].mean(skipna=True)), 3),
    }


# ---------------------------------------------------------------------------
# BATCH HELPER (called from main.py)
# ---------------------------------------------------------------------------

def process_file(
    input_path: Path,
    output_path: Path | None = None,
    bin_seconds: int = 60,
) -> Path:
    """
    Process a single YOLO output file (per-frame or summary) and write the
    calibration metrics CSV.

    Returns the output path.
    """
    df = pd.read_csv(input_path)

    # Detect file type: per-frame files have a "frame" column
    if "frame" in df.columns:
        # Infer segment / slot from filename: <segment>_<slot>_frames.csv
        stem   = input_path.stem.replace("_frames", "")
        parts  = stem.split("_", 1)
        segment, slot = (parts[0], parts[1]) if len(parts) == 2 else ("", "")
        df_sum = aggregate_frames(df, bin_seconds)
        df_sum.insert(0, "segment", segment)
        df_sum.insert(1, "slot",    slot)
    else:
        # Already a summary — segment/slot columns should be present
        segment = df["segment"].iloc[0] if "segment" in df.columns else ""
        slot    = df["slot"].iloc[0]    if "slot"    in df.columns else ""
        df_sum  = df

    df_out = extract_metrics(df_sum, segment, slot)
    stats  = summary_stats(df_out)

    out = output_path or (CALIB_DIR / f"{input_path.stem}_metrics.csv")
    df_out.to_csv(out, index=False)

    print("\n[metrics] Calibration summary")
    print("─" * 40)
    for k, v in stats.items():
        print(f"  {k:<30} {v}")
    print(f"\n  Output → {out}\n")

    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract calibration metrics from YOLO output CSV"
    )
    parser.add_argument("--input",  required=True, type=Path,
                        help="Per-frame or summary CSV from detect.py")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output path (auto-named if omitted)")
    parser.add_argument("--bin-seconds", type=int, default=60,
                        help="Aggregation window in seconds (default: 60)")
    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(f"CRITICAL: Input file not found: {args.input}")

    process_file(args.input, args.output, args.bin_seconds)


if __name__ == "__main__":
    main()
