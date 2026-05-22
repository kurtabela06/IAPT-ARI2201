"""
ml/predict.py
=============
Real-time inference for the recovery-time RandomForest (Part 3).

Given a (road, slot, incident_duration, transition) selection, this:
  1. loads models/recovery_rf_model.pkl
  2. extracts the matching baseline initial conditions
     (spatial_occupancy_ratio, hgv_ratio, backpressure_segment_count,
      escape_flow_rate) from the precomputed master_traffic_features.csv,
      averaging across the 5 seed runs for that exact configuration
  3. assembles the feature row and returns the model's predicted
     recovery_time_seconds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_PATH = PROJECT_ROOT / "models" / "recovery_rf_model.pkl"
DATA_CSV   = PROJECT_ROOT / "data" / "ml_training" / "master_traffic_features.csv"

NUMERIC_BASELINE_COLS = [
    "spatial_occupancy_ratio",
    "hgv_ratio",
    "backpressure_segment_count",
    "escape_flow_rate",
]


def _baseline_conditions(direction: str, slot: str, base_h: int,
                         transition: str) -> tuple[dict, str]:
    """
    Pull the baseline traffic-state features for this configuration from the
    precomputed dataset, averaged across the seed runs. Falls back to the
    closest match (same road + transition) if the exact (slot, duration)
    combination is missing.
    """
    df = pd.read_csv(DATA_CSV)
    exact = df[(df.road_id == direction) & (df.time_slot == slot)
               & (df.incident_duration == base_h)
               & (df.transition_type == transition)]
    if not exact.empty:
        return ({c: float(exact[c].mean()) for c in NUMERIC_BASELINE_COLS},
                f"{len(exact)} run(s) matching {direction}/{slot}/{base_h}s/{transition}")
    # Fallback: same road + transition, any slot/duration.
    loose = df[(df.road_id == direction) & (df.transition_type == transition)]
    if not loose.empty:
        return ({c: float(loose[c].mean()) for c in NUMERIC_BASELINE_COLS},
                f"fallback: {len(loose)} run(s) for {direction}/{transition} (any slot/duration)")
    # Last resort: road-wide average.
    road = df[df.road_id == direction]
    if not road.empty:
        return ({c: float(road[c].mean()) for c in NUMERIC_BASELINE_COLS},
                f"fallback: road-wide average for {direction}")
    raise ValueError(f"No baseline data found for {direction} in {DATA_CSV}")


def predict_recovery(direction: str, slot: str, base_h: int,
                     transition: str) -> dict:
    """Run inference; returns {predicted_seconds, features, baseline_source}."""
    import joblib

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No model at {MODEL_PATH}. Train it first (CLI option [5])."
        )
    bundle = joblib.load(MODEL_PATH)
    pipeline = bundle["pipeline"]

    baseline, source = _baseline_conditions(direction, slot, base_h, transition)

    features = {
        "road_id":          direction,
        "time_slot":        slot,
        "transition_type":  transition,
        "incident_duration": float(base_h),
        **baseline,
    }
    # Order columns exactly as the pipeline expects.
    cat = bundle["categorical_features"]
    num = bundle["numeric_features"]
    X = pd.DataFrame([{**{c: features[c] for c in cat},
                       **{c: features[c] for c in num}}])
    pred = float(pipeline.predict(X)[0])
    pred = max(0.0, min(pred, 3600.0))   # clamp to the engine's recovery cap

    return {
        "predicted_seconds": pred,
        "features":          features,
        "baseline_source":   source,
    }


if __name__ == "__main__":
    # Smoke CLI: python ml/predict.py regional_south 5pm 1200 2closed_to_1closed
    args = sys.argv[1:]
    if len(args) != 4:
        print("usage: python ml/predict.py <road_id> <slot> <duration> <transition>")
        sys.exit(1)
    road, slot, dur, trans = args[0], args[1], int(args[2]), args[3]
    out = predict_recovery(road, slot, dur, trans)
    s = out["predicted_seconds"]
    print(f"[ML PREDICTION] Predicted Road Recovery Time: {s:.0f} seconds "
          f"({s/60:.1f} minutes)")
    print(f"  baseline: {out['baseline_source']}")
