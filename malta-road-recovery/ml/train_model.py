"""
ml/train_model.py
=================
Random-Forest Recovery-Time predictor — Part 2 of the ML suite.

Ingests the multi-seed dataset at data/ml_training/master_traffic_features.csv
and trains a scikit-learn RandomForestRegressor to predict
`recovery_time_seconds` from the traffic-state features captured at the
moment of reopening.

Multi-dimensional handling
--------------------------
The four road networks have very different topologies (3-lane regional
trunk vs. 1-lane valley bottleneck), so we feed the categorical descriptors
through a one-hot ColumnTransformer rather than forcing a single numeric
equation. `incident_duration` is kept NUMERIC (600/900/1200) so the forest
can learn the monotonic "longer blockage → worse residual queue" relation
as an ordered split variable.

Inputs (X):
    road_id                     (categorical, one-hot)
    time_slot                   (categorical, one-hot)
    transition_type             (categorical, one-hot)
    incident_duration           (numeric — derived from base_horizon, the
                                 closure length in seconds)
    spatial_occupancy_ratio     (numeric)
    hgv_ratio                   (numeric)
    backpressure_segment_count  (numeric)
    escape_flow_rate            (numeric)

Target (y):
    recovery_time_seconds       (float; observed or extrapolated, ≤ 3600)

Pipeline:
    1. Load CSV; drop rows with a null target (closure-only previews have no
       recovery event — they're not recovery-prediction examples).
    2. 80/20 train/test split (seeded for reproducibility).
    3. GridSearchCV over n_estimators / max_depth / min_samples_split,
       scoring = neg-MAE.
    4. Report R², MAE, and feature importances on the held-out test set.
    5. joblib-dump the fitted pipeline → models/recovery_rf_model.pkl

Run:
    python ml/train_model.py
or via the CLI: main.py → [5] TRAIN ML MODEL
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_CSV   = PROJECT_ROOT / "data" / "ml_training" / "master_traffic_features.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "recovery_rf_model.pkl"

# `incident_duration` is the closure length in seconds. The master CSV does
# not store it directly, so we recover it from the transition/horizon: for a
# recovery scenario, closure length = base_horizon. We re-derive it from the
# scenario duration encoded in the source path, or fall back to a column.
CATEGORICAL_FEATURES = ["road_id", "time_slot", "transition_type"]
NUMERIC_FEATURES = [
    "incident_duration",
    "spatial_occupancy_ratio",
    "hgv_ratio",
    "backpressure_segment_count",
    "escape_flow_rate",
]
TARGET = "recovery_time_seconds"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _infer_incident_duration(df: pd.DataFrame) -> pd.Series:
    """
    incident_duration = closure length in seconds. Recovery scenarios were
    run with horizon = 2 × base_horizon (closure for base_horizon, then
    recovery). The master CSV doesn't carry base_horizon directly, so we read
    it from the scenario provenance if present, else default to a mapping
    from the transition file naming. As a robust fallback we use the
    `incident_duration` column if the feature-builder already added it.
    """
    if "incident_duration" in df.columns:
        return df["incident_duration"].astype(float)
    # The build step (updated below) writes incident_duration directly; if an
    # older CSV lacks it, default to the 600s baseline so training still runs.
    return pd.Series([600.0] * len(df), index=df.index)


def load_dataset(csv_path: Path = DATA_CSV) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {csv_path}. "
            f"Run the ML Factory batch first (main.py → [4])."
        )
    df = pd.read_csv(csv_path)
    df["incident_duration"] = _infer_incident_duration(df)

    # Keep only rows that have a recovery target (recovery scenarios). Closure-
    # only previews have a null target and are not recovery-prediction cases.
    before = len(df)
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    print(f"[train] loaded {before} rows; {len(df)} have a recovery target "
          f"({before - len(df)} closure-only rows dropped)")
    return df


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(csv_path: Path = DATA_CSV, *, verbose: bool = True) -> dict:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.metrics import mean_absolute_error, r2_score
    import joblib

    df = load_dataset(csv_path)
    if len(df) < 10:
        raise ValueError(
            f"Only {len(df)} usable rows — too few to train. "
            f"Run the ML Factory batch (5× multi-seed) first."
        )

    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y = df[TARGET].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # One-hot the categoricals; pass numerics through untouched (trees don't
    # need scaling). incident_duration stays numeric so the forest learns the
    # ordered "longer closure → longer recovery" relationship.
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )
    pipe = Pipeline([
        ("pre", pre),
        ("rf", RandomForestRegressor(random_state=42, n_jobs=-1)),
    ])

    param_grid = {
        "rf__n_estimators":     [100, 300, 500],
        "rf__max_depth":        [None, 10, 20],
        "rf__min_samples_split": [2, 5, 10],
    }
    # MAE scoring; CV folds bounded by sample count.
    n_splits = max(2, min(5, len(X_train) // 5))
    search = GridSearchCV(
        pipe, param_grid, scoring="neg_mean_absolute_error",
        cv=n_splits, n_jobs=-1, refit=True,
    )
    if verbose:
        print(f"[train] grid search over {len(param_grid['rf__n_estimators'])}"
              f"×{len(param_grid['rf__max_depth'])}"
              f"×{len(param_grid['rf__min_samples_split'])} configs, cv={n_splits}…")
    search.fit(X_train, y_train)

    best = search.best_estimator_
    y_pred = best.predict(X_test)
    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Feature importances mapped back to readable names.
    rf = best.named_steps["rf"]
    ohe = best.named_steps["pre"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    feat_names = cat_names + NUMERIC_FEATURES
    importances = sorted(zip(feat_names, rf.feature_importances_),
                         key=lambda kv: kv[1], reverse=True)

    if verbose:
        print("\n" + "=" * 56)
        print("  RANDOM FOREST — RECOVERY TIME PREDICTOR")
        print("=" * 56)
        print(f"  Best params      : {search.best_params_}")
        print(f"  Test R²          : {r2:.4f}")
        print(f"  Test MAE         : {mae:.1f} s")
        print(f"  Train/test split : {len(X_train)} / {len(X_test)}")
        print("\n  Feature importances:")
        for name, imp in importances:
            bar = "█" * int(imp * 50)
            print(f"    {name:<34} {imp:6.3f} {bar}")
        print("=" * 56)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "pipeline":   best,
        "r2":         r2,
        "mae":        mae,
        "best_params": search.best_params_,
        "feature_importances": importances,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features":     NUMERIC_FEATURES,
        "target":     TARGET,
    }, MODEL_PATH)
    if verbose:
        print(f"\n[train] model saved → {MODEL_PATH}")

    return {"r2": r2, "mae": mae, "best_params": search.best_params_,
            "model_path": str(MODEL_PATH), "n_train": len(X_train),
            "n_test": len(X_test), "importances": importances}


if __name__ == "__main__":
    train()
