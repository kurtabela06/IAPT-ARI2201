"""
simulation/build_features.py
============================
ML Feature Extractor — produces `data/ml_training/master_traffic_features.csv`.

Each scenario contributes ONE training row, snapshotted at T = base_horizon_s
(the moment of reopening for recovery scenarios; the end of the closure
phase for static-closure scenarios).

This module exposes building blocks used by the two batch modes in
`simulation/scenarios.py`:

  • purge_ml_dir()            — wipe stale csv/parquet (Prompt 1)
  • snapshot_row(...)         — build one contract-compliant feature dict
  • append_rows(...)          — append rows to the master CSV
  • build_from_cache(...)     — (re)build the CSV from the JSON cache

ML CONTRACT (Prompt 4 — adjusted per user: unrecovered → null target):

  Features (inputs):
    run_id                      int      seed/run index (1..5; 1 for Mode A)
    road_id                     string   direction (regional_north, …)
    time_slot                   string   10am / 1pm / 5pm
    transition_type             string   all_open | 1_lane_closed |
                                         2closed_to_1closed | …
    spatial_occupancy_ratio     float    network occupied_m ÷ network capacity_m @T
    hgv_ratio                   float    HGV ÷ (cars+HGV) on the network @T
    backpressure_segment_count  int      # of edge-cells in the "red" band (>80%) @T
    escape_flow_rate            float    veh/s exiting via escape edges @T

  Target (output):
    recovery_time_seconds       float    seconds from T until rolling V_avg ≥
                                         threshold. NULL if not reached within
                                         the simulation horizon (the downstream
                                         ML model treats these as censored).
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

from simulation.networks import build_network


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCN_ROOT     = PROJECT_ROOT / "data" / "synthetic" / "scenarios"
ML_DIR       = PROJECT_ROOT / "data" / "ml_training"
MASTER_CSV   = ML_DIR / "master_traffic_features.csv"

RED_RATIO_THRESHOLD = 0.80   # "backpressure"/jam band, matches the visualizer

CONTRACT_COLUMNS = [
    "run_id",
    "road_id",
    "time_slot",
    "incident_duration",
    "transition_type",
    "spatial_occupancy_ratio",
    "hgv_ratio",
    "backpressure_segment_count",
    "escape_flow_rate",
    "recovery_time_seconds",
]


# ---------------------------------------------------------------------------
# Prompt 1 — directory purge
# ---------------------------------------------------------------------------

def purge_ml_dir() -> None:
    """
    Safely purge stale ML training artefacts. Deletes every .csv / .parquet
    inside data/ml_training/ and recreates a clean directory. Does NOT touch
    data/synthetic/scenarios/ (the simulator's JSON playback files) — those
    are a separate concern (file isolation requirement).
    """
    if ML_DIR.exists():
        for f in ML_DIR.iterdir():
            if f.suffix in (".csv", ".parquet"):
                f.unlink()
    ML_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Feature derivations
# ---------------------------------------------------------------------------

def _network_capacity_m(network_key: str, edge_df_at_t: pd.DataFrame) -> float:
    """Σ (length_m × original lanes) across all edges — the denominator for
    spatial_occupancy_ratio. Uses the static graph, not effective lanes,
    so the ratio reflects demand pressure on the *full* network."""
    g = build_network(network_key)
    return float(sum(d["length_m"] * d["lanes"] for _u, _v, d in g.edges(data=True)))


def _spatial_occupancy_ratio(network_key: str, edge_df_at_t: pd.DataFrame) -> float:
    occ = float(edge_df_at_t["occupied_m"].sum())
    cap = _network_capacity_m(network_key, edge_df_at_t)
    return round(occ / cap, 5) if cap > 0 else 0.0


def _hgv_ratio(edge_df_at_t: pd.DataFrame) -> float:
    cars = float(edge_df_at_t["cars_on_edge"].sum())
    hgvs = float(edge_df_at_t["hgv_on_edge"].sum())
    total = cars + hgvs
    return round(hgvs / total, 5) if total > 0 else 0.0


def _backpressure_segment_count(network_key: str,
                                particles_df_at_t: pd.DataFrame,
                                edge_df_at_t: pd.DataFrame) -> int:
    """
    Count edge-cells whose occupancy ratio is in the "red" band (≥ 0.80).
    This is the spatial extent of the jam at the snapshot, expressed as a
    discrete segment count (each cell ≈ 100 m).
    """
    if particles_df_at_t.empty:
        return 0
    g = build_network(network_key)
    edge_len = {d["label"]: float(d["length_m"]) for _u, _v, d in g.edges(data=True)}
    edge_lanes_orig = {d["label"]: int(d["lanes"]) for _u, _v, d in g.edges(data=True)}
    eff_lanes = {}
    for _, r in edge_df_at_t.iterrows():
        eff_lanes[r["edge"]] = int(r["effective_lanes"])

    LEN_BY_CLASS = {"car": 5.0, "hgv": 8.0, "motorcycle": 0.0}
    sub = particles_df_at_t.copy()
    sub["occ_m"] = sub.apply(
        lambda r: float(r["count"]) * LEN_BY_CLASS.get(str(r["cls"]), 0.0), axis=1
    )
    red = 0
    for (edge, cell_idx), grp in sub.groupby(["edge", "cell_idx"]):
        L = edge_len.get(edge)
        if L is None:
            continue
        # n_cells: re-derive from cell index span / 100m target
        n_cells = max(int(sub.loc[sub["edge"] == edge, "cell_idx"].max()) + 1,
                      max(1, int(round(L / 100.0))))
        cell_len = L / n_cells
        lanes = eff_lanes.get(edge, edge_lanes_orig.get(edge, 1))
        if lanes <= 0:
            lanes = edge_lanes_orig.get(edge, 1)
        cell_cap = cell_len * lanes
        ratio = grp["occ_m"].sum() / cell_cap if cell_cap > 0 else 1.0
        if ratio >= RED_RATIO_THRESHOLD:
            red += 1
    return int(red)


def _escape_flow_rate(network_key: str, edge_df_at_t: pd.DataFrame) -> float:
    """
    veh/s currently flowing on the escape edges. Approximated as
    (vehicles on escape edges) ÷ (escape dwell time = length / v_free).
    """
    g = build_network(network_key)
    escape_labels = g.graph.get("escape_edges", [])
    if not escape_labels:
        return 0.0
    rate = 0.0
    for _u, _v, d in g.edges(data=True):
        if d["label"] not in escape_labels:
            continue
        row = edge_df_at_t.loc[edge_df_at_t["edge"] == d["label"]]
        if row.empty:
            continue
        n = float(row["cars_on_edge"].iloc[0] + row["hgv_on_edge"].iloc[0]
                  + row["moto_on_edge"].iloc[0])
        dwell = max(d["length_m"] / max(d["v_free_mps"], 1.0), 1.0)
        rate += n / dwell
    return round(rate, 4)


# ---------------------------------------------------------------------------
# Snapshot row builder (contract-compliant)
# ---------------------------------------------------------------------------

def snapshot_row(summary: dict,
                 edge_df: pd.DataFrame,
                 particles_df: pd.DataFrame,
                 *,
                 run_id: int) -> dict:
    """Build one contract-compliant feature row from a completed scenario."""
    base_h = int(summary.get("base_horizon_s") or summary.get("horizon_s") or 600)
    snap_t = base_h
    edge_at_t = edge_df.loc[edge_df["time_s"] == snap_t]
    if edge_at_t.empty:
        snap_t = int(edge_df["time_s"].max())
        edge_at_t = edge_df.loc[edge_df["time_s"] == snap_t]
    part_at_t = (particles_df.loc[particles_df["time_s"] == snap_t]
                 if not particles_df.empty else pd.DataFrame())

    network_key = summary["network"]
    rec_time = summary.get("recovery_time_s")   # None ⇒ not recovered ⇒ null target

    return {
        "run_id":                     int(run_id),
        "road_id":                    summary["direction"],
        "time_slot":                  summary["slot"],
        # incident_duration = closure length in seconds = base_horizon.
        "incident_duration":          int(base_h),
        "transition_type":            summary.get("variant_or_transition",
                                                  summary.get("variant")),
        "spatial_occupancy_ratio":    _spatial_occupancy_ratio(network_key, edge_at_t),
        "hgv_ratio":                  _hgv_ratio(edge_at_t),
        "backpressure_segment_count": _backpressure_segment_count(
                                          network_key, part_at_t, edge_at_t),
        "escape_flow_rate":           _escape_flow_rate(network_key, edge_at_t),
        "recovery_time_seconds":      (float(rec_time) if rec_time is not None else None),
    }


def append_rows(rows: list[dict]) -> Path:
    """Append rows to the master CSV, creating it with the contract header
    if it does not yet exist. Columns are pinned to CONTRACT_COLUMNS order."""
    ML_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=CONTRACT_COLUMNS)
    header = not MASTER_CSV.exists()
    df.to_csv(MASTER_CSV, mode="a", header=header, index=False)
    return MASTER_CSV


# ---------------------------------------------------------------------------
# Rebuild the CSV from the existing JSON cache (Mode A convenience / Option 4)
# ---------------------------------------------------------------------------

def build_from_cache(*, run_id: int = 1, purge: bool = True) -> Path:
    """
    Rebuild master_traffic_features.csv from every cached scenario JSON,
    tagging all rows with the given run_id. Purges the ML dir first by
    default (so this never double-appends a stale CSV).
    """
    if purge:
        purge_ml_dir()

    rows = []
    json_paths = sorted(SCN_ROOT.glob("*/*.json"))
    print(f"[features] scanning {len(json_paths)} scenarios under {SCN_ROOT}")
    for jp in json_paths:
        csv_p = jp.with_suffix(".csv")
        pcsv_p = jp.with_name(jp.stem + ".particles.csv")
        if not csv_p.exists() or not pcsv_p.exists():
            continue
        try:
            summary = json.loads(jp.read_text())
            edge_df = pd.read_csv(csv_p)
            part_df = pd.read_csv(pcsv_p)
            rows.append(snapshot_row(summary, edge_df, part_df, run_id=run_id))
        except Exception as exc:
            print(f"  ✗ {jp.relative_to(PROJECT_ROOT)}: {exc}")

    append_rows(rows)
    _print_summary()
    return MASTER_CSV


def _print_summary() -> None:
    if not MASTER_CSV.exists():
        return
    df = pd.read_csv(MASTER_CSV)
    print(f"[features] master_traffic_features.csv now has {len(df)} rows")
    if "run_id" in df:
        print(f"[features]   run_id distribution: {dict(df['run_id'].value_counts().sort_index())}")
    recovered = int(df["recovery_time_seconds"].notna().sum())
    print(f"[features]   recovered (numeric target): {recovered} / {len(df)}; "
          f"{len(df) - recovered} null (not recovered within horizon)")


if __name__ == "__main__":
    build_from_cache(run_id=1, purge=True)
