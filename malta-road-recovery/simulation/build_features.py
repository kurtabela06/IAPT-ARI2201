"""
simulation/build_features.py
============================
ML Snapshot Extractor — Prompt 2 deliverable.

For every cached scenario in `data/synthetic/scenarios/<H>s/<scenario>.{csv,json}`,
this script extracts a single feature row capturing the network state at
T = base_horizon_s (the moment of reopening for Mode B scenarios; the end
of the closure phase for Mode A). That row, together with the label
`time_to_recovery`, becomes one training example for the recovery-predictor
ML model.

Snapshot columns (per scenario):

    road_id                 direction (e.g. regional_north)
    network_key             RNB / RSB / VEB / VWB
    slot                    10am / 1pm / 5pm
    base_horizon_s          600 / 900 / 1200
    mode                    closure_only / recovery
    transition_type         e.g. 2closed_to_1closed, all_open, 1_lane_closed
    snapshot_time_s         T (= base_horizon_s)
    total_occupied_meters   Σ occupied_m across all edges at T
    queue_length_m          contiguous red-cluster length backward from the
                            diverge end of the closeable-edge's *predecessor*
                            (the trunk that backs up during closure)
    current_inflow_rate     veh/s being injected at T (per-second average)
    hgv_percentage          HGV share of (cars + HGV) on the network at T
    transition_type         redundant with above — kept for readability
    recovery_threshold_kph  the network's V_avg target
    time_to_recovery        seconds from T until rolling V_avg ≥ threshold
                            (null if not recovered within horizon)
    recovery_triggered_s    T value used by the engine (None for Mode A)

Output: data/ml_training/recovery_stats.csv

Run:
    python simulation/build_features.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# Allow running as a script from the project root.
PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

from simulation.networks import build_network, DIRECTION_TO_NETWORK


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCN_ROOT     = PROJECT_ROOT / "data" / "synthetic" / "scenarios"
OUT_DIR      = PROJECT_ROOT / "data" / "ml_training"
OUT_PATH     = OUT_DIR / "recovery_stats.csv"


# Red-cluster threshold for queue-length derivation (matches the visualizer
# 80 % "jam" colour band).
RED_RATIO_THRESHOLD = 0.80


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_summary_and_csv(json_p: Path) -> Optional[tuple[dict, pd.DataFrame, pd.DataFrame]]:
    csv_p = json_p.with_suffix(".csv")
    particles_p = json_p.with_name(json_p.stem + ".particles.csv")
    if not csv_p.exists() or not particles_p.exists():
        return None
    return (
        json.loads(json_p.read_text()),
        pd.read_csv(csv_p),
        pd.read_csv(particles_p),
    )


def _queue_length_m(edge_df_at_t: pd.DataFrame,
                    particles_df_at_t: pd.DataFrame,
                    network_key: str) -> float:
    """
    Approximate the contiguous "red" queue length, measured backward from the
    diverge node along the trunk-pre-diverge chain that backs up during a
    Main-Way closure.

    For Regional (RSB/RNB): the trunk edge is the immediate predecessor of
        the diverge node (mergeway_S / mergeway_N_b). Per-cell red-cluster
        run-length backwards from cell N-1.
    For Valley East (VEB): the predecessor of `branch_E` is `valley_E`
        (single edge). Apply the same backward run-length rule.
    For Valley West (VWB): the predecessor of `junction_valley_W` is
        `valley_W_pre` (the long Valley Road inflow). Backward run-length.

    Returns 0.0 if no red cells are present at the snapshot time.
    """
    # Identify the queueing edge.
    queueing_edge_by_network = {
        "RSB": "mergeway_S",
        "RNB": "mergeway_N_b",
        "VEB": "valley_E",
        "VWB": "valley_W_pre",
    }
    qe = queueing_edge_by_network.get(network_key)
    if qe is None:
        return 0.0

    # Use the particles_df to compute per-cell occupancy at the snapshot.
    # particles_df: time_s, edge, cls, count, pos_m, pos_frac, cell_idx
    g = build_network(network_key)
    edge_data = next((d for u, v, d in g.edges(data=True) if d["label"] == qe), None)
    if edge_data is None:
        return 0.0
    edge_length = float(edge_data["length_m"])

    # Per-cell capacity using effective_lanes at the snapshot.
    eff_lanes = int(edge_df_at_t.loc[edge_df_at_t["edge"] == qe, "effective_lanes"].iloc[0]) \
                if not edge_df_at_t.loc[edge_df_at_t["edge"] == qe].empty else int(edge_data["lanes"])
    if eff_lanes <= 0:
        # Closeable edge isn't this one (it's typically the post-diverge Main Way),
        # so trunk eff_lanes should normally be its original value.
        eff_lanes = int(edge_data["lanes"])

    # Group particles on this edge by cell, summing occupancy metres
    # (car=5m, hgv=8m; motorcycles excluded).
    LEN_BY_CLASS = {"car": 5.0, "hgv": 8.0, "motorcycle": 0.0}
    sub = particles_df_at_t.loc[particles_df_at_t["edge"] == qe].copy()
    if sub.empty:
        return 0.0
    sub["occupancy_m"] = sub.apply(
        lambda r: float(r["count"]) * LEN_BY_CLASS.get(str(r["cls"]), 0.0), axis=1
    )
    cell_occ = sub.groupby("cell_idx")["occupancy_m"].sum().to_dict()

    # The engine cells the edge uniformly; recover n_cells from particles or
    # by re-deriving via cell_length config (default 100 m target).
    n_cells = (int(sub["cell_idx"].max()) + 1) if not sub.empty else 1
    # If no particles exist on some cells, n_cells from particles may be
    # an under-count. Use cell-row equivalent: round(length/100) ≥ 1.
    n_cells = max(n_cells, max(1, int(round(edge_length / 100.0))))
    cell_len = edge_length / n_cells
    cell_max_m = cell_len * eff_lanes

    # Backward run-length from cell (n_cells-1) while ratio ≥ RED threshold.
    queue = 0.0
    for ci in range(n_cells - 1, -1, -1):
        occ = cell_occ.get(ci, 0.0)
        ratio = occ / cell_max_m if cell_max_m > 0 else 1.0
        if ratio >= RED_RATIO_THRESHOLD:
            queue += cell_len
        else:
            break
    return round(queue, 2)


def _total_occupied_metres(edge_df_at_t: pd.DataFrame) -> float:
    """Σ occupied_m across every edge at the snapshot tick."""
    return float(edge_df_at_t["occupied_m"].sum())


def _hgv_percentage(edge_df_at_t: pd.DataFrame) -> float:
    """HGV share of (cars + HGV) on the network at T. Motorcycles excluded."""
    cars = float(edge_df_at_t["cars_on_edge"].sum())
    hgvs = float(edge_df_at_t["hgv_on_edge"].sum())
    total = cars + hgvs
    return round(100.0 * hgvs / total, 2) if total > 0 else 0.0


def _current_inflow_rate(edge_df_at_t: pd.DataFrame, snapshot_t: int, bin_secs: int = 60) -> float:
    """
    Average vehicles-per-second arriving at the network's inflow edges
    around the snapshot tick. Computed as (vehicles entered on inflow edges
    in the last `bin_secs` seconds) / bin_secs.
    """
    # Simpler proxy: use the difference in total vehicle count
    # (cars + hgv + moto) on inflow edges between t-bin and t. But we only
    # have one tick's edge_df. Fall back to looking at cars+hgv+moto on
    # inflow edges right now and dividing by their travelling time —
    # equivalent to instantaneous occupancy / dwell-time.
    inflows = edge_df_at_t[edge_df_at_t["edge"].str.contains(
        "src|inflow|valley_W_pre|valley_E|mergeway_N_a|flyover|underpass",
        regex=True, case=False
    )]
    if inflows.empty:
        return 0.0
    n = float((inflows["cars_on_edge"] + inflows["hgv_on_edge"] + inflows["moto_on_edge"]).sum())
    # Average dwell on inflow edges ≈ length / v_free. Use 100m / 22m/s ≈ 4.5s.
    return round(n / 4.5, 4)


# ---------------------------------------------------------------------------
# Snapshot row builder
# ---------------------------------------------------------------------------

def build_row(json_p: Path) -> Optional[dict]:
    out = _read_summary_and_csv(json_p)
    if out is None:
        return None
    summary, edge_df, particles_df = out

    base_h = int(summary.get("base_horizon_s") or summary.get("horizon_s") or 600)
    snapshot_t = base_h

    edge_at_t = edge_df.loc[edge_df["time_s"] == snapshot_t]
    if edge_at_t.empty:
        # Fallback: last frame in the CSV
        snapshot_t = int(edge_df["time_s"].max())
        edge_at_t = edge_df.loc[edge_df["time_s"] == snapshot_t]
    particles_at_t = particles_df.loc[particles_df["time_s"] == snapshot_t] \
                     if not particles_df.empty else pd.DataFrame()

    network_key = summary["network"]
    direction   = summary["direction"]

    return {
        "road_id":               direction,
        "network_key":           network_key,
        "slot":                  summary["slot"],
        "base_horizon_s":        base_h,
        "horizon_s":             summary.get("horizon_s"),
        "mode":                  summary.get("mode", "closure_only"),
        "transition_type":       summary.get("variant_or_transition", summary.get("variant")),
        "snapshot_time_s":       snapshot_t,
        "total_occupied_meters": round(_total_occupied_metres(edge_at_t), 2),
        "queue_length_m":        _queue_length_m(edge_at_t, particles_at_t, network_key),
        "current_inflow_rate":   _current_inflow_rate(edge_at_t, snapshot_t),
        "hgv_percentage":        _hgv_percentage(edge_at_t),
        "recovery_threshold_kph": summary.get("recovery_threshold_kph"),
        "recovery_triggered_s":  summary.get("recovery_triggered_s"),
        "time_to_recovery":      summary.get("recovery_time_s"),  # None if not recovered
        # Provenance
        "scenario_csv":          str(json_p.with_suffix(".csv").relative_to(PROJECT_ROOT)),
    }


def build_all() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    json_paths = sorted(SCN_ROOT.glob("*/*.json"))
    print(f"[features] scanning {len(json_paths)} scenarios under {SCN_ROOT}")
    for jp in json_paths:
        try:
            row = build_row(jp)
        except Exception as exc:
            print(f"  ✗ {jp.relative_to(PROJECT_ROOT)}: {exc}")
            continue
        if row is not None:
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)
    print(f"[features] wrote {len(df)} snapshot rows → {OUT_PATH}")

    # Quick summary
    if not df.empty:
        recovered = int(df["time_to_recovery"].notna().sum())
        print(f"[features]   recovered (time_to_recovery known): {recovered} / {len(df)}")
        print(f"[features]   road_id distribution: "
              f"{dict(df['road_id'].value_counts())}")
        print(f"[features]   base_horizon distribution: "
              f"{dict(df['base_horizon_s'].value_counts().sort_index())}")
    return OUT_PATH


if __name__ == "__main__":
    build_all()
