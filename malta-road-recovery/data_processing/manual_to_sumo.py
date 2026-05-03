"""
data_processing/manual_to_sumo.py
=================================
Manual Ground-Truth → SUMO demand pipeline.

Phase 1 input is a human-filled CSV, one file per segment × direction.
Directions live in subfolders under data/raw/manual_counts/:

    data/raw/manual_counts/regional_north/<slot>.csv
    data/raw/manual_counts/regional_south/<slot>.csv
    data/raw/manual_counts/valley_east/<slot>.csv
    data/raw/manual_counts/valley_west/<slot>.csv

CSV schema  (one row per 1-minute bin — already aggregated by the observer):

    timestamp_bin        int   — the 1-minute bin index (0, 1, 2, …)
    car_count            int   — cars that crossed the line in the bin
    motorcycle_count     int   — motorcycles in the bin (defaults to 0 if absent)
    heavy_vehicle_count  int   — buses + trucks combined in the bin
    avg_car_speed_kph    float — mean car speed observed for the bin
    incident_status      str   — clear | slow | blocked | incident

PCE weights (Passenger Car Equivalents):
    CAR_PCE        = 1.0
    MOTORCYCLE_PCE = 0.5
    HEAVY_PCE      = 2.5

Outputs (per <direction>/<slot>.csv):

    data/processed/aggregated/<direction>/<slot>.csv
    simulation/demand/<direction>_<slot>.rou.xml      (vTypes + chained route)
    simulation/demand/<direction>_<slot>.flow.xml     (per-bin flows + merge bg)

The route edges come from config/roads.yaml (sumo_edges.<direction>.edge_sequence)
so multi-way mainline chains (e.g. regional_north_A → regional_north_B →
regional_north_escape) are written as a single space-separated route string.

Usage
-----
    python data_processing/manual_to_sumo.py \
        --input data/raw/manual_counts/regional_north/10am.csv

    python data_processing/manual_to_sumo.py --all
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT      = Path(__file__).resolve().parent.parent
MANUAL_IN_DIR     = PROJECT_ROOT / "data" / "raw" / "manual_counts"
AGGREGATED_DIR    = PROJECT_ROOT / "data" / "processed" / "aggregated"
DEMAND_DIR        = PROJECT_ROOT / "simulation" / "demand"
CONFIG_ROADS      = PROJECT_ROOT / "config" / "roads.yaml"

AGGREGATED_DIR.mkdir(parents=True, exist_ok=True)
DEMAND_DIR.mkdir(parents=True, exist_ok=True)

DIRECTIONS = ("regional_north", "regional_south", "valley_east", "valley_west")

DIRECTION_TO_SEGMENT = {
    "regional_north": "regional_road",
    "regional_south": "regional_road",
    "valley_east":    "valley_road",
    "valley_west":    "valley_road",
}

REQUIRED_COLUMNS = (
    "timestamp_bin",
    "car_count",
    "heavy_vehicle_count",
    "avg_car_speed_kph",
    "incident_status",
)
# motorcycle_count is optional — silently defaults to 0 when absent.

VALID_INCIDENT_STATUSES = {"clear", "slow", "blocked", "incident"}

# Passenger Car Equivalents
CAR_PCE        = 1.0
MOTORCYCLE_PCE = 0.5
HEAVY_PCE      = 2.5

# SUMO vType definitions for the generated .rou.xml.
# Colors are RGB (0–1 floats, comma-separated):
#   car        → yellow  (1,1,0)
#   motorcycle → blue    (0,0,1)
#   heavy      → red     (1,0,0)
SUMO_VTYPES: dict[str, dict[str, str]] = {
    "car":        {"vClass": "passenger",   "accel": "2.6", "decel": "4.5", "sigma": "0.5",
                   "length": "4.5",  "maxSpeed": "36.11", "guiShape": "passenger",
                   "color":  "1,1,0"},
    "motorcycle": {"vClass": "motorcycle",  "accel": "4.0", "decel": "6.0", "sigma": "0.5",
                   "length": "2.2",  "maxSpeed": "36.11", "guiShape": "motorcycle",
                   "width":  "0.8",  "color": "0,0,1"},
    "heavy":      {"vClass": "truck",       "accel": "1.3", "decel": "4.0", "sigma": "0.5",
                   "length": "9.0",  "maxSpeed": "25.00", "guiShape": "truck",
                   "color":  "1,0,0"},
}

BIN_SECONDS = 60

FALLBACK_SPEED_KPH = 50.0

# Way IDs at or above this prefix are placeholders — see roads.yaml docs.
PLACEHOLDER_PREFIX = 999000000


# ---------------------------------------------------------------------------
# CONFIG HELPERS
# ---------------------------------------------------------------------------

def _load_roads_cfg() -> dict:
    with open(CONFIG_ROADS) as f:
        return yaml.safe_load(f)["segments"]


def _resolve_topology(direction: str, roads_cfg: dict) -> dict:
    """
    Return a routing plan for *direction*, derived from roads.yaml. Two shapes:

      simple:
          {"type": "simple",
           "routes": [{"id": "main", "edges": [<edge>], "share": 1.0}]}

      merge_split:
          {"type": "merge_split",
           "routes": [
              {"id": "via_inflow_a", "edges": ["flyover_s","trunk_s","tunnel_s"], "share": 0.49},
              {"id": "via_inflow_a_div", "edges": ["flyover_s","trunk_s","escape_s"], "share": 0.21},
              {"id": "via_inflow_b", "edges": ["underpass_s","trunk_s","tunnel_s"], "share": 0.21},
              {"id": "via_inflow_b_div", "edges": ["underpass_s","trunk_s","escape_s"], "share": 0.09},
           ]}

    The four-route product uses inflow_a_share × outflow_main_share for each leg
    so the trunk total still equals the manual CSV count after split routing.
    Falls back to a synthetic simple route if the YAML is incomplete.
    """
    seg_name = DIRECTION_TO_SEGMENT.get(direction)
    if not seg_name or seg_name not in roads_cfg:
        return {"type": "simple",
                "routes": [{"id": "main",
                            "edges": [f"{direction}_in", f"{direction}_out"],
                            "share": 1.0}]}

    seg_cfg = roads_cfg[seg_name]
    dir_cfg = seg_cfg.get("directions", {}).get(direction, {})
    topo    = dir_cfg.get("topology", {})
    ttype   = topo.get("type", "simple")

    if ttype == "simple":
        edge = topo.get("mainline", {}).get("label", direction)
        return {"type": "simple",
                "routes": [{"id": "main", "edges": [edge], "share": 1.0}]}

    if ttype == "merge_split":
        flyover    = topo["inflow_a"]["label"]
        underpass  = topo["inflow_b"]["label"]
        trunk      = topo["trunk"]["label"]
        out_main   = topo["outflow_main"]["label"]
        out_div    = topo["outflow_div"]["label"]

        share_cfg = seg_cfg.get("merge_split_demand", {})
        inflow_a_share     = float(share_cfg.get("inflow_a_share",     0.70))
        outflow_main_share = float(share_cfg.get("outflow_main_share", 0.70))

        return {
            "type": "merge_split",
            "routes": [
                {"id": "via_a_main",
                 "edges": [flyover, trunk, out_main],
                 "share": inflow_a_share * outflow_main_share},
                {"id": "via_a_div",
                 "edges": [flyover, trunk, out_div],
                 "share": inflow_a_share * (1 - outflow_main_share)},
                {"id": "via_b_main",
                 "edges": [underpass, trunk, out_main],
                 "share": (1 - inflow_a_share) * outflow_main_share},
                {"id": "via_b_div",
                 "edges": [underpass, trunk, out_div],
                 "share": (1 - inflow_a_share) * (1 - outflow_main_share)},
            ],
        }

    # Unknown topology — degrade gracefully.
    return {"type": "simple",
            "routes": [{"id": "main",
                        "edges": [f"{direction}_in", f"{direction}_out"],
                        "share": 1.0}]}


def _resolve_merge_ways(direction: str, roads_cfg: dict) -> list[dict]:
    """
    Return non-placeholder merge_ways for the segment that contains *direction*.
    Placeholder ways (way_id >= 999000000) are skipped — the user hasn't
    supplied real OSM IDs for them yet.
    """
    seg = DIRECTION_TO_SEGMENT.get(direction)
    if not seg or seg not in roads_cfg:
        return []
    return [
        m for m in roads_cfg[seg].get("merge_ways", [])
        if m.get("way_id", 0) < PLACEHOLDER_PREFIX
    ]


# ---------------------------------------------------------------------------
# VALIDATION + ENRICHMENT
# ---------------------------------------------------------------------------

def validate_and_enrich(df: pd.DataFrame, *, direction: str, slot: str) -> pd.DataFrame:
    """
    Validate the manual CSV columns and enrich with calibration-ready columns.
    Returns a sorted DataFrame with one row per 1-minute bin.

    motorcycle_count is optional — the column is created and defaulted to 0
    if absent so older CSVs continue to work.
    """
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Manual CSV {direction}/{slot} is missing required columns: "
            f"{sorted(missing)}.  Expected: {list(REQUIRED_COLUMNS)}"
        )

    keep = list(REQUIRED_COLUMNS) + ["motorcycle_count"]
    df = df[[c for c in keep if c in df.columns]].copy()
    if "motorcycle_count" not in df.columns:
        df["motorcycle_count"] = 0

    # Drop entirely blank rows BEFORE coercion (catches Excel trailing rows).
    df = df.dropna(how="all").reset_index(drop=True)

    df["timestamp_bin"]       = pd.to_numeric(df["timestamp_bin"],       errors="coerce").astype("Int64")
    df["car_count"]           = pd.to_numeric(df["car_count"],           errors="coerce").fillna(0).astype(int)
    df["motorcycle_count"]    = pd.to_numeric(df["motorcycle_count"],    errors="coerce").fillna(0).astype(int)
    df["heavy_vehicle_count"] = pd.to_numeric(df["heavy_vehicle_count"], errors="coerce").fillna(0).astype(int)
    df["avg_car_speed_kph"]   = pd.to_numeric(df["avg_car_speed_kph"],   errors="coerce")
    df["incident_status"]     = df["incident_status"].astype(str).str.strip().str.lower()

    # Drop rows with no valid timestamp_bin — these are usually trailing blanks.
    bad_bins = df[df["timestamp_bin"].isna()]
    if not bad_bins.empty:
        print(
            f"  [manual] {direction}/{slot}: dropping {len(bad_bins)} row(s) with "
            f"no timestamp_bin (likely trailing blank rows): {bad_bins.index.tolist()}"
        )
        df = df.dropna(subset=["timestamp_bin"])

    bad_statuses = sorted(set(df["incident_status"]) - VALID_INCIDENT_STATUSES)
    if bad_statuses:
        raise ValueError(
            f"Manual CSV {direction}/{slot} has unknown incident_status values: {bad_statuses}. "
            f"Allowed: {sorted(VALID_INCIDENT_STATUSES)}"
        )

    df = df.sort_values("timestamp_bin").reset_index(drop=True)

    df.insert(0, "direction", direction)
    df.insert(1, "slot",      slot)

    df["timestamp_s"]   = df["timestamp_bin"].astype(int) * BIN_SECONDS
    df["vehicle_count"] = df["car_count"] + df["motorcycle_count"] + df["heavy_vehicle_count"]

    df["pce_total"] = (
        df["car_count"]            * CAR_PCE
        + df["motorcycle_count"]   * MOTORCYCLE_PCE
        + df["heavy_vehicle_count"] * HEAVY_PCE
    ).round(2)

    df["flow_rate_veh_hr"] = df["vehicle_count"] * (3600 // BIN_SECONDS)
    df["flow_rate_pce_hr"] = df["pce_total"]     * (3600 // BIN_SECONDS)

    df["density_veh_km"] = np.where(
        df["avg_car_speed_kph"] > 0,
        df["flow_rate_veh_hr"] / df["avg_car_speed_kph"],
        np.nan,
    ).round(4)

    df["density_pce_km"] = np.where(
        df["avg_car_speed_kph"] > 0,
        df["flow_rate_pce_hr"] / df["avg_car_speed_kph"],
        np.nan,
    ).round(4)

    df["incident_flag"] = (df["incident_status"] != "clear").astype(int)

    return df


# ---------------------------------------------------------------------------
# SUMO DEMAND GENERATION
# ---------------------------------------------------------------------------

def _prettify(elem: ET.Element) -> str:
    rough = ET.tostring(elem, "utf-8")
    return minidom.parseString(rough).toprettyxml(indent="  ")


def _add_vtypes(root: ET.Element) -> None:
    """
    Append the three vehicle-type definitions (car / motorcycle / heavy) to *root*.
    Each vType carries a color so the SUMO-GUI shows traffic mix at a glance:
        car=yellow  motorcycle=blue  heavy=red
    """
    for name, attrs in SUMO_VTYPES.items():
        ET.SubElement(root, "vType", id=name, **attrs)


def _add_routes(root: ET.Element, routes: list[dict]) -> None:
    """
    Append one <route> element per routing option in *routes*.
    Each route's `edges` is the chained edge list; SUMO traverses them in order.

    For a merge_split topology this writes four routes (via flyover/underpass
    × via tunnel/escape) — vehicles are distributed across them in
    write_flow_xml() according to each route's `share`.
    """
    for r in routes:
        ET.SubElement(root, "route", id=r["id"], edges=" ".join(r["edges"]))


def write_rou_xml(
    df_bins: pd.DataFrame,
    out_path: Path,
    topology: dict,
) -> Path:
    """
    Writes the .rou.xml with vehicle-type definitions and one <route> per
    topology variant. For merge_split this is four routes; for simple it is one.
    """
    root = ET.Element("routes")
    _add_vtypes(root)
    _add_routes(root, topology["routes"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_prettify(root))
    return out_path


def _split_count(count: int, shares: list[float]) -> list[int]:
    """
    Split *count* across N routes by fractional *shares* with no rounding loss.
    Uses largest-remainder rounding so the per-route ints sum exactly to count.
    """
    if not shares or count == 0:
        return [0] * len(shares)
    raw   = [count * s for s in shares]
    base  = [int(x) for x in raw]
    rem   = count - sum(base)
    # distribute the remainder to routes with the largest fractional remainders
    if rem > 0:
        order = sorted(range(len(shares)), key=lambda i: raw[i] - base[i], reverse=True)
        for i in order[:rem]:
            base[i] += 1
    return base


def write_flow_xml(
    df_bins: pd.DataFrame,
    out_path: Path,
    topology: dict,
    *,
    merge_ways: list[dict] | None = None,
    duration_s: int = 600,
) -> Path:
    """
    One `<flow>` per (route × bin × vehicle class). For merge_split topologies
    the per-bin count is split across the four routes by their `share`, so the
    sum equals the manual CSV count and the trunk total is preserved.

    Background merge flows (`merge_ways`) get one constant `<flow>` covering
    the whole simulation duration on a route consisting only of that single
    edge label.
    """
    pce_weights = {"car": CAR_PCE, "motorcycle": MOTORCYCLE_PCE, "heavy": HEAVY_PCE}

    root = ET.Element("routes")
    _add_vtypes(root)
    _add_routes(root, topology["routes"])

    routes = topology["routes"]
    shares = [r["share"] for r in routes]

    # ── Per-bin, per-class flows split across all routes ──────────────────
    for _, row in df_bins.iterrows():
        begin = int(row["timestamp_s"])
        end   = begin + BIN_SECONDS
        if int(row["vehicle_count"]) == 0:
            continue

        speed_kph = (
            float(row["avg_car_speed_kph"])
            if pd.notna(row["avg_car_speed_kph"]) and row["avg_car_speed_kph"] > 0
            else FALLBACK_SPEED_KPH
        )
        depart_speed = f"{speed_kph / 3.6:.2f}"

        for cls, count_col in (
            ("car",        "car_count"),
            ("motorcycle", "motorcycle_count"),
            ("heavy",      "heavy_vehicle_count"),
        ):
            cls_count = int(row[count_col])
            if cls_count == 0:
                continue

            per_route_counts = _split_count(cls_count, shares)
            for r, sub_count in zip(routes, per_route_counts):
                if sub_count == 0:
                    continue
                vehs_per_hr = sub_count * (3600 // BIN_SECONDS)
                pce_per_hr  = round(vehs_per_hr * pce_weights[cls], 1)

                root.append(ET.Comment(
                    f" bin {begin//BIN_SECONDS}: {cls} {sub_count} veh on {r['id']} "
                    f"→ {vehs_per_hr} veh/hr  ({pce_per_hr} PCE/hr) "
                ))
                ET.SubElement(
                    root, "flow",
                    id          = f"flow_{cls}_{r['id']}_{begin}",
                    type        = cls,
                    route       = r["id"],
                    begin       = str(begin),
                    end         = str(end),
                    vehsPerHour = str(vehs_per_hr),
                    departSpeed = depart_speed,
                )

    # ── Background merge flows (constant across full sim duration) ─────────
    for merge in (merge_ways or []):
        label  = merge["label"]
        veh_hr = merge.get("background_flow_veh_hr", 0)
        if veh_hr <= 0:
            continue
        merge_route_id = f"route_{label}"
        ET.SubElement(root, "route", id=merge_route_id, edges=label)
        root.append(ET.Comment(
            f" background merge flow: {label}  {veh_hr} veh/hr  (constant) "
        ))
        ET.SubElement(
            root, "flow",
            id          = f"flow_merge_{label}",
            type        = "car",
            route       = merge_route_id,
            begin       = "0",
            end         = str(duration_s),
            vehsPerHour = str(veh_hr),
            departSpeed = "max",
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_prettify(root))
    return out_path


# ---------------------------------------------------------------------------
# ORCHESTRATION
# ---------------------------------------------------------------------------

def _parse_direction_and_slot(input_path: Path) -> tuple[str, str]:
    """Resolve (direction, slot) from data/raw/manual_counts/<direction>/<slot>.csv."""
    direction = input_path.parent.name
    slot      = input_path.stem
    if direction not in DIRECTIONS:
        raise ValueError(
            f"Parent folder '{direction}' is not a valid direction. "
            f"Expected one of: {list(DIRECTIONS)}"
        )
    return direction, slot


def process_manual_file(input_path: Path) -> dict[str, Path]:
    """
    Full manual-entry pipeline for a single <direction>/<slot>.csv.
    Returns the paths of every file written.
    """
    direction, slot = _parse_direction_and_slot(input_path)

    df_raw  = pd.read_csv(input_path)
    df_bins = validate_and_enrich(df_raw, direction=direction, slot=slot)

    aggregated_out = AGGREGATED_DIR / direction / f"{slot}.csv"
    aggregated_out.parent.mkdir(parents=True, exist_ok=True)
    df_bins.to_csv(aggregated_out, index=False)

    try:
        roads_cfg = _load_roads_cfg()
    except Exception:
        roads_cfg = {}

    topology   = _resolve_topology(direction, roads_cfg)
    merge_ways = _resolve_merge_ways(direction, roads_cfg)

    rou_out  = DEMAND_DIR / f"{direction}_{slot}.rou.xml"
    flow_out = DEMAND_DIR / f"{direction}_{slot}.flow.xml"

    write_rou_xml (df_bins, rou_out,  topology)
    write_flow_xml(df_bins, flow_out, topology, merge_ways=merge_ways)

    mean_speed = float(df_bins["avg_car_speed_kph"].replace(0, np.nan).mean(skipna=True) or 0)
    print(f"\n[manual] {direction} / {slot}")
    print("─" * 56)
    print(f"  Bins                       : {len(df_bins)}")
    print(f"  Total vehicles             : {int(df_bins['vehicle_count'].sum())}")
    print(f"  Cars / motos / heavy       : {int(df_bins['car_count'].sum())} / {int(df_bins['motorcycle_count'].sum())} / {int(df_bins['heavy_vehicle_count'].sum())}")
    print(f"  Total PCE                  : {round(float(df_bins['pce_total'].sum()), 1)}")
    print(f"  Mean bin speed (kph)       : {round(mean_speed, 2)}")
    print(f"  Bins flagged incident      : {int(df_bins['incident_flag'].sum())}")
    print(f"  Topology                   : {topology['type']}")
    for r in topology["routes"]:
        print(f"    route {r['id']:<14} share={r['share']:.2f}  edges={' → '.join(r['edges'])}")
    if merge_ways:
        merge_summary = ", ".join(
            f"{m['label']}={m['background_flow_veh_hr']}vph" for m in merge_ways
        )
        print(f"  Merge background flows     : {merge_summary}")
    print(f"  Aggregated CSV  → {aggregated_out}")
    print(f"  SUMO .rou.xml   → {rou_out}")
    print(f"  SUMO .flow.xml  → {flow_out}")

    return {"aggregated": aggregated_out, "rou": rou_out, "flow": flow_out}


def discover_manual_csvs() -> list[Path]:
    out = []
    for d in DIRECTIONS:
        out.extend(sorted((MANUAL_IN_DIR / d).glob("*.csv")))
    return out


def process_all() -> list[dict[str, Path]]:
    csvs = discover_manual_csvs()
    if not csvs:
        sys.exit(
            f"CRITICAL: No manual CSVs found under {MANUAL_IN_DIR}/<direction>/. "
            f"Expected direction folders: {list(DIRECTIONS)}"
        )
    return [process_manual_file(p) for p in csvs]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manual count CSV → aggregated bins + SUMO .rou.xml / .flow.xml",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=Path, help="Single manual CSV to process")
    src.add_argument("--all", action="store_true",
                     help=f"Process every CSV under {MANUAL_IN_DIR.relative_to(PROJECT_ROOT)}/<direction>/")
    args = parser.parse_args()

    if args.all:
        process_all()
        return

    if not args.input.exists():
        sys.exit(f"CRITICAL: Input file not found: {args.input}")
    process_manual_file(args.input)


if __name__ == "__main__":
    main()
