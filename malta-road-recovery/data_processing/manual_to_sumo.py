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

    timestamp_bin        int or label — the 1-minute bin index (0, 1, 2, …)
    car_count            int          — cars that crossed the counting line in the bin
    motorcycle_count     int          — motorcycles in the bin (defaults to 0 if column absent)
    heavy_vehicle_count  int          — buses + trucks combined in the bin
    avg_car_speed_kph    float        — mean car speed observed for the bin
    incident_status      str          — one of: clear | slow | blocked | incident

PCE weights (Passenger Car Equivalents) are applied when computing
effective road occupancy:
    CAR_PCE        = 1.0
    MOTORCYCLE_PCE = 0.5
    HEAVY_PCE      = 2.5

Outputs (per <direction>/<slot>.csv):

    data/processed/aggregated/<direction>/<slot>.csv          # validated + enriched bins
    simulation/demand/<direction>_<slot>.rou.xml              # SUMO routes (vType defs + route)
    simulation/demand/<direction>_<slot>.flow.xml             # SUMO flows (per-bin, class-split)

The aggregated CSV keeps the raw columns plus derived calibration columns
(flow_rate_veh_hr, density_veh_km, incident flags) used by the downstream
Phase 2/3 scripts.

Usage
-----
    # Single file
    python data_processing/manual_to_sumo.py \
        --input data/raw/manual_counts/regional_north/10am.csv

    # Every CSV in every direction
    python data_processing/manual_to_sumo.py --all

    # Override SUMO edge IDs (defaults: <direction>_in / <direction>_out)
    python data_processing/manual_to_sumo.py \
        --input data/raw/manual_counts/valley_east/5pm.csv \
        --edge-from valley_A --edge-to valley_B
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

import numpy as np
import pandas as pd

PROJECT_ROOT      = Path(__file__).resolve().parent.parent
MANUAL_IN_DIR     = PROJECT_ROOT / "data" / "raw" / "manual_counts"
AGGREGATED_DIR    = PROJECT_ROOT / "data" / "processed" / "aggregated"
DEMAND_DIR        = PROJECT_ROOT / "simulation" / "demand"

AGGREGATED_DIR.mkdir(parents=True, exist_ok=True)
DEMAND_DIR.mkdir(parents=True, exist_ok=True)

DIRECTIONS = ("regional_north", "regional_south", "valley_east", "valley_west")

REQUIRED_COLUMNS = (
    "timestamp_bin",
    "car_count",
    "heavy_vehicle_count",
    "avg_car_speed_kph",
    "incident_status",
)
# motorcycle_count is optional — absent column silently defaults to 0.

VALID_INCIDENT_STATUSES = {"clear", "slow", "blocked", "incident"}

# Passenger Car Equivalents — used to compute effective road occupancy.
CAR_PCE        = 1.0
MOTORCYCLE_PCE = 0.5
HEAVY_PCE      = 2.5

# SUMO vType defaults feed the generated .rou.xml so the simulation runs
# against the manual demand without a separate type file.
SUMO_VTYPES: dict[str, dict[str, str]] = {
    "car":        {"vClass": "passenger",   "accel": "2.6", "decel": "4.5", "sigma": "0.5",
                   "length": "4.5",  "maxSpeed": "36.11", "guiShape": "passenger"},
    "motorcycle": {"vClass": "motorcycle",  "accel": "4.0", "decel": "6.0", "sigma": "0.5",
                   "length": "2.2",  "maxSpeed": "36.11", "guiShape": "motorcycle",
                   "width": "0.8"},
    "heavy":      {"vClass": "truck",       "accel": "1.3", "decel": "4.0", "sigma": "0.5",
                   "length": "9.0",  "maxSpeed": "25.00", "guiShape": "truck"},
}

BIN_SECONDS = 60

# Used by the SUMO flow file only — never written to the aggregated CSV.
FALLBACK_SPEED_KPH = 50.0


# ---------------------------------------------------------------------------
# VALIDATION + ENRICHMENT
# ---------------------------------------------------------------------------

def validate_and_enrich(df: pd.DataFrame, *, direction: str, slot: str) -> pd.DataFrame:
    """
    Validate the manual CSV columns and enrich it with calibration-ready
    derived columns. Returns a sorted DataFrame with one row per 1-minute bin.

    motorcycle_count is optional — the column is created and defaulted to 0
    if absent, so older CSVs without it continue to work without changes.
    """
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Manual CSV {direction}/{slot} is missing required columns: "
            f"{sorted(missing)}.  Expected: {list(REQUIRED_COLUMNS)}"
        )

    # Pull only the columns we care about; inject motorcycle_count if absent.
    keep = list(REQUIRED_COLUMNS) + ["motorcycle_count"]
    df = df[[c for c in keep if c in df.columns]].copy()
    if "motorcycle_count" not in df.columns:
        df["motorcycle_count"] = 0

    df["timestamp_bin"]       = pd.to_numeric(df["timestamp_bin"],       errors="coerce").astype("Int64")
    df["car_count"]           = pd.to_numeric(df["car_count"],           errors="coerce").fillna(0).astype(int)
    df["motorcycle_count"]    = pd.to_numeric(df["motorcycle_count"],    errors="coerce").fillna(0).astype(int)
    df["heavy_vehicle_count"] = pd.to_numeric(df["heavy_vehicle_count"], errors="coerce").fillna(0).astype(int)
    df["avg_car_speed_kph"]   = pd.to_numeric(df["avg_car_speed_kph"],   errors="coerce")
    df["incident_status"]     = df["incident_status"].astype(str).str.strip().str.lower()

    bad_bins = df[df["timestamp_bin"].isna()]
    if not bad_bins.empty:
        raise ValueError(
            f"Manual CSV {direction}/{slot} has rows with non-numeric timestamp_bin: "
            f"rows {bad_bins.index.tolist()}"
        )

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

    # Effective Road Occupancy in Passenger Car Equivalents
    df["pce_total"] = (
        df["car_count"]           * CAR_PCE
        + df["motorcycle_count"]  * MOTORCYCLE_PCE
        + df["heavy_vehicle_count"] * HEAVY_PCE
    ).round(2)

    df["flow_rate_veh_hr"] = df["vehicle_count"] * (3600 // BIN_SECONDS)
    df["flow_rate_pce_hr"] = df["pce_total"]     * (3600 // BIN_SECONDS)

    df["density_veh_km"] = np.where(
        df["avg_car_speed_kph"] > 0,
        df["flow_rate_veh_hr"] / df["avg_car_speed_kph"],
        np.nan,
    ).round(4)

    # PCE-weighted density gives a more accurate road-load figure for SUMO.
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
    for name, attrs in SUMO_VTYPES.items():
        ET.SubElement(root, "vType", id=name, **attrs)


def _add_route(root: ET.Element, edge_from: str, edge_to: str) -> str:
    route_id = f"route_{edge_from}_{edge_to}"
    ET.SubElement(root, "route", id=route_id, edges=f"{edge_from} {edge_to}")
    return route_id


def write_rou_xml(
    df_bins: pd.DataFrame,
    out_path: Path,
    edge_from: str,
    edge_to: str,
) -> Path:
    """
    Declares vehicle types + the single end-to-end route. Individual vehicles
    are instantiated via the companion .flow.xml — keeping the .rou.xml tiny
    and reusable across slots.
    """
    root = ET.Element("routes")
    _add_vtypes(root)
    _add_route(root, edge_from, edge_to)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_prettify(root))
    return out_path


def write_flow_xml(
    df_bins: pd.DataFrame,
    out_path: Path,
    edge_from: str,
    edge_to: str,
) -> Path:
    """
    One `<flow>` per (bin × vehicle class). `vehsPerHour` derives from the bin
    count (count * 60 for 1-minute bins). Departure speed is taken from the
    bin's avg_speed_kph, falling back to FALLBACK_SPEED_KPH if blank/zero.
    """
    root = ET.Element("routes")
    _add_vtypes(root)
    route_id = _add_route(root, edge_from, edge_to)

    for _, row in df_bins.iterrows():
        begin = int(row["timestamp_s"])
        end   = begin + BIN_SECONDS
        if int(row["vehicle_count"]) == 0:
            continue

        speed_kph    = float(row["avg_car_speed_kph"]) if pd.notna(row["avg_car_speed_kph"]) and row["avg_car_speed_kph"] > 0 else FALLBACK_SPEED_KPH
        depart_speed = f"{speed_kph / 3.6:.2f}"

        for cls, count_col in (
            ("car",        "car_count"),
            ("motorcycle", "motorcycle_count"),
            ("heavy",      "heavy_vehicle_count"),
        ):
            cls_count = int(row[count_col])
            if cls_count == 0:
                continue
            ET.SubElement(
                root, "flow",
                id          = f"flow_{cls}_{begin}",
                type        = cls,
                route       = route_id,
                begin       = str(begin),
                end         = str(end),
                vehsPerHour = str(cls_count * (3600 // BIN_SECONDS)),
                departSpeed = depart_speed,
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_prettify(root))
    return out_path


# ---------------------------------------------------------------------------
# ORCHESTRATION
# ---------------------------------------------------------------------------

def _parse_direction_and_slot(input_path: Path) -> tuple[str, str]:
    """
    Resolve (direction, slot) from a path like
        data/raw/manual_counts/regional_north/10am.csv
    """
    direction = input_path.parent.name
    slot      = input_path.stem
    if direction not in DIRECTIONS:
        raise ValueError(
            f"Parent folder '{direction}' is not a valid direction. "
            f"Expected one of: {list(DIRECTIONS)}"
        )
    return direction, slot


def process_manual_file(
    input_path: Path,
    *,
    edge_from: str | None = None,
    edge_to:   str | None = None,
) -> dict[str, Path]:
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

    edge_from = edge_from or f"{direction}_in"
    edge_to   = edge_to   or f"{direction}_out"

    rou_out  = DEMAND_DIR / f"{direction}_{slot}.rou.xml"
    flow_out = DEMAND_DIR / f"{direction}_{slot}.flow.xml"

    write_rou_xml (df_bins, rou_out,  edge_from, edge_to)
    write_flow_xml(df_bins, flow_out, edge_from, edge_to)

    mean_speed = float(df_bins["avg_car_speed_kph"].replace(0, np.nan).mean(skipna=True) or 0)
    print(f"\n[manual] {direction} / {slot}")
    print("─" * 56)
    print(f"  Bins                       : {len(df_bins)}")
    print(f"  Total vehicles             : {int(df_bins['vehicle_count'].sum())}")
    print(f"  Cars / motorcycles / heavy : {int(df_bins['car_count'].sum())} / {int(df_bins['motorcycle_count'].sum())} / {int(df_bins['heavy_vehicle_count'].sum())}")
    print(f"  Total PCE                  : {round(float(df_bins['pce_total'].sum()), 1)}")
    print(f"  Mean bin speed (kph)       : {round(mean_speed, 2)}")
    print(f"  Bins flagged incident      : {int(df_bins['incident_flag'].sum())}")
    print(f"  Aggregated CSV  → {aggregated_out}")
    print(f"  SUMO .rou.xml   → {rou_out}")
    print(f"  SUMO .flow.xml  → {flow_out}")

    return {"aggregated": aggregated_out, "rou": rou_out, "flow": flow_out}


def discover_manual_csvs() -> list[Path]:
    """Every CSV under every known direction folder."""
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

    parser.add_argument("--edge-from", default=None,
                        help="SUMO source edge ID (default: <direction>_in)")
    parser.add_argument("--edge-to", default=None,
                        help="SUMO destination edge ID (default: <direction>_out)")
    args = parser.parse_args()

    if args.all:
        process_all()
        return

    if not args.input.exists():
        sys.exit(f"CRITICAL: Input file not found: {args.input}")
    process_manual_file(args.input, edge_from=args.edge_from, edge_to=args.edge_to)


if __name__ == "__main__":
    main()
