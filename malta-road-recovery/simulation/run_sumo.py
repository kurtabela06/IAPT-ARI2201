"""
simulation/run_sumo.py
======================
Phase 2 — single beginner-friendly entry point for the SUMO pipeline.

This one script does three things you can pick from a command-line flag:

    --build      Generate network files (.nod.xml + .edg.xml + .con.xml) from
                 roads.yaml and run netconvert to produce <segment>.net.xml.
                 The OSM extract is still downloaded for context but the
                 topology comes from YAML so merge/diverge geometry is exact.
    --preview    Open SUMO-GUI on the empty network with each labelled edge
                 highlighted in a different colour (Flyover/Underpass/Trunk
                 /Tunnel/Escape) so you can verify the geometry visually.
    --simulate   Build (if needed) + generate demand from the manual CSV
                 + launch SUMO-GUI with moving, colour-coded vehicles.

----------------------------------------------------------------------------
WHAT THE SUMO FILES MEAN
----------------------------------------------------------------------------

  .osm        raw OpenStreetMap extract (informational only for merge_split)
  .nod.xml    SUMO node file: every junction with x,y coordinates and type
              (priority, traffic_light, dead_end). One per segment, generated
              from roads.yaml by build_segment_files().
  .edg.xml    SUMO edge file: every road segment, with from/to node, lanes,
              speed, allowed classes, and a `name` attribute used as a
              human-readable label (Flyover, Trunk, Tunnel, …).
  .con.xml    SUMO connection file: explicit lane-to-lane links across each
              junction. The merge node has `keepClear=1` and `pass=true` on
              the trunk-side connections so the trunk has right-of-way.
  .net.xml    The compiled network. netconvert reads .nod + .edg + .con and
              produces this. Comments are post-injected so you can see which
              edge is which directly in the file.
  .rou.xml    Vehicle types (with colour) + routes through the network.
  .flow.xml   Per-bin flows from the manual CSV.
  .sumocfg    Master config tying net + routes + flows + run params.

Vehicle COLOURS (set in manual_to_sumo.py):
    car        → yellow   motorcycle → blue   heavy → red

Edge COLOURS (preview only, set by this script's view_settings.xml):
    flyover    → orange
    underpass  → cyan
    trunk      → green
    tunnel     → magenta
    escape     → red
    inflow_n   → cyan      (regional_north 1-lane side inflow)

----------------------------------------------------------------------------
COMMANDS
----------------------------------------------------------------------------

  Build:
      python simulation/run_sumo.py --build
      python simulation/run_sumo.py --build --segment regional_road
      python simulation/run_sumo.py --build --dry-run

  Preview (no traffic — verify geometry only, with coloured edges):
      python simulation/run_sumo.py --preview --segment regional_road
      python simulation/run_sumo.py --preview --segment regional_road --netedit

  Simulate:
      python simulation/run_sumo.py --simulate --direction regional_north --slot 10am
      python simulation/run_sumo.py --simulate --direction valley_east --slot 1pm --no-gui

----------------------------------------------------------------------------
PREREQUISITES
----------------------------------------------------------------------------
  • SUMO installed and on $PATH (provides netconvert, sumo, sumo-gui, netedit)
  • pip install requests pyyaml pandas
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

import yaml

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_ROADS  = PROJECT_ROOT / "config" / "roads.yaml"
CONFIG_SIM    = PROJECT_ROOT / "config" / "simulation.yaml"
OSM_DIR       = PROJECT_ROOT / "data" / "raw" / "osm"
NETWORKS_DIR  = PROJECT_ROOT / "simulation" / "networks"
DEMAND_DIR    = PROJECT_ROOT / "simulation" / "demand"
SUMOCFG_DIR   = PROJECT_ROOT / "simulation" / "sumo_configs"

OSM_DIR.mkdir   (parents=True, exist_ok=True)
NETWORKS_DIR.mkdir(parents=True, exist_ok=True)
SUMOCFG_DIR.mkdir(parents=True, exist_ok=True)

PLACEHOLDER_PREFIX = 999000000

DIRECTION_TO_SEGMENT = {
    "regional_north": "regional_road",
    "regional_south": "regional_road",
    "valley_east":    "valley_road",
    "valley_west":    "valley_road",
}

# Edge-label → preview colour (RGB 0-1 floats). Read by SUMO-GUI from the
# view-settings.xml we generate before launching the preview.
PREVIEW_COLOURS: dict[str, str] = {
    "flyover_n":   "1,0.5,0",     # orange
    "flyover_s":   "1,0.5,0",
    "underpass_n": "0,1,1",       # cyan
    "underpass_s": "0,1,1",
    "trunk_n":     "0,1,0",       # green
    "trunk_s":     "0,1,0",
    "tunnel_n":    "1,0,1",       # magenta
    "tunnel_s":    "1,0,1",
    "escape_s":    "1,0,0",       # red
    "inflow_n":    "0,1,1",       # cyan (1-lane side inflow)
}


# ---------------------------------------------------------------------------
# CONFIG LOADING
# ---------------------------------------------------------------------------

def _load_configs() -> tuple[dict, dict]:
    with open(CONFIG_ROADS) as f: roads = yaml.safe_load(f)["segments"]
    sim_cfg: dict = {}
    if CONFIG_SIM.exists():
        with open(CONFIG_SIM) as f: sim_cfg = yaml.safe_load(f) or {}
    return roads, sim_cfg


def _all_way_ids(seg_cfg: dict) -> list[int]:
    """Collect every Way ID referenced anywhere in a segment config."""
    ids: list[int] = []
    for dir_cfg in seg_cfg.get("directions", {}).values():
        topo = dir_cfg.get("topology", {})
        ttype = topo.get("type")
        if ttype == "merge_split":
            for slot in ("inflow_a", "inflow_b", "trunk", "outflow_main", "outflow_div"):
                wid = topo.get(slot, {}).get("way_id")
                if wid is not None:
                    ids.append(wid)
        elif ttype == "simple":
            wid = topo.get("mainline", {}).get("way_id")
            if wid is not None:
                ids.append(wid)
    for merge in seg_cfg.get("merge_ways", []):
        ids.append(merge["way_id"])
    return ids


def _check_placeholders(segment_name: str, seg_cfg: dict) -> int:
    placeholders = [w for w in _all_way_ids(seg_cfg) if w >= PLACEHOLDER_PREFIX]
    if placeholders:
        print(f"\n  ⚠  WARNING [{segment_name}]: "
              f"{len(placeholders)} placeholder Way ID(s) still in roads.yaml.")
        print(f"     Replace them with real OSM IDs once the field measurements are in.\n")
    return len(placeholders)


# ---------------------------------------------------------------------------
# BUILD: GENERATE NODE/EDGE/CONNECTION FILES FROM YAML TOPOLOGY
# ---------------------------------------------------------------------------
#
# We deliberately bypass OSM-driven topology for merge_split segments because
# the merge geometry, lane counts, and connection priorities need to match
# YAML exactly. netconvert reads our generated .nod/.edg/.con files via
# --node-files / --edge-files / --connection-files and writes the .net.xml.
#
# Node coordinates are arbitrary but laid out so the GUI looks natural:
#
#                 in_a   ──┐
#                          │           merge          split
#       (200m West)   2-l. │     ──────────────►   ──────────────►
#                in_b   ──┘                       │
#                          1-l.        ▲          │
#                                      │          ▼
#                                3-lane trunk (200m)
#                                                    │
#                                                    ▼
#                                          out_main / out_div
#
# Distances are 200 m on each leg by default — easy to read in the GUI but
# short enough that flows establish quickly.

LEG_LEN_M = 200.0     # length of each leg in metres


def _xy_for_merge_split(direction: str) -> dict[str, tuple[float, float]]:
    """
    Lay out the 6 junction nodes for a merge_split direction. Coordinates are
    purely cosmetic — the topology comes from the connections file. NB sits
    above SB on the y-axis so the two carriageways are visually separate
    without sharing nodes (each direction uses distinct node IDs anyway).
    """
    base_y = +80.0 if "north" in direction else -80.0   # NB above, SB below
    sign   = 1.0   if "north" in direction else -1.0    # mirror inflow side
    return {
        "inflow_a_node": (-LEG_LEN_M,        base_y + 30.0 * sign),
        "inflow_b_node": (-LEG_LEN_M,        base_y - 30.0 * sign),
        "merge_node":    (         0.0,      base_y),
        "split_node":    (+LEG_LEN_M,        base_y),
        "out_main_node": (+2 * LEG_LEN_M,    base_y + 30.0 * sign),
        "out_div_node":  (+2 * LEG_LEN_M,    base_y - 30.0 * sign),
    }


def _emit_merge_split_files(
    segment_name: str,
    direction:    str,
    dir_cfg:      dict,
    seg_cfg:      dict,
    nod_root:     ET.Element,
    edg_root:     ET.Element,
    con_root:     ET.Element,
) -> list[tuple[str, str, dict]]:
    """
    Append nodes / edges / connections for a single merge_split direction.
    Returns a list of (label, edge_id, slot_metadata) for downstream use
    (preview colouring, demand splitting, net.xml comments).
    """
    topo  = dir_cfg["topology"]
    speed = float(seg_cfg.get("speed_limit_kph", 80)) / 3.6   # m/s
    xy    = _xy_for_merge_split(direction)

    # Node names from YAML
    in_a   = topo["inflow_a_node"]
    in_b   = topo["inflow_b_node"]
    merge  = topo["merge_node"]
    split  = topo["split_node"]
    out_m  = topo["out_main_node"]
    out_d  = topo["out_div_node"]

    nodes = [
        (in_a,  "priority", *xy["inflow_a_node"]),
        (in_b,  "priority", *xy["inflow_b_node"]),
        (merge, "priority", *xy["merge_node"]),    # priority merge (per spec)
        (split, "priority", *xy["split_node"]),    # priority diverge
        (out_m, "priority", *xy["out_main_node"]),
        (out_d, "priority", *xy["out_div_node"]),
    ]
    for nid, ntype, x, y in nodes:
        ET.SubElement(nod_root, "node", id=nid, x=str(x), y=str(y), type=ntype)

    # Edges. Each carries:
    #   - id          unique within the network
    #   - name        human label, used by net.xml comment annotation
    #   - numLanes    explicit lane count from YAML
    #   - speed       posted limit
    edges_meta: list[tuple[str, str, dict]] = []
    edge_specs = [
        ("inflow_a",    in_a,   merge, topo["inflow_a"]),
        ("inflow_b",    in_b,   merge, topo["inflow_b"]),
        ("trunk",       merge,  split, topo["trunk"]),
        ("outflow_main", split, out_m, topo["outflow_main"]),
        ("outflow_div",  split, out_d, topo["outflow_div"]),
    ]
    for slot, frm, to, slot_cfg in edge_specs:
        edge_id = slot_cfg["label"]
        ET.SubElement(
            edg_root, "edge",
            id=edge_id, **{"from": frm, "to": to},
            name=edge_id,
            numLanes=str(slot_cfg["lanes"]),
            speed=f"{speed:.2f}",
        )
        edges_meta.append((slot, edge_id, slot_cfg))

    # Connections — make the merge a priority junction (trunk has right-of-way)
    # and the split a free diverge (lane-by-lane explicit links).
    in_a_lanes = topo["inflow_a"]["lanes"]
    in_b_lanes = topo["inflow_b"]["lanes"]
    trunk_lanes = topo["trunk"]["lanes"]
    out_main_lanes = topo["outflow_main"]["lanes"]
    out_div_lanes  = topo["outflow_div"]["lanes"]

    # Merge: inflow_a (wider) keeps the lower trunk lanes; inflow_b feeds the
    # rightmost (kerbside) trunk lane. With --lefthand, lane 0 is leftmost.
    #
    # We assume in_a contributes lanes 0..in_a_lanes-1 of the trunk and in_b
    # contributes the remaining trunk lane(s). For 2+1→3 this gives a clean
    # natural merge.
    for fl in range(in_a_lanes):
        ET.SubElement(con_root, "connection",
                      **{"from": topo["inflow_a"]["label"]},
                      to=topo["trunk"]["label"], fromLane=str(fl), toLane=str(fl))
    for fl in range(in_b_lanes):
        # inflow_b feeds the rightmost (highest-index) trunk lane(s)
        target_lane = trunk_lanes - in_b_lanes + fl
        ET.SubElement(con_root, "connection",
                      **{"from": topo["inflow_b"]["label"]},
                      to=topo["trunk"]["label"], fromLane=str(fl), toLane=str(target_lane))

    # Split: trunk lanes 0..out_main_lanes-1 → outflow_main, the rest → outflow_div.
    for tl in range(out_main_lanes):
        ET.SubElement(con_root, "connection",
                      **{"from": topo["trunk"]["label"]},
                      to=topo["outflow_main"]["label"], fromLane=str(tl), toLane=str(tl))
    for offset in range(out_div_lanes):
        from_lane = trunk_lanes - out_div_lanes + offset
        ET.SubElement(con_root, "connection",
                      **{"from": topo["trunk"]["label"]},
                      to=topo["outflow_div"]["label"], fromLane=str(from_lane), toLane=str(offset))

    return edges_meta


def _emit_simple_files(
    segment_name: str,
    direction:    str,
    dir_cfg:      dict,
    seg_cfg:      dict,
    nod_root:     ET.Element,
    edg_root:     ET.Element,
    con_root:     ET.Element,
    y_offset:     float,
) -> list[tuple[str, str, dict]]:
    """Emit a single straight edge for a simple (single-way) direction."""
    topo  = dir_cfg["topology"]
    speed = float(seg_cfg.get("speed_limit_kph", 60)) / 3.6
    main  = topo["mainline"]
    frm   = topo["from_node"]
    to    = topo["to_node"]

    # Both directions share the same boundary nodes; emit each only once.
    existing = {n.get("id") for n in nod_root.findall("node")}
    if frm not in existing:
        ET.SubElement(nod_root, "node", id=frm, x="0",   y=str(y_offset), type="priority")
    if to not in existing:
        ET.SubElement(nod_root, "node", id=to,  x=str(LEG_LEN_M), y=str(y_offset), type="priority")

    edge_id = main["label"]
    ET.SubElement(
        edg_root, "edge",
        id=edge_id, **{"from": frm, "to": to},
        name=edge_id,
        numLanes=str(main["lanes"]),
        speed=f"{speed:.2f}",
    )
    return [("mainline", edge_id, main)]


def _build_segment_files(
    segment_name: str,
    seg_cfg:      dict,
    net_dir:      Path,
) -> tuple[Path, Path, Path, list[tuple[str, str, str, dict]]]:
    """
    Generate the .nod.xml, .edg.xml, .con.xml for the segment and return
    their paths plus a list of (direction, slot, edge_id, slot_cfg) tuples
    used by the demand and preview steps.
    """
    nod_root = ET.Element("nodes")
    edg_root = ET.Element("edges")
    con_root = ET.Element("connections")

    all_edges: list[tuple[str, str, str, dict]] = []
    y_off = 0.0
    for direction, dir_cfg in seg_cfg.get("directions", {}).items():
        ttype = dir_cfg.get("topology", {}).get("type")
        if ttype == "merge_split":
            entries = _emit_merge_split_files(
                segment_name, direction, dir_cfg, seg_cfg,
                nod_root, edg_root, con_root,
            )
        elif ttype == "simple":
            entries = _emit_simple_files(
                segment_name, direction, dir_cfg, seg_cfg,
                nod_root, edg_root, con_root,
                y_offset=y_off,
            )
            y_off += 100.0   # keep east/west visually separated for valley_road
        else:
            sys.exit(f"Unknown topology.type='{ttype}' for {segment_name}/{direction}")

        for slot, edge_id, slot_cfg in entries:
            all_edges.append((direction, slot, edge_id, slot_cfg))

    net_dir.mkdir(parents=True, exist_ok=True)
    nod_path = net_dir / f"{segment_name}.nod.xml"
    edg_path = net_dir / f"{segment_name}.edg.xml"
    con_path = net_dir / f"{segment_name}.con.xml"

    for path, root in (
        (nod_path, nod_root),
        (edg_path, edg_root),
        (con_path, con_root),
    ):
        rough = ET.tostring(root, "utf-8")
        path.write_text(minidom.parseString(rough).toprettyxml(indent="  "))

    print(f"  [build] wrote {nod_path.name}, {edg_path.name}, {con_path.name}")
    return nod_path, edg_path, con_path, all_edges


# ---------------------------------------------------------------------------
# OSM DOWNLOAD (informational — not used for merge_split topology)
# ---------------------------------------------------------------------------

def _download_osm(segment_name: str, seg_cfg: dict, out_path: Path, *, force: bool) -> None:
    if out_path.exists() and not force:
        print(f"  [osm] {segment_name}: using existing {out_path.name}")
        return
    try:
        import requests
    except ImportError:
        print("  [osm] 'requests' not installed — skipping OSM download.")
        return

    real_ids = [w for w in _all_way_ids(seg_cfg) if w < PLACEHOLDER_PREFIX]
    bbox = seg_cfg["bbox"]
    s, w, n, e = bbox["south"], bbox["west"], bbox["north"], bbox["east"]
    if real_ids:
        way_filter = f"way(id:{','.join(str(i) for i in real_ids)});"
    else:
        way_filter = f'way["highway"]({s},{w},{n},{e});'
    query = f"""
[out:xml][timeout:90];
(
  {way_filter}
  way["highway"]({s},{w},{n},{e});
  >;
);
out body;
"""
    print(f"  [osm] {segment_name}: downloading bbox=({s},{w},{n},{e})  "
          f"+ {len(real_ids)} explicit ways…")
    resp = requests.post("https://overpass-api.de/api/interpreter",
                         data={"data": query}, timeout=120)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)
    print(f"  [osm] {segment_name}: saved {out_path}  "
          f"({out_path.stat().st_size // 1024} KB)")


# ---------------------------------------------------------------------------
# NETCONVERT
# ---------------------------------------------------------------------------

def _run_netconvert(
    segment_name: str,
    nod_path:     Path,
    edg_path:     Path,
    con_path:     Path,
    net_dir:      Path,
    nc_cfg:       dict,
    *,
    dry_run:      bool,
) -> Path:
    if shutil.which("netconvert") is None and not dry_run:
        sys.exit("CRITICAL: 'netconvert' not found on $PATH. Install SUMO first.")

    out_path = net_dir / f"{segment_name}.net.xml"
    cmd = [
        "netconvert",
        "--node-files",       str(nod_path),
        "--edge-files",       str(edg_path),
        "--connection-files", str(con_path),
        "--output-file",      str(out_path),
        "--lefthand",                  # Malta drives left
        "--no-turnarounds",
        "--tls.guess",        "false", # we declared all junctions priority
        "--sidewalks.guess",  "false",
        "--crossings.guess",  "false",
        "--precision",        str(nc_cfg.get("geometry_precision", 4)),
    ]

    print(f"\n  [netconvert] {segment_name}")
    print("  $ " + " ".join(cmd))
    if dry_run:
        print("  (dry-run — not executing)")
        return out_path

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        sys.exit(f"CRITICAL: netconvert failed for {segment_name} (exit {result.returncode})")
    print(f"  → {out_path}")
    return out_path


def _annotate_net_xml(
    net_path:  Path,
    edges_meta: list[tuple[str, str, str, dict]],
) -> None:
    """
    Inject a comment above each <edge> element in the .net.xml giving its
    label, source Way ID and lane count. Helps the user identify which
    netconvert-generated edge corresponds to which YAML slot.
    """
    if not net_path.exists():
        return

    text = net_path.read_text()
    edge_lookup = {edge_id: (direction, slot, slot_cfg)
                   for direction, slot, edge_id, slot_cfg in edges_meta}

    out_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("<edge "):
            # naive but sufficient: pull id="..."
            for piece in stripped.split():
                if piece.startswith('id="'):
                    edge_id = piece[4:].split('"', 1)[0]
                    if edge_id in edge_lookup:
                        direction, slot, cfg = edge_lookup[edge_id]
                        indent = line[:len(line) - len(stripped)]
                        comment = (
                            f"{indent}<!-- LABEL: {edge_id}  "
                            f"({direction} / {slot})  "
                            f"way={cfg.get('way_id')}  lanes={cfg.get('lanes')} -->"
                        )
                        out_lines.append(comment)
                    break
        out_lines.append(line)

    net_path.write_text("\n".join(out_lines))
    print(f"  [annotate] injected labels into {net_path.name}")


def cmd_build(
    segments_filter: list[str] | None,
    *,
    force_download: bool,
    dry_run:        bool,
) -> None:
    roads_cfg, sim_cfg = _load_configs()
    nc_cfg = sim_cfg.get("netconvert", {})

    targets = list(segments_filter) if segments_filter else list(roads_cfg.keys())

    for segment_name in targets:
        if segment_name not in roads_cfg:
            sys.exit(f"Unknown segment '{segment_name}'. Available: {list(roads_cfg)}")
        seg_cfg = roads_cfg[segment_name]

        print(f"\n[build] Processing {segment_name}")
        n_real = sum(1 for w in _all_way_ids(seg_cfg) if w < PLACEHOLDER_PREFIX)
        print(f"  Generated network from {n_real} real Way ID(s) in roads.yaml")
        _check_placeholders(segment_name, seg_cfg)

        net_dir  = NETWORKS_DIR / segment_name
        osm_path = OSM_DIR / f"{segment_name}.osm"

        if not dry_run:
            _download_osm(segment_name, seg_cfg, osm_path, force=force_download)
        else:
            print(f"  [osm] {segment_name}: (dry-run — skipping download)")

        nod, edg, con, edges_meta = _build_segment_files(segment_name, seg_cfg, net_dir)
        net_path = _run_netconvert(segment_name, nod, edg, con, net_dir, nc_cfg, dry_run=dry_run)

        if not dry_run:
            _annotate_net_xml(net_path, edges_meta)

    print("\n[build] Done.\n")


# ---------------------------------------------------------------------------
# PREVIEW (with coloured edges)
# ---------------------------------------------------------------------------

def _write_view_settings(
    out_path:     Path,
    edges_meta:   list[tuple[str, str, str, dict]],
) -> None:
    """
    Write a SUMO-GUI view-settings file that paints each labelled edge a
    different colour. SUMO reads this via --gui-settings-file.
    """
    root = ET.Element("viewsettings")
    scheme = ET.SubElement(root, "scheme", name="phase2-preview")
    edges_el = ET.SubElement(scheme, "edges", **{
        "laneEdgeMode": "0",                 # colour by user data (param)
        "laneEdgeScaleMode": "0",
        "edgeName_show": "1", "edgeName_size": "60", "edgeName_color": "0,0,0",
    })
    # Per-edge colours via <colorScheme>. Names must match edge IDs.
    for direction, slot, edge_id, _slot_cfg in edges_meta:
        colour = PREVIEW_COLOURS.get(edge_id, "0.6,0.6,0.6")  # grey default
        ET.SubElement(scheme, "edgeColor",
                      id=edge_id, value=colour, name=edge_id)

    rough = ET.tostring(root, "utf-8")
    out_path.write_text(minidom.parseString(rough).toprettyxml(indent="  "))


def _build_edges_meta_for_segment(seg_cfg: dict) -> list[tuple[str, str, str, dict]]:
    """Re-derive the edge-metadata list from YAML without rebuilding the network."""
    out: list[tuple[str, str, str, dict]] = []
    for direction, dir_cfg in seg_cfg.get("directions", {}).items():
        topo = dir_cfg.get("topology", {})
        if topo.get("type") == "merge_split":
            for slot in ("inflow_a", "inflow_b", "trunk", "outflow_main", "outflow_div"):
                cfg = topo.get(slot)
                if cfg:
                    out.append((direction, slot, cfg["label"], cfg))
        elif topo.get("type") == "simple":
            cfg = topo.get("mainline")
            if cfg:
                out.append((direction, "mainline", cfg["label"], cfg))
    return out


def cmd_preview(segment_name: str, *, use_netedit: bool, dry_run: bool) -> None:
    roads_cfg, _ = _load_configs()
    if segment_name not in roads_cfg:
        sys.exit(f"Unknown segment '{segment_name}'. Available: {list(roads_cfg)}")

    net_xml = NETWORKS_DIR / segment_name / f"{segment_name}.net.xml"
    if not net_xml.exists():
        sys.exit(
            f"CRITICAL: {net_xml} does not exist.\n"
            f"  → Run first:  python simulation/run_sumo.py --build --segment {segment_name}"
        )

    binary = "netedit" if use_netedit else "sumo-gui"
    if shutil.which(binary) is None and not dry_run:
        sys.exit(f"CRITICAL: '{binary}' not found on $PATH. Install SUMO first.")

    edges_meta = _build_edges_meta_for_segment(roads_cfg[segment_name])
    view_path = NETWORKS_DIR / segment_name / "view_settings.xml"
    _write_view_settings(view_path, edges_meta)

    if binary == "netedit":
        cmd = ["netedit", "-n", str(net_xml)]
    else:
        cmd = ["sumo-gui", "--net-file", str(net_xml),
               "--gui-settings-file", str(view_path)]

    print(f"\n[preview] Launching {binary} on {net_xml.name}  (no vehicles)")
    print(f"  Coloured edges:")
    for direction, slot, edge_id, cfg in edges_meta:
        colour_name = {"1,0.5,0":"orange","0,1,1":"cyan","0,1,0":"green",
                       "1,0,1":"magenta","1,0,0":"red"}.get(
            PREVIEW_COLOURS.get(edge_id, ""), "grey")
        print(f"    {edge_id:<14} {colour_name:<8} ({direction} / {slot}, "
              f"way={cfg.get('way_id')}, lanes={cfg.get('lanes')})")
    print(f"  $ " + " ".join(cmd))

    if dry_run:
        print("  (dry-run — not launching GUI)")
        return
    subprocess.run(cmd)


# ---------------------------------------------------------------------------
# SIMULATE
# ---------------------------------------------------------------------------

def _ensure_demand(direction: str, slot: str) -> tuple[Path, Path]:
    rou  = DEMAND_DIR / f"{direction}_{slot}.rou.xml"
    flow = DEMAND_DIR / f"{direction}_{slot}.flow.xml"
    if rou.exists() and flow.exists():
        return rou, flow

    print(f"\n[demand] {direction}/{slot}: demand files missing — generating now…")
    csv_path = PROJECT_ROOT / "data" / "raw" / "manual_counts" / direction / f"{slot}.csv"
    if not csv_path.exists():
        sys.exit(f"CRITICAL: manual CSV not found:\n  {csv_path}\n"
                 f"Create it (see data/raw/manual_counts/README.md) and re-run.")

    from data_processing.manual_to_sumo import process_manual_file
    process_manual_file(csv_path)
    return rou, flow


def _write_sumocfg(
    net_xml:    Path,
    rou_xml:    Path,
    flow_xml:   Path,
    duration_s: int,
    step_len:   float,
    seed:       int,
    out_path:   Path,
) -> None:
    root = ET.Element("configuration")
    inp = ET.SubElement(root, "input")
    ET.SubElement(inp, "net-file",     value=str(net_xml))
    ET.SubElement(inp, "route-files",  value=f"{rou_xml},{flow_xml}")
    time_el = ET.SubElement(root, "time")
    ET.SubElement(time_el, "begin",       value="0")
    ET.SubElement(time_el, "end",         value=str(duration_s))
    ET.SubElement(time_el, "step-length", value=str(step_len))
    rand = ET.SubElement(root, "random_number")
    ET.SubElement(rand, "seed",          value=str(seed))
    rep = ET.SubElement(root, "report")
    ET.SubElement(rep, "verbose",        value="true")
    ET.SubElement(rep, "no-step-log",    value="true")
    rough = ET.tostring(root, "utf-8")
    out_path.write_text(minidom.parseString(rough).toprettyxml(indent="  "))


def _count_vehicles_in_first_bin(flow_xml: Path) -> int:
    try:
        flows = list(ET.parse(flow_xml).getroot().iter("flow"))
        if not flows:
            return 0
        first_begin = min(int(f.attrib.get("begin", 0)) for f in flows)
        first_bin = [f for f in flows if int(f.attrib.get("begin", 0)) == first_begin]
        return sum(int(float(f.attrib.get("vehsPerHour", 0))) for f in first_bin) // 60
    except Exception:
        return 0


def cmd_simulate(direction: str, slot: str, *, gui: bool, dry_run: bool) -> None:
    roads_cfg, sim_cfg = _load_configs()
    segment = DIRECTION_TO_SEGMENT.get(direction)
    if segment is None:
        sys.exit(f"Unknown direction '{direction}'. Choose one of: "
                 f"{list(DIRECTION_TO_SEGMENT)}")

    net_xml = NETWORKS_DIR / segment / f"{segment}.net.xml"
    if not net_xml.exists():
        sys.exit(f"CRITICAL: {net_xml} does not exist.\n"
                 f"  → Run first:  python simulation/run_sumo.py --build --segment {segment}")

    rou_xml, flow_xml = _ensure_demand(direction, slot)

    sumo_params = sim_cfg.get("sumo", {})
    duration_s  = int  (sumo_params.get("duration_s",   600))
    step_len    = float(sumo_params.get("step_length_s", 1.0))
    seed        = int  (sumo_params.get("seed",          42))

    cfg_path = SUMOCFG_DIR / f"{direction}_{slot}.sumocfg"
    _write_sumocfg(net_xml, rou_xml, flow_xml, duration_s, step_len, seed, cfg_path)

    veh_first_bin = _count_vehicles_in_first_bin(flow_xml)
    binary = "sumo-gui" if gui else "sumo"
    if shutil.which(binary) is None and not dry_run:
        sys.exit(f"CRITICAL: '{binary}' not found on $PATH. Install SUMO first.")

    print(f"\n[simulate] {direction} / {slot}")
    print(f"  Segment   : {segment}")
    print(f"  Network   : {net_xml.name}")
    print(f"  Demand    : {rou_xml.name}  +  {flow_xml.name}")
    print(f"  Sumocfg   : {cfg_path}")
    print(f"  Binary    : {binary}")
    print(f"  Duration  : {duration_s}s   step={step_len}s   seed={seed}")
    print(f"  Starting simulation with ~{veh_first_bin} vehicles from CSV row 0")
    print(f"  Vehicle colors: car=YELLOW   motorcycle=BLUE   heavy=RED")

    cmd = [binary, "-c", str(cfg_path)]
    if gui:
        cmd += ["--start", "--quit-on-end"]
    print(f"  $ " + " ".join(cmd))
    if dry_run:
        print("  (dry-run — not launching SUMO)")
        return
    subprocess.run(cmd)
    print(f"\n[simulate] {direction}/{slot} run complete.\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2 SUMO driver — build, preview, or simulate.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Pick exactly one of:  --build  |  --preview  |  --simulate",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build",    action="store_true",
                      help="Generate network files from roads.yaml and run netconvert")
    mode.add_argument("--preview",  action="store_true",
                      help="Open SUMO-GUI on the empty network with coloured edges")
    mode.add_argument("--simulate", action="store_true",
                      help="Generate demand from manual CSV and run SUMO with traffic")

    parser.add_argument("--segment",   default=None,
                        choices=("regional_road", "valley_road"),
                        help="(--build / --preview) which segment to process")
    parser.add_argument("--direction", default=None,
                        choices=tuple(DIRECTION_TO_SEGMENT.keys()),
                        help="(--simulate) which direction to simulate")
    parser.add_argument("--slot", default=None,
                        help="(--simulate) which time slot label, e.g. 10am")

    parser.add_argument("--force-download", action="store_true",
                        help="(--build) re-fetch the .osm extract")
    parser.add_argument("--netedit", action="store_true",
                        help="(--preview) open in netedit instead of sumo-gui")
    parser.add_argument("--no-gui",  action="store_true",
                        help="(--simulate) headless run — use sumo instead of sumo-gui")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing them")
    args = parser.parse_args()

    if args.build:
        seg_filter = [args.segment] if args.segment else None
        cmd_build(seg_filter, force_download=args.force_download, dry_run=args.dry_run)
        return
    if args.preview:
        if args.segment is None:
            sys.exit("--preview requires --segment <regional_road|valley_road>")
        cmd_preview(args.segment, use_netedit=args.netedit, dry_run=args.dry_run)
        return
    if args.simulate:
        if not args.direction or not args.slot:
            sys.exit("--simulate requires both --direction and --slot")
        cmd_simulate(args.direction, args.slot,
                     gui=not args.no_gui, dry_run=args.dry_run)
        return


if __name__ == "__main__":
    main()
