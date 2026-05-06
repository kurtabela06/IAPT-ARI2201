"""
simulation/networks.py
======================
Hardcoded NetworkX road skeletons for the Malta Road Recovery study.

We model four directed road networks as networkx.DiGraph objects:

    RNB   Regional Road Northbound
    RSB   Regional Road Southbound
    VEB   Valley Road Eastbound
    VWB   Valley Road Westbound

Every edge carries three structural attributes:

    length_m        — physical length of the road segment in metres
    lanes           — number of through lanes
    max_occupancy_m — length_m × lanes  (the spatial-density denominator
                      used by the Greenshields engine)

Plus a free-flow speed (`v_free_mps`) inherited from the segment's posted
limit, and an `is_escape` flag identifying diverge edges that traffic spills
to under congestion (Q3 dynamic diversion).

The four DiGraphs are returned by `build_all_networks()`.
Geometry numbers come from the user-supplied spec — they are intentionally
embedded here as Python literals (Q2 = b) so the road model is read in one
file with no YAML round-trip.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

# ---------------------------------------------------------------------------
# Free-flow speeds (km/h → m/s) per segment.
# ---------------------------------------------------------------------------
# Regional: posted 80 km/h. Valley: posted 60 km/h. Naxxar slip 50 km/h.
KPH_TO_MPS = 1.0 / 3.6
V_FREE_REGIONAL_MPS = 80.0 * KPH_TO_MPS    # ≈ 22.22 m/s
V_FREE_VALLEY_MPS   = 60.0 * KPH_TO_MPS    # ≈ 16.67 m/s
V_FREE_NAXXAR_MPS   = 50.0 * KPH_TO_MPS    # ≈ 13.89 m/s


# ---------------------------------------------------------------------------
# Edge metadata helper
# ---------------------------------------------------------------------------

@dataclass
class EdgeSpec:
    src:          str
    dst:          str
    label:        str       # internal ID (used by the engine and telemetry)
    length_m:     float
    lanes:        int
    v_free:       float
    is_escape:    bool = False
    is_inflow:    bool = False     # demand-injection edge
    is_exit:      bool = False     # measurement edge for calibration / recovery
    role:         str = ""         # human-readable role tag (flyover, trunk, …)
    display_name: str = ""         # academic label shown by the visualizer
                                   # (e.g. "Regional Road", "Entry 1", "Exit 2")


def _add_edges(g: nx.DiGraph, specs: list[EdgeSpec]) -> None:
    for s in specs:
        g.add_edge(
            s.src, s.dst,
            label           = s.label,
            length_m        = s.length_m,
            lanes           = s.lanes,
            max_occupancy_m = s.length_m * s.lanes,
            v_free_mps      = s.v_free,
            is_escape       = s.is_escape,
            is_inflow       = s.is_inflow,
            is_exit         = s.is_exit,
            role            = s.role,
            display_name    = s.display_name or s.label,
        )


# ---------------------------------------------------------------------------
# A. Regional Road Southbound (RSB)
# ---------------------------------------------------------------------------
# Geometry (Start → End):
#
#   [Flyover Entry,   2L, 140 m]──┐
#                                  ├──[Main Mergeway, 3L, 235 m]──┐
#   [Underpass Entry, 1L, 140 m]──┘                                │
#                                                                  ▼
#                                            ┌──[Escape Exit, 1L, 175 m] ── (escape sink)
#                                            │  diverges at start of Main Way
#   ────────────[Main Way, 2L, 175 m]────────┴──► (south sink)
#
# Key node IDs:
#   src_flyover_S, src_underpass_S    — inflow boundaries
#   merge_S                           — flyover + underpass meet
#   diverge_S                         — start of the Main Way / Escape split
#   sink_main_S, sink_escape_S        — exit boundaries

def build_rsb() -> nx.DiGraph:
    g = nx.DiGraph(name="RSB", display_name="Regional Road Southbound")
    specs = [
        EdgeSpec("src_flyover_S",   "merge_S",
                 "flyover_S",   140.0, 2, V_FREE_REGIONAL_MPS,
                 is_inflow=True, role="flyover",
                 display_name="Entry 1 (Flyover)"),
        EdgeSpec("src_underpass_S", "merge_S",
                 "underpass_S", 140.0, 1, V_FREE_REGIONAL_MPS,
                 is_inflow=True, role="underpass",
                 display_name="Entry 2 (Underpass)"),
        EdgeSpec("merge_S",         "diverge_S",
                 "mergeway_S",  235.0, 3, V_FREE_REGIONAL_MPS,
                 role="trunk",
                 display_name="Regional Road"),
        EdgeSpec("diverge_S",       "sink_main_S",
                 "mainway_S",   175.0, 2, V_FREE_REGIONAL_MPS,
                 is_exit=True, role="main_way",
                 display_name="Exit 1 (Tunnel)"),
        EdgeSpec("diverge_S",       "sink_escape_S",
                 "escape_S",    175.0, 1, V_FREE_REGIONAL_MPS,
                 is_escape=True, role="escape",
                 display_name="Exit 2 (Escape)"),
    ]
    _add_edges(g, specs)
    g.graph["inflow_edges"]   = ["flyover_S", "underpass_S"]
    g.graph["exit_edges"]     = ["mainway_S"]
    g.graph["escape_edges"]   = ["escape_S"]
    # Closures act on the post-diverge Main Way only — never the trunk and
    # never the escape. That way the escape route stays available and
    # vehicles can peel off to it as the Main Way saturates.
    g.graph["closeable_edge"] = "mainway_S"
    return g


# ---------------------------------------------------------------------------
# B. Regional Road Northbound (RNB)
# ---------------------------------------------------------------------------
# Geometry (Start → End):
#
#   ─[Main Merge Way, 3L, 140 m]──merge_join_N──[Main Merge Way (cont), 3L, 120 m]──diverge_N──┬──[Final Main Way, 2L, 465 m]──► (north sink)
#                                  ▲                                                          │
#                                  │                                                          └──[Escape Way, 1L, 465 m]──► (escape sink)
#                                  │
#                          [Underpass Entry 1, 1L, 250 m]
#                                  ▲
#                                  │
#                              src_underpass_N
#
# Node IDs:
#   src_main_N           — inflow start (before the first 140 m segment)
#   merge_join_N         — where Underpass Entry 1 joins the trunk
#   diverge_N            — split point
#   sink_final_N         — Final Main Way exit
#   sink_escape_N        — Escape Way exit
#   src_underpass_N      — Underpass Entry 1 inflow

def build_rnb() -> nx.DiGraph:
    g = nx.DiGraph(name="RNB", display_name="Regional Road Northbound")
    specs = [
        # The 3-lane Main Merge is itself the wider inflow (Entry 1).
        EdgeSpec("src_main_N",       "merge_join_N",
                 "mergeway_N_a", 140.0, 3, V_FREE_REGIONAL_MPS,
                 is_inflow=True, role="trunk_pre",
                 display_name="Entry 1 (Main Merge)"),
        EdgeSpec("src_underpass_N",  "merge_join_N",
                 "underpass_N",  250.0, 1, V_FREE_REGIONAL_MPS,
                 is_inflow=True, role="underpass",
                 display_name="Entry 2 (Underpass)"),
        EdgeSpec("merge_join_N",     "diverge_N",
                 "mergeway_N_b", 120.0, 3, V_FREE_REGIONAL_MPS,
                 role="trunk_post",
                 display_name="Regional Road"),
        EdgeSpec("diverge_N",        "sink_final_N",
                 "mainway_N",    465.0, 2, V_FREE_REGIONAL_MPS,
                 is_exit=True, role="main_way",
                 display_name="Exit 1 (Final Main)"),
        EdgeSpec("diverge_N",        "sink_escape_N",
                 "escape_N",     465.0, 1, V_FREE_REGIONAL_MPS,
                 is_escape=True, role="escape",
                 display_name="Exit 2 (Escape)"),
    ]
    _add_edges(g, specs)
    g.graph["inflow_edges"]   = ["mergeway_N_a", "underpass_N"]
    g.graph["exit_edges"]     = ["mainway_N"]
    g.graph["escape_edges"]   = ["escape_N"]
    # Closure acts on the post-diverge Main Way only — the trunk stays open
    # so the diverge node remains reachable and traffic can peel off to escape.
    g.graph["closeable_edge"] = "mainway_N"
    return g


# ---------------------------------------------------------------------------
# C. Valley Road Eastbound (VEB)
# ---------------------------------------------------------------------------
# Geometry:
#
#   src_valley_E ──[Valley Road, 1L, 180 m]── branch_E ──[Valley Cont., 1L, 350 m]── sink_main_E
#                                              │
#                                              └─[Naxxar Road, 1L, 400 m]── sink_naxxar_E   (escape)
#
# The Naxxar Road branch is the dynamic-diversion target ("Pressure-Valve"):
# when the Valley Continuation queues up, a fraction of vehicles spill onto
# Naxxar Road. Same mechanism as the regional escape edges.

def build_veb() -> nx.DiGraph:
    g = nx.DiGraph(name="VEB", display_name="Valley Road Eastbound")
    specs = [
        EdgeSpec("src_valley_E",  "branch_E",
                 "valley_E",        180.0, 1, V_FREE_VALLEY_MPS,
                 is_inflow=True, role="valley_pre",
                 display_name="Valley Road"),
        EdgeSpec("branch_E",      "sink_main_E",
                 "valley_cont_E",   350.0, 1, V_FREE_VALLEY_MPS,
                 is_exit=True, role="valley_continuation",
                 display_name="Exit 1 (Valley Continuation)"),
        EdgeSpec("branch_E",      "sink_naxxar_E",
                 "naxxar_E",        400.0, 1, V_FREE_NAXXAR_MPS,
                 is_escape=True, role="naxxar_diversion",
                 display_name="Exit 2 (Naxxar Road)"),
    ]
    _add_edges(g, specs)
    g.graph["inflow_edges"]   = ["valley_E"]
    g.graph["exit_edges"]     = ["valley_cont_E"]
    g.graph["escape_edges"]   = ["naxxar_E"]
    g.graph["closeable_edge"] = "valley_cont_E"
    return g


# ---------------------------------------------------------------------------
# D. Valley Road Westbound (VWB)
# ---------------------------------------------------------------------------
# Geometry: Valley Road = 1000 m total in the westbound direction.
#
# Inflows:
#   Psaila Street (200 m)                 ──┐
#                                            ├── Fleur-de-lys Entry (135 m) ── joins Valley Road
#   Fleur-de-lys (200 m, single branch)   ──┘                                  130 m before the network end
#
#   src_psaila_W ─[psaila_W, 1L, 200 m]─┐
#                                        │
#                         (Psaila joins Fleur-de-lys *between* the 200 m and 135 m segments)
#                                        ▼
#   src_fleur_a ─[fleur_a_W, 1L, 200 m]─ junction_fleur_W ─[fleur_b_W, 1L, 135 m]─ junction_valley_W
#
#   src_valley_W ─[valley_W_pre, 1L, 370 m]─ junction_valley_W ─[valley_W_post, 1L, 130 m]─ sink_W
#
# Total Valley Road in this direction: 370 + 130 = 500 m (revised spec).

def build_vwb() -> nx.DiGraph:
    g = nx.DiGraph(name="VWB", display_name="Valley Road Westbound")
    # Main stretch shrunk from 1000 m → 500 m. Fleur-de-lys still joins 130 m
    # before the network end, so valley_W_pre = 500 - 130 = 370 m.
    specs = [
        EdgeSpec("src_valley_W",       "junction_valley_W",
                 "valley_W_pre",   370.0, 1, V_FREE_VALLEY_MPS,
                 is_inflow=True, role="valley_main_pre",
                 display_name="Entry 1 (Valley Road)"),
        EdgeSpec("junction_valley_W",  "sink_W",
                 "valley_W_post",  130.0, 1, V_FREE_VALLEY_MPS,
                 is_exit=True, role="valley_main_post",
                 display_name="Exit 1 (Valley Road)"),

        # Fleur-de-lys branch: 200 m + 135 m, with Psaila joining at the seam.
        EdgeSpec("src_fleur_a",        "junction_fleur_W",
                 "fleur_a_W",      200.0, 1, V_FREE_VALLEY_MPS,
                 is_inflow=True, role="fleur_de_lys_a",
                 display_name="Entry 2 (Fleur-de-lys)"),
        EdgeSpec("src_psaila_W",       "junction_fleur_W",
                 "psaila_W",       200.0, 1, V_FREE_VALLEY_MPS,
                 is_inflow=True, role="psaila_street",
                 display_name="Entry 3 (Psaila Street)"),
        EdgeSpec("junction_fleur_W",   "junction_valley_W",
                 "fleur_b_W",      135.0, 1, V_FREE_VALLEY_MPS,
                 role="fleur_de_lys_b",
                 display_name="Valley Road"),
    ]
    _add_edges(g, specs)
    g.graph["inflow_edges"]   = ["valley_W_pre", "fleur_a_W", "psaila_W"]
    g.graph["exit_edges"]     = ["valley_W_post"]
    g.graph["escape_edges"]   = []   # no escape route in this direction
    g.graph["closeable_edge"] = "valley_W_post"
    return g


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

NETWORK_BUILDERS = {
    "RNB": build_rnb,
    "RSB": build_rsb,
    "VEB": build_veb,
    "VWB": build_vwb,
}

# Map manual-CSV direction names → DiGraph keys
DIRECTION_TO_NETWORK = {
    "regional_north": "RNB",
    "regional_south": "RSB",
    "valley_east":    "VEB",
    "valley_west":    "VWB",
}

# ---------------------------------------------------------------------------
# Hardcoded node positions per network — used by the visualizer instead of
# nx.spring_layout. Coordinates are abstract (no real-world meaning) but
# arranged so the topology reads naturally on screen:
#   inflows on the left, outflows on the right; merge / split nodes line up
#   vertically; long edges get more horizontal room.
# ---------------------------------------------------------------------------

NODE_LAYOUTS: dict[str, dict[str, tuple[float, float]]] = {
    "RNB": {
        # Two inflows (top + bottom) merge → trunk → diverge → main+escape
        "src_main_N":      (-0.7,  +0.5),
        "src_underpass_N": (-0.7,  -0.5),
        "merge_join_N":    ( 0.0,   0.0),
        "diverge_N":       ( 0.5,   0.0),
        "sink_final_N":    ( 1.5,  +0.4),
        "sink_escape_N":   ( 1.5,  -0.4),
    },
    "RSB": {
        "src_flyover_S":   (-0.7,  +0.5),
        "src_underpass_S": (-0.7,  -0.5),
        "merge_S":         ( 0.0,   0.0),
        "diverge_S":       ( 0.6,   0.0),
        "sink_main_S":     ( 1.5,  +0.4),
        "sink_escape_S":   ( 1.5,  -0.4),
    },
    "VEB": {
        # Single inflow → branch → main vs Naxxar diversion
        "src_valley_E":    (-0.8,   0.0),
        "branch_E":        ( 0.0,   0.0),
        "sink_main_E":     ( 1.0,  +0.4),
        "sink_naxxar_E":   ( 1.0,  -0.4),
    },
    "VWB": {
        # 3-way merge: Psaila + Fleur-de-lys, plus the long Valley pre-edge
        "src_valley_W":      (-1.2,   0.0),
        "src_fleur_a":       (-0.7,  -0.7),
        "src_psaila_W":      (-0.7,  -0.4),
        "junction_fleur_W":  (-0.2,  -0.55),
        "junction_valley_W": ( 0.4,   0.0),
        "sink_W":            ( 1.0,   0.0),
    },
}


def build_network(key: str) -> nx.DiGraph:
    if key not in NETWORK_BUILDERS:
        raise KeyError(f"Unknown network '{key}'. Available: {list(NETWORK_BUILDERS)}")
    return NETWORK_BUILDERS[key]()


def build_all_networks() -> dict[str, nx.DiGraph]:
    return {k: f() for k, f in NETWORK_BUILDERS.items()}


def get_layout(key: str) -> dict[str, tuple[float, float]]:
    """Return the hardcoded (x, y) layout for a network key."""
    return dict(NODE_LAYOUTS[key])
