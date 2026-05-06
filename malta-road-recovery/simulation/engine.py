"""
simulation/engine.py
====================
Custom NetworkX-based traffic engine — cell-transmission flavour.

Each road **edge** is subdivided into a sequence of small **cells**
(default ~100 m each). Cells are the smallest unit of state in the engine:

  state.cells[edge_label][cell_idx] : list[Particle]

Particles carry (class, count, pos_m_in_cell). Their speed is determined by
*their cell's* occupancy, not the whole edge — so a long edge with a queue
near the exit shows a clear speed gradient (free flow at the entry, slow
at the bottleneck).

Per-second update:

  1. Inject demand pulse if a 1-minute boundary just crossed.
  2. Advance pass: every particle moves at its cell's Greenshields speed
     (motorcycles always at v_free). Particles that overflow their cell are
     parked at cell-end and queued for transfer to the next cell (or the
     next edge, if they were in the last cell).
  3. Capacity-capped transfer pass:
       a. Each receiving cell can accept at most `eff_lanes × per_second_acc`
          vehicles per second (default 0.6 veh/s/lane ≈ 2200 veh/hr/lane).
       b. AND at most `available_m / vehicle_length` worth of spatial slack.
       Whichever is tighter wins. Rejected particles park at the source
       cell's end and accrue cumulative_delay.
       At a merge node, both inbound edges' particles compete for the same
       trunk cell — natural merge bottleneck (item 3a).
  4. Telemetry: per-edge AND per-cell rows + per-particle position log.

A single `Scenario` describes one (network × demand × variant) run.
`simulate(scenario, ...)` returns (telemetry_df, particles_df, summary).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import networkx as nx
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_SIM   = PROJECT_ROOT / "config" / "simulation.yaml"

VEHICLE_CLASSES = ("car", "hgv", "motorcycle")


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursive dict merge — values in `overlay` override `base`,
    sub-dicts merged at every level rather than wholesale replaced."""
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_engine_config() -> dict:
    if not CONFIG_SIM.exists():
        return _DEFAULT_CFG.copy()
    with open(CONFIG_SIM) as f:
        cfg = yaml.safe_load(f) or {}
    return _deep_merge(_DEFAULT_CFG, cfg)


_DEFAULT_CFG = {
    "vehicles": {
        "car":        {"length_m": 5.0, "contributes_to_occupancy": True},
        "hgv":        {"length_m": 8.0, "contributes_to_occupancy": True},
        "motorcycle": {"length_m": 0.5, "contributes_to_occupancy": False},
    },
    "greenshields": {"jam_ratio_cap": 1.0, "min_speed_mps": 0.5},
    "engine":       {
        "step_s":          1,
        "horizon_s":       600,
        "bin_seconds":     60,
        "closure_start_s": 60,
        "cell_length_m":   100.0,    # target cell size; actual cells are
                                     # equal-length within an edge, derived
                                     # from edge length / cell_length_m.
        "per_lane_acceptance_per_s": 0.6,   # ≈ 2160 veh/hr/lane saturation
    },
    "diversion":    {"saturation_threshold": 0.80,
                     "weight_growth_k":     8.0,
                     "max_divert_share":    0.40},
    "backpressure": {"downstream_threshold": 0.90},
}


# ---------------------------------------------------------------------------
# Particle and Scenario data classes
# ---------------------------------------------------------------------------

@dataclass
class Particle:
    """A homogeneous bundle of vehicles travelling along one edge cell."""
    cls:      str         # 'car' | 'hgv' | 'motorcycle'
    count:    int
    pos_m:    float       # distance from the start of the *cell*


@dataclass
class Scenario:
    network_key:     str
    direction:       str
    slot:            str
    variant:         str
    pulses:          list[dict]
    injection_count: Optional[int] = None


# ---------------------------------------------------------------------------
# Edge → cell layout
# ---------------------------------------------------------------------------

def _num_cells_for(length_m: float, cfg: dict) -> int:
    target = cfg["engine"]["cell_length_m"]
    return max(1, int(round(length_m / target)))


def _cell_length(length_m: float, num_cells: int) -> float:
    return length_m / num_cells


# ---------------------------------------------------------------------------
# Engine state
# ---------------------------------------------------------------------------

@dataclass
class _EngineState:
    cfg:       dict
    g:         nx.DiGraph
    cells:     dict[str, list[list[Particle]]] = field(default_factory=dict)   # edge → cells of particles
    num_cells: dict[str, int]                  = field(default_factory=dict)
    cell_len:  dict[str, float]                = field(default_factory=dict)
    cum_delay: dict[str, float]                = field(default_factory=dict)
    exit_counts: dict[str, dict[str, int]]     = field(default_factory=dict)
    edge_rows:     list[dict]                  = field(default_factory=list)
    cell_rows:     list[dict]                  = field(default_factory=list)
    effective_lanes: dict[str, int]            = field(default_factory=dict)
    # Source queue per inflow edge — vehicles waiting to enter when cell 0
    # is already saturated. Each item is a Particle (count + class).
    source_queue: dict[str, list[Particle]]    = field(default_factory=dict)


def _init_state(g: nx.DiGraph, cfg: dict) -> _EngineState:
    st = _EngineState(cfg=cfg, g=g)
    for u, v, d in g.edges(data=True):
        lbl = d["label"]
        n = _num_cells_for(d["length_m"], cfg)
        st.cells[lbl] = [[] for _ in range(n)]
        st.num_cells[lbl] = n
        st.cell_len[lbl]  = _cell_length(d["length_m"], n)
        st.cum_delay[lbl] = 0.0
        st.exit_counts[lbl] = {c: 0 for c in VEHICLE_CLASSES}
        st.effective_lanes[lbl] = d["lanes"]
        st.source_queue[lbl] = []
    return st


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _edge_data_by_label(g: nx.DiGraph, label: str) -> tuple[str, str, dict]:
    for u, v, d in g.edges(data=True):
        if d["label"] == label:
            return u, v, d
    raise KeyError(f"No edge labelled '{label}' on graph {g.graph.get('name')}")


def _cell_occupancy_metres(particles: list[Particle], cfg: dict) -> float:
    veh_cfg = cfg["vehicles"]
    occ = 0.0
    for p in particles:
        v = veh_cfg.get(p.cls, {})
        if v.get("contributes_to_occupancy", False):
            occ += p.count * v.get("length_m", 0.0)
    return occ


def _cell_max_m(state: _EngineState, edge_label: str) -> float:
    """Per-cell capacity = cell_length × effective_lanes."""
    eff_lanes = state.effective_lanes.get(edge_label, 0)
    return state.cell_len[edge_label] * max(eff_lanes, 0)


def _edge_total_occupancy(state: _EngineState, edge_label: str, cfg: dict) -> float:
    return sum(_cell_occupancy_metres(c, cfg) for c in state.cells[edge_label])


def _edge_max_m(state: _EngineState, edge_label: str, edge_data: dict) -> float:
    eff_lanes = state.effective_lanes.get(edge_label, edge_data["lanes"])
    return edge_data["length_m"] * max(eff_lanes, 0)


def _greenshields_speed(v_free: float, ratio: float, cfg: dict) -> float:
    cap   = cfg["greenshields"]["jam_ratio_cap"]
    floor = cfg["greenshields"]["min_speed_mps"]
    ratio = min(max(ratio, 0.0), cap)
    return max(v_free * (1.0 - ratio / cap), floor)


# ---------------------------------------------------------------------------
# Routing — diverge nodes use the *first cell* of each candidate edge for
# saturation feedback (since that's the cell that would receive the particle).
# ---------------------------------------------------------------------------

def _choose_next_edge(state: _EngineState, node: str, cls: str,
                      rng_share: float) -> Optional[str]:
    out_edges = list(state.g.out_edges(node, data=True))
    if not out_edges:
        return None
    if len(out_edges) == 1:
        return out_edges[0][2]["label"]

    diversion_cfg = state.cfg["diversion"]
    threshold = diversion_cfg["saturation_threshold"]
    growth_k  = diversion_cfg["weight_growth_k"]
    max_div   = diversion_cfg["max_divert_share"]

    base_weights: list[tuple[str, float]] = []
    for _u, _v, d in out_edges:
        lbl = d["label"]
        cap_inv = 1.0 / max(d["max_occupancy_m"], 1.0)
        edge_max = _edge_max_m(state, lbl, d)
        if edge_max <= 0:
            weight = float("inf")
        else:
            edge_occ = _edge_total_occupancy(state, lbl, state.cfg)
            ratio = edge_occ / edge_max
            overflow = max(0.0, ratio - threshold)
            weight = cap_inv * math.exp(min(growth_k * overflow, 50.0))
        base_weights.append((lbl, weight))

    main_lbl = None
    escape_lbl = None
    main_score = math.inf
    for _u, _v, d in out_edges:
        if d.get("is_escape"):
            escape_lbl = d["label"]
        else:
            cap_inv = 1.0 / max(d["max_occupancy_m"], 1.0)
            if cap_inv < main_score:
                main_score = cap_inv
                main_lbl = d["label"]

    if cls == "motorcycle" and main_lbl is not None:
        return main_lbl

    if escape_lbl is None:
        total = sum(1.0 / w for _, w in base_weights)
        cum = 0.0
        for lbl, w in base_weights:
            cum += (1.0 / w) / total
            if rng_share <= cum:
                return lbl
        return base_weights[-1][0]

    main_w   = next(w for lbl, w in base_weights if lbl == main_lbl)
    escape_w = next(w for lbl, w in base_weights if lbl == escape_lbl)
    if main_w == float("inf"):
        return escape_lbl
    if main_w <= escape_w:
        return main_lbl
    raw_share = (main_w - escape_w) / (main_w + escape_w)
    divert_share = min(raw_share, max_div)
    return escape_lbl if rng_share < divert_share else main_lbl


# ---------------------------------------------------------------------------
# Demand injection — into cell 0 of each inflow edge
# ---------------------------------------------------------------------------

def _largest_remainder(n: int, weights: list[float]) -> list[int]:
    """Distribute integer *n* across len(weights) buckets proportional to
    *weights*, preserving the total via largest-remainder rounding."""
    total_w = sum(weights) or 1.0
    raw = [n * w / total_w for w in weights]
    base = [int(x) for x in raw]
    rem = n - sum(base)
    if rem > 0:
        order = sorted(range(len(weights)), key=lambda i: raw[i] - base[i],
                       reverse=True)
        for i in order[:rem]:
            base[i] += 1
    return base


def _inject_pulse_smeared(state: _EngineState, pulse: dict, *,
                          scaling: float, offset_in_bin: int,
                          bin_secs: int) -> None:
    """
    Spread a 1-minute demand pulse uniformly across `bin_secs` seconds and
    across the network's inflow edges (weighted by their free-flow capacity).

    Per call we inject exactly the slice of the bin that corresponds to
    `offset_in_bin` (0..bin_secs-1). Apportionment is computed in two stages:
      1. n_per_edge[cls][lbl]  = bin total for cls, distributed across edges
      2. n_per_second[cls][lbl] = the (edge×class) total, distributed across
                                  the bin's seconds.
    Both stages use largest-remainder rounding so per-bin totals are exact.
    """
    inflow_labels = state.g.graph.get("inflow_edges", [])
    if not inflow_labels:
        return

    caps_for_lbl = []
    for lbl in inflow_labels:
        _, _, d = _edge_data_by_label(state.g, lbl)
        caps_for_lbl.append(d["max_occupancy_m"])

    for cls in VEHICLE_CLASSES:
        n_total = int(round(pulse.get(cls, 0) * scaling))
        if n_total <= 0:
            continue
        per_edge = _largest_remainder(n_total, caps_for_lbl)
        for lbl, edge_total in zip(inflow_labels, per_edge):
            if edge_total <= 0:
                continue
            seconds_w = [1.0] * bin_secs
            per_sec = _largest_remainder(edge_total, seconds_w)
            n_now = per_sec[offset_in_bin]
            if n_now > 0:
                # Park in the source queue; the main loop drains it into cell 0
                # subject to backpressure.
                state.source_queue[lbl].append(
                    Particle(cls=cls, count=n_now, pos_m=0.0)
                )


def _inject_pulse(state: _EngineState, pulse: dict, scaling: float = 1.0) -> None:
    inflow_labels = state.g.graph.get("inflow_edges", [])
    if not inflow_labels:
        return

    caps: list[tuple[str, float]] = []
    for lbl in inflow_labels:
        _, _, d = _edge_data_by_label(state.g, lbl)
        caps.append((lbl, d["max_occupancy_m"]))
    total_cap = sum(c for _, c in caps) or 1.0

    for cls in VEHICLE_CLASSES:
        n = int(round(pulse.get(cls, 0) * scaling))
        if n <= 0:
            continue
        raw = [(lbl, n * c / total_cap) for lbl, c in caps]
        base = [(lbl, int(x), x - int(x)) for lbl, x in raw]
        used = sum(b for _, b, _ in base)
        rem = n - used
        if rem > 0:
            order = sorted(range(len(base)), key=lambda i: base[i][2], reverse=True)
            for i in order[:rem]:
                lbl, b, frac = base[i]
                base[i] = (lbl, b + 1, frac)
        for lbl, count, _ in base:
            if count > 0:
                # Inject into cell 0 of the inflow edge at pos_m=0.
                state.cells[lbl][0].append(Particle(cls=cls, count=count, pos_m=0.0))


# ---------------------------------------------------------------------------
# Per-cell speed
# ---------------------------------------------------------------------------

def _cell_speed(state: _EngineState, edge_label: str, cell_idx: int,
                v_free: float, cfg: dict, *, motorcycle: bool) -> float:
    if motorcycle:
        return v_free
    occ = _cell_occupancy_metres(state.cells[edge_label][cell_idx], cfg)
    cap = _cell_max_m(state, edge_label)
    if cap <= 0:
        return cfg["greenshields"]["min_speed_mps"]
    ratio = occ / cap
    return _greenshields_speed(v_free, ratio, cfg)


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def simulate(scenario: Scenario, *, rng_seed: int = 42,
             cfg: Optional[dict] = None,
             g_override: Optional[nx.DiGraph] = None,
             emit_particles: bool = True
             ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    import random
    rng = random.Random(rng_seed)

    cfg = cfg or load_engine_config()
    if g_override is not None:
        g = g_override
    else:
        from simulation.networks import build_network
        g = build_network(scenario.network_key)

    horizon  = cfg["engine"]["horizon_s"]
    bin_secs = cfg["engine"]["bin_seconds"]
    closure_start = cfg["engine"]["closure_start_s"]
    per_lane_acc  = cfg["engine"]["per_lane_acceptance_per_s"]
    bp_threshold  = cfg["backpressure"]["downstream_threshold"]

    state = _init_state(g, cfg)

    particles_rows: list[dict] = []

    if scenario.injection_count is not None and scenario.pulses:
        target_total = scenario.injection_count
        observed_total = sum(p.get("car",0) + p.get("hgv",0) + p.get("motorcycle",0)
                             for p in scenario.pulses)
        scaling = (target_total / observed_total) if observed_total > 0 else 1.0
    else:
        scaling = 1.0

    closeable = g.graph.get("closeable_edge")
    closure_lane_drop = {
        "all_open":       0,
        "1_lane_closed":  1,
        "2_lanes_closed": 2,
        "full_closure":   None,
    }.get(scenario.variant, 0)

    for t in range(horizon + 1):
        # ── 1. Apply closure if scheduled ──────────────────────────────
        if t == closure_start and closeable:
            _, _, cd = _edge_data_by_label(g, closeable)
            if closure_lane_drop is None:
                state.effective_lanes[closeable] = 0
            else:
                state.effective_lanes[closeable] = max(cd["lanes"] - closure_lane_drop, 0)

        # ── 2. Inject demand pulse, smeared across the minute ──────────
        bin_idx = t // bin_secs
        if bin_idx < len(scenario.pulses):
            _inject_pulse_smeared(state, scenario.pulses[bin_idx],
                                  scaling=scaling,
                                  offset_in_bin=t % bin_secs,
                                  bin_secs=bin_secs)

        # ── 2b. Drain the source queue into cell 0 of each inflow ──────
        # Vehicles wait in the queue when cell 0 is saturated; each second
        # we try to push as much of the queue as the cell will accept.
        for lbl in state.source_queue:
            if not state.source_queue[lbl]:
                continue
            eff_lanes = state.effective_lanes.get(lbl, 0)
            if eff_lanes == 0:
                # Edge closed: keep waiting, accrue delay.
                stuck = sum(p.count for p in state.source_queue[lbl])
                state.cum_delay[lbl] += stuck * 1.0
                continue
            cell_max_m  = _cell_max_m(state, lbl)
            cell_used   = _cell_occupancy_metres(state.cells[lbl][0], cfg)
            cell_ratio  = (cell_used / cell_max_m) if cell_max_m > 0 else 1.0
            available_m = max(cell_max_m - cell_used, 0.0)

            still_queued: list[Particle] = []
            for p in state.source_queue[lbl]:
                vc = cfg["vehicles"].get(p.cls, {})
                len_per = vc.get("length_m", 0.0)
                contributes = vc.get("contributes_to_occupancy", False)

                if (not contributes) or (cell_ratio < bp_threshold):
                    state.cells[lbl][0].append(
                        Particle(cls=p.cls, count=p.count, pos_m=0.0)
                    )
                    if contributes:
                        cell_used += p.count * len_per
                        available_m = max(cell_max_m - cell_used, 0.0)
                        cell_ratio = cell_used / cell_max_m if cell_max_m > 0 else 1.0
                    continue

                max_by_space = int(available_m // max(len_per, 0.01)) if len_per > 0 else p.count
                accept_n = max(min(p.count, max_by_space), 0)
                if accept_n > 0:
                    state.cells[lbl][0].append(
                        Particle(cls=p.cls, count=accept_n, pos_m=0.0)
                    )
                    cell_used += accept_n * len_per
                    available_m = max(cell_max_m - cell_used, 0.0)
                    cell_ratio = cell_used / cell_max_m if cell_max_m > 0 else 1.0
                rejected = p.count - accept_n
                if rejected > 0:
                    still_queued.append(
                        Particle(cls=p.cls, count=rejected, pos_m=0.0)
                    )
                    state.cum_delay[lbl] += rejected * 1.0
            state.source_queue[lbl] = still_queued

        # ── 3. Advance pass — within each cell ─────────────────────────
        # `boundary_arrivals[(edge_label, cell_idx)]` collects particles
        # that overflowed cell (cell_idx-1) and want to enter cell_idx of
        # the same edge OR (when cell_idx=='exit') particles that overflowed
        # the last cell and want to enter the next edge.
        edge_to_edge_arrivals: dict[str, list[tuple[Particle, str, int]]] = {}
        # entries: (particle, src_edge_label, src_cell_idx)
        intra_edge_arrivals: dict[tuple[str,int], list[tuple[Particle,int]]] = {}
        # entries: (particle, src_cell_idx)
        sink_arrivals: list[tuple[Particle, str, int]] = []

        for src_node, dst_node, d in g.edges(data=True):
            lbl = d["label"]
            v_free = d["v_free_mps"]
            cell_len = state.cell_len[lbl]
            n_cells  = state.num_cells[lbl]
            eff_lanes = state.effective_lanes.get(lbl, d["lanes"])

            if eff_lanes == 0:
                # Fully-closed edge: count delay but freeze positions.
                stranded = sum(p.count for c in state.cells[lbl] for p in c)
                state.cum_delay[lbl] += stranded * 1.0
                continue

            for ci in range(n_cells):
                speed_motor  = _cell_speed(state, lbl, ci, v_free, cfg, motorcycle=True)
                speed_other  = _cell_speed(state, lbl, ci, v_free, cfg, motorcycle=False)
                still_here:  list[Particle] = []
                for p in state.cells[lbl][ci]:
                    speed = speed_motor if p.cls == "motorcycle" else speed_other
                    p.pos_m += speed * 1.0
                    if p.cls != "motorcycle" and speed < v_free:
                        state.cum_delay[lbl] += p.count * (1.0 - speed / v_free)
                    if p.pos_m >= cell_len:
                        # Park at cell end; queue for transfer.
                        p.pos_m = cell_len
                        if ci + 1 < n_cells:
                            intra_edge_arrivals.setdefault((lbl, ci + 1), []).append((p, ci))
                        else:
                            # Last cell → cross to next edge.
                            next_lbl = _choose_next_edge(state, dst_node, p.cls,
                                                         rng.random())
                            if next_lbl is None:
                                sink_arrivals.append((p, lbl, ci))
                            else:
                                if state.effective_lanes.get(next_lbl, 1) == 0:
                                    # Reroute around closures.
                                    fb = None
                                    for _u, _v, dd in g.out_edges(dst_node, data=True):
                                        if state.effective_lanes.get(dd["label"], dd["lanes"]) > 0:
                                            fb = dd["label"]; break
                                    if fb is None:
                                        # Park at last cell — backpressure ripple.
                                        still_here.append(p)
                                        state.cum_delay[lbl] += p.count * 1.0
                                        continue
                                    next_lbl = fb
                                edge_to_edge_arrivals.setdefault(next_lbl, []).append(
                                    (p, lbl, ci)
                                )
                    else:
                        still_here.append(p)
                state.cells[lbl][ci] = still_here

        # ── 4. Resolve intra-edge transfers (cell→cell) ────────────────
        # Each receiving cell can accept up to (lanes × per_lane_acc) vehicles
        # per second AND up to its remaining spatial slack.
        #
        # Process in reverse order (high cell idx first) so that when an
        # earlier cell pushes into the cell we're currently filling, the
        # downstream cell has already been resolved — preserves FIFO.
        # Actually — we want to process earliest-arrival-first.  Since
        # intra-edge transfers within a single second are independent
        # (one cell pushes into one downstream), process them in order.
        for (lbl, target_cell_idx), arrivals in intra_edge_arrivals.items():
            _resolve_arrivals(
                state, cfg, lbl, target_cell_idx, arrivals,
                per_lane_acc=per_lane_acc, bp_threshold=bp_threshold,
                source_edge_for_park=lbl,
            )

        # ── 5. Resolve edge-to-edge transfers (last cell of src → cell 0 of dst) ──
        # At a merge node, multiple source edges feed the same dst edge.
        # We aggregate them in arrival order so the per-second cap is shared
        # — this realises the merge-bottleneck physics (item 3a).
        for target_edge, arrivals in edge_to_edge_arrivals.items():
            _resolve_edge_arrivals(
                state, cfg, target_edge, arrivals,
                per_lane_acc=per_lane_acc, bp_threshold=bp_threshold,
            )

        # Sink arrivals — count them.
        for p, src_lbl, _ci in sink_arrivals:
            state.exit_counts[src_lbl][p.cls] += p.count

        # ── 6. Telemetry rows ──────────────────────────────────────────
        for u, v, d in g.edges(data=True):
            lbl = d["label"]
            length = d["length_m"]
            eff_lanes = state.effective_lanes.get(lbl, d["lanes"])
            occ_m = _edge_total_occupancy(state, lbl, cfg)
            max_m = max(length * eff_lanes, 1.0)
            ratio = min(occ_m / max_m, 1.0)
            cars = sum(p.count for c in state.cells[lbl] for p in c if p.cls == "car")
            hgvs = sum(p.count for c in state.cells[lbl] for p in c if p.cls == "hgv")
            mots = sum(p.count for c in state.cells[lbl] for p in c if p.cls == "motorcycle")
            v_free = d["v_free_mps"]
            v_now  = _greenshields_speed(v_free, ratio, cfg)
            state.edge_rows.append({
                "time_s":           t,
                "edge":             lbl,
                "occupancy_ratio":  round(ratio, 4),
                "occupied_m":       round(occ_m, 2),
                "max_occupancy_m":  round(max_m, 2),
                "speed_mps":        round(v_now, 3),
                "speed_kph":        round(v_now * 3.6, 2),
                "cars_on_edge":     cars,
                "hgv_on_edge":      hgvs,
                "moto_on_edge":     mots,
                "cumulative_delay": round(state.cum_delay[lbl], 2),
                "effective_lanes":  eff_lanes,
            })

            # Per-cell rows (used by the visualizer to render speed gradient)
            for ci in range(state.num_cells[lbl]):
                cell_max = _cell_max_m(state, lbl)
                cell_occ = _cell_occupancy_metres(state.cells[lbl][ci], cfg)
                cell_ratio = (cell_occ / cell_max) if cell_max > 0 else 1.0
                cell_v = _greenshields_speed(v_free, cell_ratio, cfg)
                state.cell_rows.append({
                    "time_s":   t,
                    "edge":     lbl,
                    "cell_idx": ci,
                    "ratio":    round(cell_ratio, 4),
                    "speed_kph": round(cell_v * 3.6, 2),
                    "n_cells":  state.num_cells[lbl],
                })

        # ── 7. Per-particle position log ──────────────────────────────
        if emit_particles:
            for u, v, d in g.edges(data=True):
                lbl    = d["label"]
                length = d["length_m"]
                cell_len = state.cell_len[lbl]
                n_cells = state.num_cells[lbl]
                for ci in range(n_cells):
                    for p in state.cells[lbl][ci]:
                        # Global position on edge
                        pos_global = ci * cell_len + p.pos_m
                        particles_rows.append({
                            "time_s":   t,
                            "edge":     lbl,
                            "cls":      p.cls,
                            "count":    p.count,
                            "pos_m":    round(pos_global, 2),
                            "pos_frac": round(pos_global / max(length, 1.0), 4),
                            "cell_idx": ci,
                        })

    df = pd.DataFrame(state.edge_rows)
    particles_df = pd.DataFrame(particles_rows) if emit_particles else pd.DataFrame()

    exit_total = 0
    for lbl in g.graph.get("exit_edges", []):
        exit_total += sum(state.exit_counts.get(lbl, {}).values())
    escape_total = 0
    for lbl in g.graph.get("escape_edges", []):
        escape_total += sum(state.exit_counts.get(lbl, {}).values())

    queued_total = sum(p.count for q in state.source_queue.values() for p in q)
    in_network_total = sum(p.count for cells in state.cells.values()
                           for c in cells for p in c)
    summary = {
        "network":           scenario.network_key,
        "direction":         scenario.direction,
        "slot":              scenario.slot,
        "variant":           scenario.variant,
        "exit_total":        exit_total,
        "escape_total":      escape_total,
        "queued_at_entry":   queued_total,
        "still_in_network":  in_network_total,
        "total_delay_veh_s": round(sum(state.cum_delay.values()), 2),
        "scaling_applied":   round(scaling, 4),
    }
    return df, particles_df, summary


# ---------------------------------------------------------------------------
# Capacity-capped transfer helpers
# ---------------------------------------------------------------------------

def _resolve_arrivals(state: _EngineState, cfg: dict,
                      target_edge: str, target_cell_idx: int,
                      arrivals: list[tuple[Particle, int]],
                      *, per_lane_acc: float, bp_threshold: float,
                      source_edge_for_park: str) -> None:
    """
    Resolve particles wanting to enter cell `target_cell_idx` of `target_edge`.
    Source is the previous cell of the same edge.
    """
    """
    Intra-edge cell→cell transfer.
    Free movement below the backpressure threshold; capped by spatial slack
    only above it. Cell-traversal time on the source already paces arrivals
    so no per-second budget is needed at this scale.
    """
    eff_lanes = state.effective_lanes.get(target_edge, 0)
    if eff_lanes == 0:
        for p, src_ci in arrivals:
            cell_len_src = state.cell_len[source_edge_for_park]
            state.cells[source_edge_for_park][src_ci].append(
                Particle(cls=p.cls, count=p.count, pos_m=cell_len_src)
            )
            state.cum_delay[target_edge] += p.count * 1.0
        return

    cell_max_m = _cell_max_m(state, target_edge)
    cell_used  = _cell_occupancy_metres(state.cells[target_edge][target_cell_idx], cfg)
    cell_ratio = cell_used / cell_max_m if cell_max_m > 0 else 1.0
    available_m = max(cell_max_m - cell_used, 0.0)

    for p, src_ci in arrivals:
        veh_cfg = cfg["vehicles"].get(p.cls, {})
        len_per = veh_cfg.get("length_m", 0.0)
        contributes = veh_cfg.get("contributes_to_occupancy", False)

        if (not contributes) or (cell_ratio < bp_threshold):
            state.cells[target_edge][target_cell_idx].append(
                Particle(cls=p.cls, count=p.count, pos_m=0.0)
            )
            if contributes:
                cell_used += p.count * len_per
                available_m = max(cell_max_m - cell_used, 0.0)
                cell_ratio = cell_used / cell_max_m if cell_max_m > 0 else 1.0
            continue

        max_by_space = int(available_m // max(len_per, 0.01)) if len_per > 0 else p.count
        accept_n = max(min(p.count, max_by_space), 0)
        if accept_n > 0:
            state.cells[target_edge][target_cell_idx].append(
                Particle(cls=p.cls, count=accept_n, pos_m=0.0)
            )
            cell_used += accept_n * len_per
            available_m = max(cell_max_m - cell_used, 0.0)
            cell_ratio = cell_used / cell_max_m if cell_max_m > 0 else 1.0
        rejected = p.count - accept_n
        if rejected > 0:
            cell_len_src = state.cell_len[source_edge_for_park]
            state.cells[source_edge_for_park][src_ci].append(
                Particle(cls=p.cls, count=rejected, pos_m=cell_len_src)
            )
            state.cum_delay[source_edge_for_park] += rejected * 1.0


def _resolve_edge_arrivals(state: _EngineState, cfg: dict,
                           target_edge: str,
                           arrivals: list[tuple[Particle, str, int]],
                           *, per_lane_acc: float, bp_threshold: float) -> None:
    """
    Resolve particles wanting to cross from various source edges into cell 0
    of `target_edge`. At a merge node, both source edges feed the same target
    so they share a single per-second acceptance budget — natural merge
    bottleneck (item 3a).
    """
    """
    Edge→edge transfer (last cell of source → cell 0 of target).

    Below the backpressure threshold: free transfer (cell-traversal time on
    the source edge already paces arrivals naturally). Above the threshold,
    transfers are capped by remaining spatial slack on the target's first
    cell. When the trunk's first cell saturates, two inflow edges both lose
    capacity to push in — that's the merge bottleneck (item 3a).

    Motorcycles always pass through (queue-filtering).
    """
    eff_lanes = state.effective_lanes.get(target_edge, 0)
    if eff_lanes == 0:
        for p, src_lbl, src_ci in arrivals:
            src_cell_len = state.cell_len[src_lbl]
            state.cells[src_lbl][src_ci].append(
                Particle(cls=p.cls, count=p.count, pos_m=src_cell_len)
            )
            state.cum_delay[src_lbl] += p.count * 1.0
        return

    cell_max_m = _cell_max_m(state, target_edge)
    cell_used  = _cell_occupancy_metres(state.cells[target_edge][0], cfg)
    cell_ratio = cell_used / cell_max_m if cell_max_m > 0 else 1.0
    available_m = max(cell_max_m - cell_used, 0.0)

    for p, src_lbl, src_ci in arrivals:
        veh_cfg = cfg["vehicles"].get(p.cls, {})
        len_per = veh_cfg.get("length_m", 0.0)
        contributes = veh_cfg.get("contributes_to_occupancy", False)

        if not contributes:
            state.cells[target_edge][0].append(
                Particle(cls=p.cls, count=p.count, pos_m=0.0)
            )
            continue

        if cell_ratio < bp_threshold:
            state.cells[target_edge][0].append(
                Particle(cls=p.cls, count=p.count, pos_m=0.0)
            )
            cell_used += p.count * len_per
            available_m = max(cell_max_m - cell_used, 0.0)
            cell_ratio = cell_used / cell_max_m if cell_max_m > 0 else 1.0
            continue

        max_by_space = int(available_m // max(len_per, 0.01)) if len_per > 0 else p.count
        accept_n = max(min(p.count, max_by_space), 0)
        if accept_n > 0:
            state.cells[target_edge][0].append(
                Particle(cls=p.cls, count=accept_n, pos_m=0.0)
            )
            cell_used += accept_n * len_per
            available_m = max(cell_max_m - cell_used, 0.0)
            cell_ratio = cell_used / cell_max_m if cell_max_m > 0 else 1.0
        rejected = p.count - accept_n
        if rejected > 0:
            src_cell_len = state.cell_len[src_lbl]
            state.cells[src_lbl][src_ci].append(
                Particle(cls=p.cls, count=rejected, pos_m=src_cell_len)
            )
            state.cum_delay[src_lbl] += rejected * 1.0
