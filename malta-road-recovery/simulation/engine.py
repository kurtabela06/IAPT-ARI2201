"""
simulation/engine.py
====================
Custom NetworkX-based traffic engine.

Macro-particle model: instead of tracking each vehicle individually we track
"particles" on each edge — small bundles of homogeneous vehicles that share
a position offset. Each particle carries (class, count, distance_on_edge).
This keeps the simulation fast (≤ 1 ms per step on Phase 2 networks) while
still resolving heterogeneous fleet dynamics.

Dynamics per simulation second:

  1.  Inject demand for the current 1-minute bin (if any), apportioned across
      the network's inflow edges by their share of total inflow capacity.

  2.  For each edge, compute spatial occupancy:
            occupied_m = sum(car.count*5 + hgv.count*8 over particles on edge)
            ratio      = occupied_m / max_occupancy_m   (clamped [0, 1])

  3.  For each particle on each edge, advance distance_on_edge:
            cars/HGV : v = v_free * (1 - ratio)        (Greenshields)
            motos    : v = v_free                       (queue-filter)
      …subject to a configured `min_speed_mps` floor.

  4.  Particles that cross the edge end transfer to a downstream edge chosen
      by routing logic (see `_choose_next_edge`). At diverge nodes we use
      `nx.shortest_path` over weights inflated by the saturation of the
      downstream edges — this realises Q3 dynamic diversion / Pressure-Valve.

  5.  Telemetry: per-edge metrics are written to a list-of-dicts that the
      caller turns into a pandas DataFrame at the end of the run. The
      `cumulative_delay` column accumulates the difference between actual
      and free-flow travel time across all vehicles on the edge.

A single `Scenario` object describes one (network × demand × variant) run.
A scenario is run with `simulate(scenario, ...)` returning the telemetry
DataFrame plus a `summary` dict (exit count, total delay, recovery flags).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import networkx as nx
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Defaults — overridden by config/simulation.yaml when run via the CLI.
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_SIM   = PROJECT_ROOT / "config" / "simulation.yaml"

VEHICLE_CLASSES = ("car", "hgv", "motorcycle")


def load_engine_config() -> dict:
    if not CONFIG_SIM.exists():
        return _DEFAULT_CFG.copy()
    with open(CONFIG_SIM) as f:
        cfg = yaml.safe_load(f)
    out = _DEFAULT_CFG.copy()
    out.update(cfg or {})
    return out


_DEFAULT_CFG = {
    "vehicles": {
        "car":        {"length_m": 5.0, "contributes_to_occupancy": True},
        "hgv":        {"length_m": 8.0, "contributes_to_occupancy": True},
        "motorcycle": {"length_m": 0.5, "contributes_to_occupancy": False},
    },
    "greenshields": {"jam_ratio_cap": 1.0, "min_speed_mps": 0.5},
    "engine":       {"step_s": 1, "horizon_s": 600, "bin_seconds": 60,
                     "closure_start_s": 60},
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
    """A homogeneous bundle of vehicles travelling along one edge."""
    cls:      str         # 'car' | 'hgv' | 'motorcycle'
    count:    int
    pos_m:    float       # distance from edge source, in metres


@dataclass
class Scenario:
    """One experiment: a network, a per-bin demand schedule, and an incident."""
    network_key:  str                          # 'RNB' | 'RSB' | 'VEB' | 'VWB'
    direction:    str                          # 'regional_north' | …  (CSV name)
    slot:         str                          # '08:00' / '10am' / …
    variant:      str                          # 'all_open' | '1_lane_closed' | 'full_closure'
    pulses:       list[dict]                   # one dict per 1-min bin, e.g.
                                               #   {'car':30,'hgv':5,'motorcycle':3,'incident_status':'clear'}
    injection_count: Optional[int] = None      # set by calibrator (Q4)


# ---------------------------------------------------------------------------
# Engine state
# ---------------------------------------------------------------------------

@dataclass
class _EngineState:
    cfg:       dict
    g:         nx.DiGraph
    particles: dict[str, list[Particle]] = field(default_factory=dict)   # edge_label → particles
    cum_delay: dict[str, float]          = field(default_factory=dict)   # edge_label → seconds
    exit_counts: dict[str, dict[str, int]] = field(default_factory=dict) # edge_label → {cls: count}
    rows:      list[dict]                = field(default_factory=list)
    effective_lanes: dict[str, int]      = field(default_factory=dict)   # closure-aware


def _init_state(g: nx.DiGraph, cfg: dict) -> _EngineState:
    st = _EngineState(cfg=cfg, g=g)
    for u, v, d in g.edges(data=True):
        lbl = d["label"]
        st.particles[lbl] = []
        st.cum_delay[lbl] = 0.0
        st.exit_counts[lbl] = {c: 0 for c in VEHICLE_CLASSES}
        st.effective_lanes[lbl] = d["lanes"]
    return st


# ---------------------------------------------------------------------------
# Helpers — edge lookup, occupancy, speed
# ---------------------------------------------------------------------------

def _edge_data_by_label(g: nx.DiGraph, label: str) -> tuple[str, str, dict]:
    for u, v, d in g.edges(data=True):
        if d["label"] == label:
            return u, v, d
    raise KeyError(f"No edge labelled '{label}' on graph {g.graph.get('name')}")


def _occupancy_metres(particles: list[Particle], cfg: dict) -> float:
    veh_cfg = cfg["vehicles"]
    occ = 0.0
    for p in particles:
        v = veh_cfg.get(p.cls, {})
        if v.get("contributes_to_occupancy", False):
            occ += p.count * v.get("length_m", 0.0)
    return occ


def _max_occupancy_for(state: _EngineState, label: str, edge_data: dict) -> float:
    """Max occupancy in metres, accounting for closures (effective_lanes)."""
    eff_lanes = state.effective_lanes.get(label, edge_data["lanes"])
    return edge_data["length_m"] * max(eff_lanes, 0)


def _greenshields_speed(v_free: float, ratio: float, cfg: dict) -> float:
    cap   = cfg["greenshields"]["jam_ratio_cap"]
    floor = cfg["greenshields"]["min_speed_mps"]
    ratio = min(max(ratio, 0.0), cap)
    return max(v_free * (1.0 - ratio / cap), floor)


# ---------------------------------------------------------------------------
# Routing — choose the next edge at a node, with dynamic diversion
# ---------------------------------------------------------------------------

def _choose_next_edge(
    state:     _EngineState,
    node:      str,
    cls:       str,
    rng_share: float,         # uniform [0,1) random number for the particle
) -> Optional[str]:
    """
    Pick the next edge label leaving *node*. If the node has multiple
    outgoing edges, weight them by free-flow capacity then inflate weights
    by saturation (Q3 dynamic diversion / Pressure-Valve).

    Motorcycles never divert — they always take the primary path because
    they are queue-filtering and don't experience congestion.
    """
    out_edges = list(state.g.out_edges(node, data=True))
    if not out_edges:
        return None

    if len(out_edges) == 1:
        return out_edges[0][2]["label"]

    # Build per-edge weights. Default = inverse free-flow capacity (so wider,
    # faster edges are preferred). Then inflate weights when downstream is
    # saturated, pushing flow onto the escape route.
    diversion_cfg = state.cfg["diversion"]
    threshold = diversion_cfg["saturation_threshold"]
    growth_k  = diversion_cfg["weight_growth_k"]
    max_div   = diversion_cfg["max_divert_share"]

    base_weights: list[tuple[str, float]] = []
    for _u, _v, d in out_edges:
        lbl = d["label"]
        cap_inv = 1.0 / max(d["max_occupancy_m"], 1.0)

        max_occ = _max_occupancy_for(state, lbl, d)
        if max_occ <= 0:
            # Edge fully closed — treat as infinitely costly so diversion fires.
            weight = float("inf")
        else:
            ratio = _occupancy_metres(state.particles[lbl], state.cfg) / max_occ
            overflow = max(0.0, ratio - threshold)
            weight = cap_inv * math.exp(min(growth_k * overflow, 50.0))
        base_weights.append((lbl, weight))

    # Identify the "main" edge (lowest non-escape weight at free flow) and
    # the "escape" edge (`is_escape=True`). For motos always pick main.
    main_lbl = None
    escape_lbl = None
    main_score = math.inf
    for u, v, d in out_edges:
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
        # No escape edge — distribute by inverse weight.
        total = sum(1.0 / w for _, w in base_weights)
        cum = 0.0
        for lbl, w in base_weights:
            cum += (1.0 / w) / total
            if rng_share <= cum:
                return lbl
        return base_weights[-1][0]

    # Compute divert share = how saturated the main edge is, capped by max_div.
    # Special case: if main is fully closed (weight=inf) divert *everything* —
    # the cap only applies to dynamic-spill cases, not to hard closures.
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
# Demand injection
# ---------------------------------------------------------------------------

def _inject_pulse(state: _EngineState, pulse: dict, scaling: float = 1.0) -> None:
    """
    Apportion this minute's demand across the network's inflow edges by their
    share of total inflow free-flow capacity (max_occupancy_m). Vehicles
    arrive at the start of each inflow edge as a single particle.
    """
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
        # Distribute via largest-remainder rounding so totals are preserved.
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
                state.particles[lbl].append(Particle(cls=cls, count=count, pos_m=0.0))


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def simulate(scenario: Scenario,
             *,
             rng_seed: int = 42,
             cfg: Optional[dict] = None,
             g_override: Optional[nx.DiGraph] = None,
             emit_particles: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Run a single Scenario.

    Returns (telemetry_df, particles_df, summary_dict):
      - telemetry_df : one row per (sim_second × edge)
      - particles_df : one row per (sim_second × particle), giving the exact
                       position of every vehicle bundle on every edge so
                       the visualiser can interpolate dot positions smoothly
      - summary_dict : aggregate metrics for the run
    """
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

    # Backpressure trigger: when downstream occupancy ≥ this, transfers are
    # capped by the remaining available metres. Below it, transfers are free.
    backpressure_threshold = cfg.get("backpressure", {}).get("downstream_threshold", 0.90)

    state = _init_state(g, cfg)

    # Per-particle position log (Fix 1b + Fix 3 visualisation).
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
        # ── 1. Apply closure if scheduled ────────────────────────────────
        if t == closure_start and closeable and closure_lane_drop is not None:
            _, _, d = _edge_data_by_label(g, closeable)
            new_lanes = max(d["lanes"] - closure_lane_drop, 0)
            state.effective_lanes[closeable] = new_lanes
        if t == closure_start and closeable and closure_lane_drop is None:
            state.effective_lanes[closeable] = 0

        # ── 2. Inject this minute's demand pulse ─────────────────────────
        if t % bin_secs == 0:
            bin_idx = t // bin_secs
            if bin_idx < len(scenario.pulses):
                _inject_pulse(state, scenario.pulses[bin_idx], scaling=scaling)

        # ── 3. First pass: advance positions in place, collect transfer requests
        # ────────────────────────────────────────────────────────────────
        # `transfer_requests` maps target_edge_label → list of (Particle, src_edge)
        # tuples, in arrival order.  We resolve them after every edge has
        # advanced, so capacity decisions see all arrivals at once.
        transfer_requests: dict[str, list[tuple[Particle, str]]] = {}
        sink_arrivals: list[tuple[Particle, str]] = []   # particles exiting the network

        for src_node, dst_node, d in g.edges(data=True):
            lbl    = d["label"]
            length = d["length_m"]
            v_free = d["v_free_mps"]

            eff_lanes = state.effective_lanes.get(lbl, d["lanes"])
            if eff_lanes == 0:
                stranded = sum(p.count for p in state.particles[lbl])
                state.cum_delay[lbl] += stranded * 1.0
                continue

            occ_m = _occupancy_metres(state.particles[lbl], cfg)
            max_m = max(length * eff_lanes, 1.0)
            ratio = occ_m / max_m
            v_other = _greenshields_speed(v_free, ratio, cfg)

            still_here:  list[Particle] = []
            for p in state.particles[lbl]:
                speed = v_free if p.cls == "motorcycle" else v_other
                p.pos_m += speed * 1.0
                if p.cls != "motorcycle" and speed < v_free:
                    state.cum_delay[lbl] += p.count * (1.0 - speed / v_free)
                if p.pos_m >= length:
                    # Park at the edge end; resolution happens in pass 2.
                    p.pos_m = length
                    next_lbl = _choose_next_edge(state, dst_node, p.cls, rng.random())
                    if next_lbl is None:
                        # No outgoing edge → genuine network exit (sink).
                        sink_arrivals.append((p, lbl))
                    else:
                        # Reroute around fully-closed downstream edges.
                        if state.effective_lanes.get(next_lbl, 1) == 0:
                            fallback = None
                            for _u, _v, dd in g.out_edges(dst_node, data=True):
                                if state.effective_lanes.get(dd["label"], dd["lanes"]) > 0:
                                    fallback = dd["label"]
                                    break
                            if fallback is None:
                                # Every downstream edge is closed — park the
                                # particle here and accrue delay. This is the
                                # backpressure trigger that ripples upstream.
                                still_here.append(p)
                                state.cum_delay[lbl] += p.count * 1.0
                                continue
                            next_lbl = fallback
                        transfer_requests.setdefault(next_lbl, []).append((p, lbl))
                else:
                    still_here.append(p)
            state.particles[lbl] = still_here

        # ── 4. Capacity-capped transfers (Backpressure / Fix 1) ──────────
        # For each target edge, accept particles in arrival order until the
        # remaining capacity is exhausted (only when the edge is at or above
        # the backpressure threshold). Rejected particles stay parked at the
        # end of their source edge — they accrue delay there and contribute
        # to upstream occupancy, causing the queue to ripple upstream.
        for target, arrivals in transfer_requests.items():
            _, _, td = _edge_data_by_label(g, target)
            tgt_eff_lanes = state.effective_lanes.get(target, td["lanes"])
            tgt_max_m   = td["length_m"] * max(tgt_eff_lanes, 0)
            tgt_used    = _occupancy_metres(state.particles[target], cfg)
            tgt_ratio   = (tgt_used / tgt_max_m) if tgt_max_m > 0 else 1.0
            available_m = max(tgt_max_m - tgt_used, 0.0)

            for p, src_lbl in arrivals:
                veh_cfg = cfg["vehicles"].get(p.cls, {})
                len_per = veh_cfg.get("length_m", 0.0)
                contributes = veh_cfg.get("contributes_to_occupancy", False)

                # Free transfer when target is below backpressure threshold,
                # OR the particle doesn't contribute to occupancy AND target
                # is not fully closed (motorcycles squeeze through queues).
                free_transfer = (
                    tgt_eff_lanes > 0
                    and (not contributes or tgt_ratio < backpressure_threshold)
                )

                if free_transfer:
                    state.particles[target].append(
                        Particle(cls=p.cls, count=p.count, pos_m=0.0)
                    )
                    if contributes:
                        added = p.count * len_per
                        tgt_used += added
                        available_m = max(tgt_max_m - tgt_used, 0.0)
                        tgt_ratio = tgt_used / tgt_max_m if tgt_max_m > 0 else 1.0
                    continue

                # Capped transfer — accept up to available capacity, park rest.
                accept_n = int(available_m // max(len_per, 0.01)) if len_per > 0 else p.count
                accept_n = max(0, min(accept_n, p.count))
                if accept_n > 0:
                    state.particles[target].append(
                        Particle(cls=p.cls, count=accept_n, pos_m=0.0)
                    )
                    added = accept_n * len_per
                    tgt_used += added
                    available_m = max(tgt_max_m - tgt_used, 0.0)
                    tgt_ratio = tgt_used / tgt_max_m if tgt_max_m > 0 else 1.0
                # The rest are pushed back onto the source edge, parked at
                # the end. They will accrue delay until headroom opens up.
                rejected = p.count - accept_n
                if rejected > 0:
                    src_len = next(d["length_m"] for _u,_v,d in g.edges(data=True)
                                   if d["label"] == src_lbl)
                    state.particles[src_lbl].append(
                        Particle(cls=p.cls, count=rejected, pos_m=src_len)
                    )
                    # Each rejected vehicle loses a full second of free flow:
                    state.cum_delay[src_lbl] += rejected * 1.0

        # Sink arrivals — count them.
        for p, src_lbl in sink_arrivals:
            state.exit_counts[src_lbl][p.cls] += p.count

        # ── 5. Telemetry row per edge ────────────────────────────────────
        for u, v, d in g.edges(data=True):
            lbl = d["label"]
            length = d["length_m"]
            occ_m = _occupancy_metres(state.particles[lbl], cfg)
            max_m = max(length * state.effective_lanes.get(lbl, d["lanes"]), 1.0)
            ratio = min(occ_m / max_m, 1.0)
            cars = sum(p.count for p in state.particles[lbl] if p.cls == "car")
            hgvs = sum(p.count for p in state.particles[lbl] if p.cls == "hgv")
            mots = sum(p.count for p in state.particles[lbl] if p.cls == "motorcycle")
            v_free = d["v_free_mps"]
            v_now  = _greenshields_speed(v_free, ratio, cfg)
            state.rows.append({
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
                "effective_lanes":  state.effective_lanes.get(lbl, d["lanes"]),
            })

        # ── 6. Per-particle position log (Fix 1b / Fix 3) ────────────────
        if emit_particles:
            for u, v, d in g.edges(data=True):
                lbl = d["label"]
                length = d["length_m"]
                for p in state.particles[lbl]:
                    particles_rows.append({
                        "time_s":   t,
                        "edge":     lbl,
                        "cls":      p.cls,
                        "count":    p.count,
                        "pos_m":    round(p.pos_m, 2),
                        "pos_frac": round(p.pos_m / max(length, 1.0), 4),
                    })

    df = pd.DataFrame(state.rows)
    particles_df = pd.DataFrame(particles_rows) if emit_particles else pd.DataFrame()

    exit_total = 0
    for lbl in g.graph.get("exit_edges", []):
        exit_total += sum(state.exit_counts.get(lbl, {}).values())
    escape_total = 0
    for lbl in g.graph.get("escape_edges", []):
        escape_total += sum(state.exit_counts.get(lbl, {}).values())

    summary = {
        "network":           scenario.network_key,
        "direction":         scenario.direction,
        "slot":              scenario.slot,
        "variant":           scenario.variant,
        "exit_total":        exit_total,
        "escape_total":      escape_total,
        "total_delay_veh_s": round(sum(state.cum_delay.values()), 2),
        "scaling_applied":   round(scaling, 4),
    }
    return df, particles_df, summary
