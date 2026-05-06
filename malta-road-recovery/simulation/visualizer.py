"""
simulation/visualizer.py
========================
Matplotlib heatmap + particle animation for a pre-computed scenario.

Reads two CSVs from the scenario cache:
  • <name>.csv             per-edge telemetry  (one row per sim_second × edge)
  • <name>.particles.csv   particle positions  (one row per sim_second × particle)

Visual layers (all rendered concurrently in every 1-second frame):

  1. Static graph layout (hardcoded per network — see networks.NODE_LAYOUTS).
  2. Edge colour by occupancy ratio:
        green   < 40 %   capacity
        orange  40 – 80 % capacity
        red     > 80 %   capacity
  3. Edge line-style by closure state:
        solid   open / partially closed (multi-lane edge with ≥ 1 lane open)
        dashed  fully closed (effective_lanes = 0)
  4. Edge label at midpoint shows the academic display_name plus the
     real-time occupancy % and speed (km/h).
  5. Concurrent particle dots — colour-coded by vehicle class, positioned at
     each particle's `pos_frac` along its edge.
  6. Telemetry banner at the top of the figure: scenario name, sim time,
     and a running cross-network mean speed (km/h).

Closing the window returns control to the persistent CLI menu.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as anim
import networkx as nx
import pandas as pd

from simulation.networks import build_network, DIRECTION_TO_NETWORK, get_layout


# Colours — vehicle dots
CLASS_COLOR = {
    "car":         "#1f4e79",   # navy blue (cars)
    "hgv":         "#a93226",   # dark red (HGVs)
    "motorcycle":  "#117a65",   # teal (motorcycles)
}

# Heatmap palette
COL_GREEN  = "#2ecc71"
COL_ORANGE = "#e67e22"
COL_RED    = "#e74c3c"
COL_CLOSED = "#1c1c1c"     # dashed line colour for fully-closed edges

# Maximum dots to draw per particle bundle (caps GUI cost).
MAX_DOTS_PER_BUNDLE = 50


def _ratio_colour(ratio: float) -> str:
    if ratio < 0.40:
        return COL_GREEN
    if ratio < 0.80:
        return COL_ORANGE
    return COL_RED


def _expand_particles_to_dots(particles_at_t: pd.DataFrame) -> pd.DataFrame:
    """One row per particle → multiple rows (one per visible dot)."""
    if particles_at_t.empty:
        return particles_at_t.assign(dot_x_frac=[])
    rows = []
    for _, r in particles_at_t.iterrows():
        n = min(int(r["count"]), MAX_DOTS_PER_BUNDLE)
        if n <= 0:
            continue
        if n == 1:
            rows.append({**r.to_dict(), "dot_x_frac": float(r["pos_frac"])})
            continue
        half_band = 0.02
        for i in range(n):
            t = i / max(n - 1, 1)
            offset = -half_band + t * 2 * half_band
            frac = max(0.0, min(1.0, float(r["pos_frac"]) + offset))
            rows.append({**r.to_dict(), "dot_x_frac": frac})
    return pd.DataFrame(rows)


VARIANT_LABEL = {
    "all_open":        "All Lanes Open",
    "1_lane_closed":   "1 Lane Closed",
    "2_lanes_closed":  "2 Lanes Closed",
    "full_closure":    "Full Closure",
}

DIRECTION_SHORT = {
    "regional_north": "RNB",
    "regional_south": "RSB",
    "valley_east":    "VEB",
    "valley_west":    "VWB",
}

SLOT_LABEL = {"10am": "10:00 AM", "1pm": "1:00 PM", "5pm": "5:00 PM"}


def visualise_scenario(
    csv_path:    Path,
    *,
    direction:   str,
    slot:        str,
    variant:     str,
) -> None:
    """Open an animation window. Closing it returns control."""
    if not csv_path.exists():
        print(f"  Scenario data not found: {csv_path}")
        print(f"  Please run Option 2 to pre-compute or select Option 1 for a live run.")
        return
    particles_path = csv_path.with_name(csv_path.stem + ".particles.csv")
    if not particles_path.exists():
        print(f"  Particles CSV not found: {particles_path}")
        print(f"  Re-run scenario with Option 1 (force) so the particle log is generated.")
        return

    network_key = DIRECTION_TO_NETWORK[direction]
    g = build_network(network_key)
    edge_df  = pd.read_csv(csv_path)
    part_df  = pd.read_csv(particles_path)
    times    = sorted(edge_df["time_s"].unique())
    edges    = list(g.edges(data=True))

    pos = get_layout(network_key)

    # ── Figure layout ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 9.5))
    fig.patch.set_facecolor("#fafafa")

    # Telemetry banner (top), main map (middle), legend (right side)
    banner_ax = fig.add_axes([0.04, 0.92, 0.92, 0.06])
    banner_ax.set_axis_off()
    banner_ax.set_xlim(0, 1); banner_ax.set_ylim(0, 1)

    main_ax = fig.add_axes([0.04, 0.06, 0.74, 0.84])
    main_ax.set_aspect("equal")
    main_ax.set_axis_off()
    main_ax.set_facecolor("#fafafa")

    legend_ax = fig.add_axes([0.79, 0.06, 0.18, 0.84])
    legend_ax.set_axis_off()

    # ── Telemetry banner ──────────────────────────────────────────────
    scenario_label = (
        f"{DIRECTION_SHORT[direction]}  •  {SLOT_LABEL.get(slot, slot)}  •  "
        f"{VARIANT_LABEL.get(variant, variant)}"
    )
    banner_ax.text(0.0, 0.55, scenario_label, fontsize=15,
                   ha="left", va="center", weight="bold", color="#2c3e50")
    time_text  = banner_ax.text(0.50, 0.55, "", fontsize=13, ha="center",
                                va="center", color="#34495e",
                                family="monospace")
    speed_text = banner_ax.text(1.0, 0.55, "", fontsize=13, ha="right",
                                va="center", color="#34495e",
                                family="monospace")
    banner_ax.text(0.0, 0.05, g.graph["display_name"], fontsize=10,
                   ha="left", va="center", color="#7f8c8d", style="italic")

    # ── Graph layout: nodes + static edge lines ──────────────────────
    nx.draw_networkx_nodes(g, pos, ax=main_ax,
                           node_size=380, node_color="#ecf0f1",
                           edgecolors="#2c3e50", linewidths=1.4)

    edge_lines: dict[str, plt.Line2D] = {}
    edge_texts: dict[str, plt.Text]   = {}
    for u, v, d in edges:
        x0, y0 = pos[u]; x1, y1 = pos[v]
        line, = main_ax.plot([x0, x1], [y0, y1], color="#bdc3c7", linewidth=6.0,
                             solid_capstyle="round", zorder=1)
        edge_lines[d["label"]] = line
        tx = (x0 + x1) / 2
        ty = (y0 + y1) / 2 + 0.07
        txt = main_ax.text(tx, ty, "", fontsize=8, ha="center",
                           color="#2c3e50", zorder=5, weight="bold",
                           bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                     ec="#bdc3c7", alpha=0.92))
        edge_texts[d["label"]] = txt

    # Particle scatters per class
    scatters: dict[str, plt.PathCollection] = {}
    for cls, colour in CLASS_COLOR.items():
        scatters[cls] = main_ax.scatter([], [], s=22, c=colour, zorder=6,
                                        alpha=0.92, edgecolors="white",
                                        linewidths=0.4)

    edge_pos_lookup = {d["label"]: (pos[u], pos[v]) for u, v, d in edges}
    edge_data_by_label = {d["label"]: d for _, _, d in edges}
    edge_df_idx  = edge_df.set_index(["time_s", "edge"])
    part_df_by_t = {t: gp.reset_index(drop=True) for t, gp in part_df.groupby("time_s")}

    # ── Sectioned legend on the right side ─────────────────────────────
    _draw_legend(legend_ax)

    # ── Animation update ──────────────────────────────────────────────
    def frame(t_idx: int):
        t = times[t_idx]
        time_text.set_text(f"t = {t:>3d} s / {times[-1]} s")

        # Compute network-wide mean speed (vehicle-weighted)
        try:
            t_slice = edge_df_idx.xs(t, level="time_s")
            n_veh = (t_slice["cars_on_edge"] + t_slice["hgv_on_edge"]
                     + t_slice["moto_on_edge"]).sum()
            if n_veh > 0:
                weighted = ((t_slice["cars_on_edge"] + t_slice["hgv_on_edge"]
                             + t_slice["moto_on_edge"]) * t_slice["speed_kph"]).sum()
                mean_speed = weighted / n_veh
            else:
                mean_speed = 0.0
        except KeyError:
            mean_speed = 0.0
        speed_text.set_text(f"avg speed: {mean_speed:5.1f} km/h")

        # 1. Edge colours, line styles, labels
        for u, v, d in edges:
            lbl = d["label"]
            try:
                row = edge_df_idx.loc[(t, lbl)]
            except KeyError:
                continue
            ratio = float(row["occupancy_ratio"])
            eff_lanes = int(row["effective_lanes"])
            orig_lanes = int(d["lanes"])

            # Line-style rule:
            #   single-lane edge AND eff_lanes==0  → dashed, black
            #   multi-lane edge AND eff_lanes==0   → dashed, black (full closure)
            #   any other case (incl. partial)     → solid
            fully_closed = (eff_lanes == 0)
            if fully_closed:
                edge_lines[lbl].set_color(COL_CLOSED)
                edge_lines[lbl].set_linewidth(7.0)
                edge_lines[lbl].set_linestyle((0, (4, 4)))
            else:
                edge_lines[lbl].set_color(_ratio_colour(ratio))
                edge_lines[lbl].set_linewidth(4.0 + 8.0 * min(ratio, 1.0))
                edge_lines[lbl].set_linestyle("-")

            # Edge label: academic name + occupancy % + speed
            display_name = d.get("display_name") or d["label"]
            partial = (eff_lanes < orig_lanes) and (eff_lanes > 0)
            partial_tag = "  ½cap" if partial else ""
            edge_texts[lbl].set_text(
                f"{display_name}{partial_tag}\n"
                f"{int(ratio*100)}%   {float(row['speed_kph']):.0f} km/h"
            )

        # 2. Particle dots (concurrent across all edges)
        per_class_xs = {c: [] for c in CLASS_COLOR}
        per_class_ys = {c: [] for c in CLASS_COLOR}
        rows_t = part_df_by_t.get(t)
        if rows_t is not None and not rows_t.empty:
            expanded = _expand_particles_to_dots(rows_t)
            for _, r in expanded.iterrows():
                lbl = r["edge"]; cls = r["cls"]
                if lbl not in edge_pos_lookup or cls not in per_class_xs:
                    continue
                (x0, y0), (x1, y1) = edge_pos_lookup[lbl]
                f = float(r["dot_x_frac"])
                per_class_xs[cls].append(x0 + (x1 - x0) * f)
                per_class_ys[cls].append(y0 + (y1 - y0) * f)
        for cls in CLASS_COLOR:
            xs = per_class_xs[cls]; ys = per_class_ys[cls]
            scatters[cls].set_offsets(list(zip(xs, ys)) if xs else [[0, 0]])

        return (
            list(edge_lines.values())
            + list(edge_texts.values())
            + list(scatters.values())
            + [time_text, speed_text]
        )

    # Render the t=0 state immediately so the window opens with content
    # populated (labels, telemetry text) instead of a blank frame.
    frame(0)

    a = anim.FuncAnimation(
        fig, frame, frames=len(times), interval=60, blit=False, repeat=False,
    )
    plt.show()
    del a


# ---------------------------------------------------------------------------
# Sectioned legend — right side panel
# ---------------------------------------------------------------------------

def _draw_legend(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_axis_off()

    y = 0.97
    line_h = 0.035

    def section(title: str, y_top: float) -> float:
        ax.text(0.02, y_top, title, fontsize=11, weight="bold",
                color="#2c3e50", transform=ax.transAxes)
        return y_top - 0.04

    # ── COLOUR SCALE ──
    y = section("OCCUPANCY", y)
    for label, colour in [("< 40 %  free flow", COL_GREEN),
                          ("40 – 80 %  congested", COL_ORANGE),
                          ("> 80 %  jam", COL_RED)]:
        ax.add_patch(plt.Rectangle((0.04, y - 0.018), 0.10, 0.022,
                                    facecolor=colour, edgecolor="none",
                                    transform=ax.transAxes))
        ax.text(0.18, y - 0.005, label, fontsize=9, va="center",
                color="#2c3e50", transform=ax.transAxes)
        y -= line_h

    # ── LINE STYLES ──
    y -= 0.02
    y = section("LANE STATE", y)
    style_specs = [
        ("solid",        "-",      "Open / partial closure"),
        ("dashed",       (0,(4,4)),"Full closure (all lanes)"),
    ]
    for _, ls, label in style_specs:
        line = plt.Line2D([0.04, 0.14], [y - 0.005, y - 0.005],
                          color=COL_CLOSED if ls != "-" else "#7f8c8d",
                          linewidth=5, linestyle=ls,
                          transform=ax.transAxes, solid_capstyle="round")
        ax.add_line(line)
        ax.text(0.18, y - 0.005, label, fontsize=9, va="center",
                color="#2c3e50", transform=ax.transAxes)
        y -= line_h

    # ── VEHICLE CLASSES ──
    y -= 0.02
    y = section("VEHICLE CLASS", y)
    veh_specs = [
        ("Car  (5 m)",        CLASS_COLOR["car"]),
        ("HGV  (8 m)",        CLASS_COLOR["hgv"]),
        ("Motorcycle  (0.5 m)",CLASS_COLOR["motorcycle"]),
    ]
    for label, colour in veh_specs:
        ax.scatter([0.09], [y - 0.005], s=42, c=colour, zorder=3,
                   edgecolors="white", linewidths=0.6,
                   transform=ax.transAxes)
        ax.text(0.18, y - 0.005, label, fontsize=9, va="center",
                color="#2c3e50", transform=ax.transAxes)
        y -= line_h

    # ── NOTES ──
    y -= 0.03
    ax.text(0.02, y, "NOTES", fontsize=11, weight="bold",
            color="#2c3e50", transform=ax.transAxes)
    y -= 0.04
    notes = [
        "• Edge thickness scales",
        "  with occupancy.",
        "• ½cap tag shown on",
        "  partially-closed edges.",
        "• Motorcycles never block",
        "  (queue-filtering).",
    ]
    for note in notes:
        ax.text(0.02, y, note, fontsize=8, color="#566573",
                transform=ax.transAxes)
        y -= 0.025
