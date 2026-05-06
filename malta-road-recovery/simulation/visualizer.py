"""
simulation/visualizer.py
========================
Matplotlib heatmap + particle animation for a pre-computed scenario.

Reads two CSVs from the scenario cache:
  • <name>.csv             per-edge telemetry  (one row per sim_second × edge)
  • <name>.particles.csv   particle positions  (one row per sim_second × particle)

Visual layers, all rendered concurrently every frame (Fix 3):

  1. Static graph layout (NetworkX spring layout — abstract, Q8 = c)
  2. Edge colour by occupancy ratio:
        green   < 40 %   capacity
        orange  40 – 80 % capacity
        red     > 80 %   capacity
     Edge thickness scales with occupancy.
  3. Edge label at midpoint: occupancy % + speed (km/h)
  4. Concurrent particle dots — one dot per particle, positioned at the
     particle's exact `pos_frac` along the edge. As particles transfer to
     downstream edges (including escape branches), the dots appear at the
     start of the new edge in the next frame, so the visual flow is
     continuous and split-points are visible in real time.
  5. Closure overlay — when an edge is fully closed (effective_lanes = 0)
     it's drawn with a thick black hatched stripe so the user sees what
     the engine is reacting to.

Closing the figure returns control to the caller (the persistent CLI menu).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as anim
import networkx as nx
import pandas as pd

from simulation.networks import build_network, DIRECTION_TO_NETWORK, get_layout


CLASS_COLOR = {
    "car":         "#2c3e50",   # dark blue (so it stands out on green/orange/red)
    "hgv":         "#7f1d1d",   # dark red dot
    "motorcycle":  "#1e3a8a",   # navy
}

# Maximum dots to draw per edge (avoids GUI lag on heavily-occupied edges).
MAX_DOTS_PER_EDGE = 50


def _ratio_colour(ratio: float) -> str:
    if ratio < 0.40:
        return "#2ecc71"
    if ratio < 0.80:
        return "#e67e22"
    return "#e74c3c"


def _expand_particles_to_dots(particles_at_t: pd.DataFrame) -> pd.DataFrame:
    """
    Each particle row carries `count` (potentially > 1). We turn it into
    (up to MAX_DOTS_PER_EDGE) individual dots distributed around the
    particle's pos_frac, so a 30-vehicle bundle shows as 30 visible dots
    rather than one. Distribution is small uniform jitter on either side
    of the particle's centre so dots don't all overlap.
    """
    if particles_at_t.empty:
        return particles_at_t.assign(dot_x_frac=[])
    rows = []
    for _, r in particles_at_t.iterrows():
        n = min(int(r["count"]), MAX_DOTS_PER_EDGE)
        if n <= 0:
            continue
        if n == 1:
            rows.append({**r.to_dict(), "dot_x_frac": float(r["pos_frac"])})
            continue
        # Spread n dots over a 0.04-fraction band centred on pos_frac.
        half_band = 0.02
        for i in range(n):
            t = i / max(n - 1, 1)
            offset = -half_band + t * 2 * half_band
            frac = max(0.0, min(1.0, float(r["pos_frac"]) + offset))
            rows.append({**r.to_dict(), "dot_x_frac": frac})
    return pd.DataFrame(rows)


def visualise_scenario(
    csv_path:    Path,
    *,
    direction:   str,
    slot:        str,
    variant:     str,
) -> None:
    """
    Open an animation window. Closing it returns control.
    Expects <stem>.particles.csv next to csv_path.
    """
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

    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_aspect("equal")
    ax.set_axis_off()
    title_obj = ax.set_title("", fontsize=14)

    nx.draw_networkx_nodes(g, pos, ax=ax,
                           node_size=420, node_color="#ecf0f1",
                           edgecolors="#34495e", linewidths=1.5)
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=8, font_weight="bold")

    edge_lines: dict[str, plt.Line2D] = {}
    edge_texts: dict[str, plt.Text]   = {}
    for u, v, d in edges:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        line, = ax.plot([x0, x1], [y0, y1], color="#bdc3c7", linewidth=6.0,
                        solid_capstyle="round", zorder=1)
        edge_lines[d["label"]] = line
        tx = (x0 + x1) / 2
        ty = (y0 + y1) / 2 + 0.04
        txt = ax.text(tx, ty, "", fontsize=7, ha="center",
                      color="#2c3e50", zorder=4,
                      bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                ec="none", alpha=0.7))
        edge_texts[d["label"]] = txt

    # One scatter per vehicle class so we can colour them independently.
    scatters: dict[str, plt.PathCollection] = {}
    for cls, colour in CLASS_COLOR.items():
        scatters[cls] = ax.scatter([], [], s=18, c=colour, zorder=5,
                                   alpha=0.85, label=f"{cls}")

    edge_pos_lookup = {d["label"]: (pos[u], pos[v]) for u, v, d in edges}
    edge_geom_by_label = {d["label"]: d for _, _, d in edges}

    edge_df_idx = edge_df.set_index(["time_s", "edge"])
    part_df_by_t = {t: g.reset_index(drop=True) for t, g in part_df.groupby("time_s")}

    def frame(t_idx: int):
        t = times[t_idx]
        title_obj.set_text(
            f"{direction.upper()}  |  slot {slot}  |  {variant}  |  t = {t} s"
        )

        # 1. Update edge colours and labels.
        for u, v, d in edges:
            lbl = d["label"]
            try:
                row = edge_df_idx.loc[(t, lbl)]
            except KeyError:
                continue
            ratio = float(row["occupancy_ratio"])
            eff_lanes = int(row["effective_lanes"])
            if eff_lanes == 0:
                edge_lines[lbl].set_color("#000000")
                edge_lines[lbl].set_linewidth(8.0)
                edge_lines[lbl].set_linestyle((0, (3, 3)))
            else:
                edge_lines[lbl].set_color(_ratio_colour(ratio))
                edge_lines[lbl].set_linewidth(4.0 + 8.0 * min(ratio, 1.0))
                edge_lines[lbl].set_linestyle("-")
            edge_texts[lbl].set_text(
                f"{int(ratio*100)}%\n{float(row['speed_kph']):.0f} km/h"
            )

        # 2. Update particle dots (concurrent across all edges).
        per_class_xs = {c: [] for c in CLASS_COLOR}
        per_class_ys = {c: [] for c in CLASS_COLOR}

        rows_t = part_df_by_t.get(t, None)
        if rows_t is not None and not rows_t.empty:
            expanded = _expand_particles_to_dots(rows_t)
            for _, r in expanded.iterrows():
                lbl = r["edge"]
                cls = r["cls"]
                if lbl not in edge_pos_lookup or cls not in per_class_xs:
                    continue
                (x0, y0), (x1, y1) = edge_pos_lookup[lbl]
                f = float(r["dot_x_frac"])
                per_class_xs[cls].append(x0 + (x1 - x0) * f)
                per_class_ys[cls].append(y0 + (y1 - y0) * f)

        for cls in CLASS_COLOR:
            xs = per_class_xs[cls]
            ys = per_class_ys[cls]
            scatters[cls].set_offsets(list(zip(xs, ys)) if xs else [[0, 0]])

        return (
            list(edge_lines.values())
            + list(edge_texts.values())
            + list(scatters.values())
            + [title_obj]
        )

    a = anim.FuncAnimation(
        fig, frame, frames=len(times), interval=60, blit=False, repeat=False,
    )

    handles = [
        plt.Line2D([0], [0], color="#2ecc71",  lw=6,  label="< 40 %"),
        plt.Line2D([0], [0], color="#e67e22",  lw=6,  label="40 – 80 %"),
        plt.Line2D([0], [0], color="#e74c3c",  lw=6,  label="> 80 %"),
        plt.Line2D([0], [0], color="#000000",  lw=6,  ls=(0,(3,3)),  label="closed"),
        plt.Line2D([0], [0], marker="o", lw=0,  color="#2c3e50", label="cars"),
        plt.Line2D([0], [0], marker="o", lw=0,  color="#7f1d1d", label="HGV"),
        plt.Line2D([0], [0], marker="o", lw=0,  color="#1e3a8a", label="motorcycle"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=9)

    plt.tight_layout()
    plt.show()
    del a
