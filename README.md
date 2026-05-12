# IAPT-ARI2201 — Malta Road Recovery

A traffic-flow & recovery-time analysis suite for Malta's Regional Road and
Valley Road. Built on a **custom NetworkX simulation engine** (no SUMO) with
Pandas state and Matplotlib visualisation.

---

## Phases at a glance

- **Phase 1 — Manual ground truth.** Observers fill one CSV per (direction × slot);
  `data_processing/manual_to_pulse.py` reshapes them into 1-minute demand pulses.
- **Phase 2 — Custom engine + scenario library.** Four NetworkX `DiGraph` road
  skeletons, Greenshields with spatial-occupancy density, dynamic diversion at
  escape exits / Naxxar pressure-valve, plus a 120-scenario synthetic dataset.
- **Phase 3 — Visualiser.** Matplotlib heatmap + animated dots over the graph
  layout; called interactively from the persistent CLI menu.
- **Phase 4 — ML.** XGBoost / RF / linear baselines on the 120-scenario telemetry
  to predict cumulative-delay recovery time. (Stub in the menu — future work.)

---

## Phase 1 — Manual ground-truth extraction

### Folder layout

```
data/raw/manual_counts/
    regional_north/<slot>.csv
    regional_south/<slot>.csv
    valley_east/<slot>.csv
    valley_west/<slot>.csv
```

### CSV schema (one row per 1-minute bin)

| column                | type  | required | description                                                      |
| --------------------- | ----- | -------- | ---------------------------------------------------------------- |
| timestamp_bin         | int   | yes      | 1-minute bin index (0, 1, 2, …)                                  |
| car_count             | int   | yes      | Cars crossing the line in this bin                               |
| motorcycle_count      | int   | no       | Motorcycles in this bin (defaults to 0 if column absent)         |
| heavy_vehicle_count   | int   | yes      | Buses + trucks combined                                          |
| avg_car_speed_kph     | float | yes      | Mean car speed observed for the bin (informational only)         |
| incident_status       | str   | yes      | `clear` \| `slow` \| `blocked` \| `incident`                     |

PCE / density / flow-rate are **not** stored in the CSV. The engine derives
them on the fly from spatial occupancy on each graph edge.

### Convert manual CSVs → demand pulses

```bash
cd malta-road-recovery
python data_processing/manual_to_pulse.py --all
```

Outputs land at `data/processed/pulses/<direction>/<slot>.csv` with the
schema:

```
timestamp_bin, car, hgv, motorcycle, incident_status, observed_speed_kph
```

---

## Phase 2 — Custom NetworkX engine

Phase 2 is **the core** of this project. Run it via:

```bash
cd malta-road-recovery
python main.py
```

You'll see a persistent menu. Closing a visualisation always returns you to
this menu; it never exits Python.

```
============================================================
  Malta Road Recovery — Phase 2 Engine
  Cached scenarios: 0
============================================================
  1. Run a single scenario (live)
  2. Pre-compute all scenarios (batch)
  3. View pre-computed scenario
  4. Train ML model on scenarios   [Phase 4 stub]
  5. Predict on a new scenario     [Phase 4 stub]
  6. Quit
```

### The four road skeletons

Each direction is a separate `networkx.DiGraph` with hardcoded geometry
(metres) and lane counts. See `simulation/networks.py` for the literals.

| Network | Description                          | Edges                                                                     |
| ------- | ------------------------------------ | ------------------------------------------------------------------------- |
| **RNB** | Regional Road Northbound             | mergeway_N_a (3L 140m) + underpass_N (1L 250m) → mergeway_N_b (3L 120m) → diverge → mainway_N (2L 465m) / escape_N (1L 465m) |
| **RSB** | Regional Road Southbound             | flyover_S (2L 140m) + underpass_S (1L 140m) → mergeway_S (3L 235m) → diverge → mainway_S (2L 175m) / escape_S (1L 175m) |
| **VEB** | Valley Road Eastbound                | valley_E (1L 180m) → branch → valley_cont_E (1L 350m) / **naxxar_E (1L 400m)** ← Pressure-Valve diversion |
| **VWB** | Valley Road Westbound                | valley_W_pre (1L 370m) and (psaila_W + fleur_a_W → fleur_b_W) → junction → valley_W_post (1L 130m) — main stretch 500 m total |

Every edge stores: `length_m`, `lanes`, `max_occupancy_m = length_m × lanes`,
and `v_free_mps` (from posted speed limit).

### Vehicle physics

Spatial-occupancy Greenshields model:

```
occupied_m = car_count × 5 + hgv_count × 8           (motorcycles excluded)
ratio      = occupied_m / max_occupancy_m
v_car_hgv  = v_free × (1 − ratio)         (≥ min_speed_mps floor)
v_moto     = v_free                       (always — queue-filtering)
```

Configurable via `config/simulation.yaml`:

| key                                          | default    |
| -------------------------------------------- | ---------- |
| `vehicles.car.length_m` / `hgv` / `moto`    | 5 / 8 / 0.5 |
| `greenshields.min_speed_mps`                | 0.5        |
| `engine.step_s` / `horizon_s` / `bin_seconds` | 1 / 600 / 60 |
| `engine.closure_start_s`                    | 60         |
| `engine.cell_length_m`                      | 100.0      |
| `engine.per_lane_acceptance_per_s`          | 0.6        |
| `closure.partial_throughput_factor`         | 0.5        |
| `backpressure.downstream_threshold`         | 0.90       |
| `diversion.baseline_share`                  | 0.07       |
| `diversion.saturation_threshold`            | 0.80       |
| `diversion.max_divert_share`                | 0.40       |
| `diversion.weight_growth_k`                 | 8.0        |

### Demand calibration (Q4 — fixed-injection rule)

For each (direction × slot) the **All-Lanes-Open** scenario calibrates an
**injection count**: a multiplicative scaling on the manual pulse such that
the simulated exit count matches the manual CSV total.

That same injection count is then **reused unchanged** by the closure
variants — so when you compare "all_open" vs "1_lane_closed" the only
thing that differs is the closure mechanics, never the demand. This gives
you a clean signal for **recovery time**: how long it takes the closed
variant's exit flow to return to the all-open baseline once the closure is
lifted.

Cached at `data/processed/calibration/<direction>_<slot>_injection.json`.

```bash
# Calibrate just one slot
python simulation/calibrator.py --direction regional_north --slot 10am --verbose
```

### Cell-transmission engine

Every edge is subdivided into equal-length **cells** (default ~100 m each),
configurable via `engine.cell_length_m`. Each cell holds its own particle
list and computes its own Greenshields speed from its own occupancy. This
is what makes the visible **speed gradient** along a long edge possible:
free flow at the entry of `valley_W_pre`, slow-down approaching the
bottleneck at the far end.

State shape: `state.cells[edge_label][cell_idx] : list[Particle]`. Each
particle carries `(class, count, pos_m_in_cell)`. Particles cross from
cell `i` to cell `i+1` when their `pos_m` exceeds the cell length;
particles in the last cell cross to the first cell of the next edge
(routing chosen by `_choose_next_edge`).

### Backpressure (spatial, ripple effect)

Each second the engine performs a two-pass update **per cell**:

1. **Advance pass.** Every particle moves at its cell's Greenshields-derived
   speed (motorcycles always at v_free). Particles that reach the end of a
   cell are parked at `pos_m = cell_length` and queued for transfer to the
   chosen downstream cell (next cell on the same edge, or first cell of
   the next edge).

2. **Capacity-capped transfer.** For each receiving cell, the engine
   accepts particles in arrival order while available spatial slack
   remains. Below the configurable backpressure threshold
   (`backpressure.downstream_threshold`, default 0.90) transfers are free
   — natural arrivals are paced by the cell-traversal time on the source.
   At/above the threshold, the engine caps each transfer by the
   remaining `available_m`: accepted vehicles enter the target at
   `pos_m = 0`; **rejected vehicles park at the source cell's end** and
   accrue `cumulative_delay`. Their occupancy counts on the source cell,
   which raises the source's saturation, which then back-propagates the
   same way through the upstream cell, all the way to the entry edge.

3. **Source queue.** Vehicles arriving from manual demand wait in a
   per-edge `source_queue` if cell 0 of their inflow edge is already
   saturated. Each second the engine drains the queue into cell 0 subject
   to backpressure. This is how the **merge bottleneck (item 3a)**
   manifests: when the trunk's first cell saturates, both inflow edges'
   first cells back up, then their source queues. Mass is preserved
   exactly: every vehicle is either still queued at its source, still in
   the network, or counted in `exit_total` / `escape_total`.

4. **Hard closure.** When `effective_lanes = 0`, every downstream-bound
   transfer is rejected. If no alternative edge exists, particles park on
   the upstream cell and accrue delay until the closure lifts.

5. **Demand smearing.** Each minute's pulse is spread uniformly across
   the 60 seconds of the bin so injection looks like a steady stream
   rather than a sudden burst. Largest-remainder rounding preserves bin
   totals exactly.

### Closures act on the Main Way only

Closures (`1_lane_closed`, `2_lanes_closed`, `full_closure`) reduce
`effective_lanes` strictly on the **post-diverge Main Way** edge — never the
trunk and never the escape. That keeps the diverge node reachable so traffic
can peel off to the escape route while the Main Way fills with stranded cars.
Per network:

| Network | Closeable edge | Reduces lanes |
| --- | --- | --- |
| RNB | `mainway_N`     | 2 → 1 / 0 |
| RSB | `mainway_S`     | 2 → 1 / 0 |
| VEB | `valley_cont_E` | 1 → 0     |
| VWB | `valley_W_post` | 1 → 0 (no escape route exists) |

### Recovery API and extended horizons

The engine accepts a dynamic `horizon_s` per `Scenario` (default 600 s).
Spec-supported values: 600, 900, 1200, 1800, 2400 s. The "Closure + Recovery"
mode runs for `2 × H` seconds: closure phase 0 → H, recovery phase H → 2H.

**Looped CSV inflow**: when the horizon exceeds the manual CSV's bin count,
the engine **loops** the CSV back to bin 0. So a 1800 s recovery run cycles
through the 10 manual bins ~3 times. Demand is *not* artificially reduced
during the recovery phase — a queue forming during closure has to drain
against the same realistic demand pattern that produced it.

A `RecoveryPolicy(start_s, new_lane_count)` attached to a `Scenario`
triggers a mid-run lane swap. Example:

```python
from simulation.engine import Scenario, RecoveryPolicy, simulate

sc = Scenario(
    network_key="RSB", direction="regional_south", slot="5pm",
    variant="1_lane_closed",        # closure starts at t=60
    pulses=pulses,
    horizon_s=1200,                  # extended horizon
    recovery=RecoveryPolicy(start_s=600, new_lane_count=2),  # full reopen at t=600
)
df, particles_df, summary = simulate(sc)
print(summary["recovery_time_s"])     # seconds elapsed from trigger to V_avg ≥ threshold
```

**Recovery thresholds** (network-wide vehicle-weighted rolling-average
speed, configurable in `simulation.yaml`):

| Network | Threshold |
| --- | --- |
| Regional (RNB / RSB) | 60 km/h |
| Valley (VEB / VWB)   | 45 km/h |

The summary dict surfaces:

| Key | Description |
| --- | --- |
| `horizon_s` | The horizon actually used (override or config default) |
| `recovery_triggered_s` | When the lane swap fired (or `None`) |
| `recovery_threshold_kph` | The threshold used for this network |
| `recovered_at_s` | First sim_second after trigger when rolling V_avg ≥ threshold |
| `recovery_time_s` | `recovered_at_s − recovery_triggered_s` |
| `final_rolling_speed_kph` | Rolling V_avg over the last `rolling_window_s` (default 30) |

### Transition Penalty (merge friction)

Real merges slow traffic for the first ~30 m of the destination edge as
drivers adjust. The engine applies a multiplicative speed penalty (drawn
once per cell from `[speed_factor_min, speed_factor_max]`, default
`[0.70, 0.80]` = 20–30 % reduction) to every cell that lies within
`merge_zone_m` (default 30 m) of a merge-target edge — i.e. any edge
whose source node has ≥ 2 inbound edges. Persists during both closure and
recovery phases. Motorcycles are subject to the penalty too because real
riders also slow at junctions, even though they don't contribute to
occupancy density.

Configuration knobs in `config/simulation.yaml`:

| Key | Default |
| --- | --- |
| `merge_friction.merge_zone_m` | 30.0 |
| `merge_friction.speed_factor_min` | 0.70 |
| `merge_friction.speed_factor_max` | 0.80 |
| `recovery.rolling_window_s` | 30 |
| `recovery.regional_threshold_kph` | 60.0 |
| `recovery.valley_threshold_kph` | 45.0 |

### Bottleneck-cap acceptance accumulator

The Bottleneck Penalty's per-second cap is a *fractional* rate (e.g. 0.6
veh/s for a 2L→1L closure). The engine implements this via a **per-edge
acceptance-credit accumulator**:

- Each tick adds `partial_cap` worth of credit to the edge's credit pool.
- Each accepted contributing vehicle consumes 1.0 credit.
- The pool is bounded by `credit_buffer_s × cap` (default 3 seconds of
  saved-up throughput) so unused capacity does not accumulate
  indefinitely. Without this cap, a road that sat empty for a while
  would suddenly accept dozens of vehicles in one tick when demand
  arrived — which produced inconsistent behaviour between
  steady-state and post-recovery runs that share the same effective_lanes.
- Vehicles arrive but credit is empty → particle stays parked on the
  upstream cell, contributes to backpressure.

This is what makes the partially-reopened lane during recovery absorb
traffic identically to a steady-state partially-closed road. An earlier
implementation used `int(partial_cap)` which silently quantised 0.6 veh/s
to 0 — the open lane looked closed because it accepted nothing per second.
The credit accumulator fixes that: 0.6 + 0.6 = 1.2 → one vehicle accepted
on the second tick, leaving 0.2 credit; average rate = 0.6 veh/s.

### Bottleneck Penalty (2-lane partial closure)

When a 2-lane edge has 1 lane closed (`effective_lanes < original_lanes`
but `> 0`), the engine applies an explicit per-second acceptance cap on
that edge's first cell:

```
partial_cap_per_s = original_lanes × per_lane_acceptance_per_s × partial_throughput_factor
                  = 2 × 0.6 × 0.5 = 0.6 veh/s   (default)
```

This is the spec's "fixed throughput rate of 50 % of the original
capacity." The `partial_throughput_factor` knob lives at
`closure.partial_throughput_factor` in `simulation/simulation.yaml`.
Combined with the diversion curve, ~50 % of arrivals at the diverge node
are pushed to the escape route once Main Way saturates, while the
remaining ~50 % trickle through the half-cap Main Way at v_free.

### Dynamic diversion (Pressure-Valve, three-stage curve)

The diversion share at each diverge node is a piecewise function of the
**Main Way's whole-edge occupancy ratio** `ρ`:

```
Stage 1  (Normal)        ρ ≤ 0.80           share = 0.07         (5–10 % "local exiters")
Stage 2  (Queue forming) 0.80 < ρ < 1.00    share = linear interp 0.07 → 0.40
Stage 3  (High pressure) ρ ≥ 1.00           share = 0.40         (cap)
```

Default knobs (`config/simulation.yaml`):
- `diversion.baseline_share = 0.07`
- `diversion.saturation_threshold = 0.80`
- `diversion.max_divert_share = 0.40`

This single curve drives **all** escape diversion across the four networks:
RNB/RSB escape ramps, the Naxxar Road Pressure-Valve on VEB. Motorcycles
always pick the Main Way (queue-filtering — they don't experience the jam).

**Why no forced-reroute around closed edges**: when the Main Way is closed
(`effective_lanes = 0`), routing still picks Main vs Escape per the curve
above. Particles that pick the closed Main Way park on the trunk's last cell
and back-pressure ripples upstream. As Main Way occupancy stays at 100 %,
the diversion share is pinned at 40 %, so a continuous **40 % stream of
arrivals peels off to the escape** — exactly the spec's "red pile-up at end
of Main Way + steady stream of orange/green dots through the escape."

This produces the right visual: Main Way red and saturated; Escape running
green/orange continuously; trunk and inflows orange/red as backpressure
builds.

### The 30-scenario grid (Q6 — restricted to filmed slots)

```
4 directions × 3 time slots × variants
    regional: 3 variants  (all_open, 1_lane_closed, 2_lanes_closed)
    valley:   2 variants  (all_open, 1_lane_closed)
= 3 × (3 + 3 + 2 + 2) = 30 scenarios
```

Time slots are strictly the three we filmed in the field: **10am / 1pm /
5pm**. No interpolation, no synthetic slots — the dataset reflects only
what we collected. Each scenario writes three files:

```
data/synthetic/scenarios/<direction>_<slot>_<variant>.csv             # telemetry  (sim_second × edge)
data/synthetic/scenarios/<direction>_<slot>_<variant>.particles.csv   # particle positions  (sim_second × particle)
data/synthetic/scenarios/<direction>_<slot>_<variant>.json            # summary
```

Telemetry schema (one row per `sim_second × edge`):

```
time_s, edge, occupancy_ratio, occupied_m, max_occupancy_m,
speed_mps, speed_kph, cars_on_edge, hgv_on_edge, moto_on_edge,
cumulative_delay, effective_lanes
```

Particles schema (one row per `sim_second × particle`):

```
time_s, edge, cls, count, pos_m, pos_frac, cell_idx
```

`pos_frac` is `pos_m / edge_length` (the *global* position along the edge,
combining cell index + intra-cell position) — used by the visualiser to
interpolate dot positions smoothly across cell boundaries.

`cumulative_delay` is the Phase 4 ML target.

```bash
# Pre-compute the whole 30-scenario library
python simulation/scenarios.py --all
```

### Visualisation (Q8 — abstract layout, currently)

`simulation/visualizer.py` opens a Matplotlib window with a **hardcoded
layout** (one per network — defined in `simulation/networks.py:NODE_LAYOUTS`)
so each direction has a stable, topology-faithful node arrangement that
will line up cleanly when real-world top-down images are added in Phase 3.
Every edge and every particle is rendered concurrently in each 1-second
frame, so split points (diverges) and merges are visible in real time.

**Edge naming for academic clarity**: every edge carries a `display_name`
attribute used by the visualizer label. Trunks are labelled "Regional Road"
or "Valley Road"; inflows are "Entry 1", "Entry 2"… numbered in flow order;
outflows are "Exit 1", "Exit 2"… same convention. Internal IDs (`mainway_N`,
`flyover_S`, etc.) remain unchanged in the engine and telemetry CSVs so
existing downstream code stays compatible.

**Telemetry banner** at the top of the figure shows scenario name (e.g.
`RSB • 5:00 PM • 1 Lane Closed`), current sim time (`t = 300 s / 600 s`),
and a vehicle-weighted **network-mean speed** (km/h) recomputed every frame.

**Line-style rules** per spec:

| Edge state | Line style |
| --- | --- |
| Open / partially closed (any lanes ≥ 1) | Solid |
| Fully closed (effective_lanes = 0) | Dashed black |

Partially-closed edges also display a `½cap` tag next to their label so
you can see at a glance which edges are the bottleneck.

**Sectioned legend** on the right side: OCCUPANCY (colour scale) / LANE
STATE (line styles) / VEHICLE CLASS (dot colours) / NOTES.

| Range          | Edge colour |
| -------------- | ----------- |
| `< 40 %`       | green       |
| `40 – 80 %`    | orange      |
| `> 80 %`       | red         |
| closed         | black, dashed |

Particle dots are read directly from the `*.particles.csv` log, so a
particle at `pos_frac = 0.62` shows up 62 % of the way along its edge —
no jumps, no vanishing. Vehicle classes are colour-coded:

| Class      | Dot colour |
| ---------- | ---------- |
| Car        | dark blue  |
| HGV        | dark red   |
| Motorcycle | navy       |

Closing the window returns to the persistent CLI menu.

A real-world top-down image overlay is planned for Phase 3 once the
abstract layout is validated.

### CLI drill-down (Options 1 and 3)

```
Level 1: Direction       (Regional N/S, Valley E/W)
Level 2: Time slot       (10:00 AM / 1:00 PM / 5:00 PM — the filmed slots)
Level 3: Base horizon H  (600 s / 900 s / 1200 s)
Level 4: Mode
            (A) Closure preview only  — runs to H
            (B) Closure + Recovery    — runs to 2 × H
Level 5: Variant / Transition
            Mode A → all_open / 1_lane_closed / 2_lanes_closed   (regional)
                     all_open / 1_lane_closed                    (valley)
            Mode B (Regional) → 2closed_to_1closed (2 lanes closed → 1 lane closed)
                                2closed_to_0closed (2 lanes closed → fully reopened)
                                1closed_to_0closed (1 lane closed  → fully reopened)
            Mode B (Valley)   → 1closed_to_0closed (1-lane road fully reopens)
```

Cache layout uses **duration-specific subfolders**:

```
data/synthetic/scenarios/600s/<scenario>.{csv, json, particles.csv}     # Mode A, H=600    (30 scenarios)
data/synthetic/scenarios/900s/<scenario>.{csv, …}                       # Mode A, H=900    (30 scenarios)
data/synthetic/scenarios/1200s/<scenario>.{csv, …}                      # Mode A, H=1200 + Mode B, H=600  (54 scenarios)
data/synthetic/scenarios/1800s/<scenario>.{csv, …}                      # Mode B, H=900    (24 scenarios)
data/synthetic/scenarios/2400s/<scenario>.{csv, …}                      # Mode B, H=1200   (24 scenarios)
```

**Total: 162 scenarios** — the full grid covers every combination the CLI
drill-down can reach (`4 directions × 3 slots × 3 horizons × variants`).
Run `python simulation/scenarios.py --all` to populate the cache. The batch
runner also rebuilds the ML feature CSV at the end (see below).

### ML feature snapshots (Phase 4 input)

After every batch pre-compute, the engine extracts one **feature snapshot
per scenario** at `T = base_horizon_s` — the moment of reopening for
Mode B, the end of the closure phase for Mode A. All snapshots aggregate
into a single CSV:

```
data/ml_training/recovery_stats.csv
```

Schema (one row per scenario, 162 rows total):

| column                  | description                                                       |
| ----------------------- | ----------------------------------------------------------------- |
| `road_id`               | direction (regional_north / regional_south / valley_east / valley_west) |
| `network_key`           | RNB / RSB / VEB / VWB                                             |
| `slot`                  | 10am / 1pm / 5pm                                                  |
| `base_horizon_s`        | 600 / 900 / 1200 (H)                                              |
| `horizon_s`             | actual run length (H for Mode A, 2H for Mode B)                   |
| `mode`                  | closure_only / recovery                                           |
| `transition_type`       | e.g. `1_lane_closed`, `2closed_to_1closed`                        |
| `snapshot_time_s`       | T = base_horizon_s                                                |
| `total_occupied_meters` | Σ occupied_m across all edges at T (5m × cars + 8m × HGVs)        |
| `queue_length_m`        | contiguous "red" cell run-length backwards from the trunk's diverge end |
| `current_inflow_rate`   | vehicles/s being injected at T                                    |
| `hgv_percentage`        | HGV share of (cars + HGV) on the network at T                     |
| `recovery_threshold_kph` | network's V_avg target (60 Regional / 45 Valley)                 |
| `recovery_triggered_s`  | T value used by the engine (None for Mode A)                      |
| `time_to_recovery`      | **ML label** — seconds from T until rolling V_avg ≥ threshold; null if not recovered within horizon |
| `scenario_csv`          | provenance path back to the source telemetry CSV                  |

Run on-demand (without re-simulating) via:

```bash
python simulation/build_features.py
```

Or interactively from the CLI menu — **Option 4: Rebuild ML feature snapshots**.

When "View pre-computed" doesn't find a scenario, the warning prints the
exact expected path so you can see whether the bug is the missing run or a
wrong horizon selection.

### Phase-aware GUI HUD

When the visualizer renders a scenario whose summary JSON includes
`recovery_triggered_s`, the figure adds:

- **Phase Indicator** (top-left, banner): `STATUS: ACCIDENT PHASE` (red)
  while `t < base_horizon_s`, switching to `STATUS: RECOVERY PHASE` (teal)
  the moment the recovery trigger fires.
- **Time Since Reopening** (top-centre): `reopened: + N s` counter that
  starts at 0 when the recovery trigger fires and increments each second.
- **Transition Flash**: a yellow-bordered banner appears on the main map
  for ~30 seconds after the trigger, displaying the human-readable
  transition (e.g. *Recovery: 2 lanes closed → 1 lane reopens*). Fades to
  zero opacity over the 30 s window.
- **Recovery Progress Bar** (bottom strip): horizontal bar comparing
  current network V_avg to the network's recovery threshold (60 km/h
  Regional / 45 km/h Valley). Bar colour: red < 85 % of threshold, orange
  approaching, green at/above. A dashed vertical marker shows the target
  on the bar's axis.

Closing a Matplotlib window returns control to the persistent CLI menu —
no Python process exit.

Selecting a not-yet-cached scenario prints:

> _Scenario data not found. Please run Option 2 to pre-compute or select Option 1 for a live run._

---

## Subsequent phases

- **Phase 3 — Real-world overlays.** Replace the abstract NetworkX layout
  with georeferenced top-down images of the road segments.
- **Phase 4 — ML.** Train recovery-time models on the 120-scenario
  telemetry. `cumulative_delay` is the supervised target; features are
  derived from the per-second occupancy / speed timeseries.
