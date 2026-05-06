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
| **VWB** | Valley Road Westbound                | valley_W_pre (1L 870m) and (psaila_W + fleur_a_W → fleur_b_W) → junction → valley_W_post (1L 130m) |

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

### CLI drill-down (Option 3)

```
Level 1: Direction      (Regional N/S, Valley E/W)
Level 2: Time slot      (10:00 AM / 1:00 PM / 5:00 PM — strictly the filmed slots)
Level 3: Incident variant
            Regional → 3 options (All Open / 1 Lane / 2 Lanes Closed)
            Valley   → 2 options (All Open / 1 Lane Closed)
```

Selecting a not-yet-cached scenario prints:

> _Scenario data not found. Please run Option 2 to pre-compute or select Option 1 for a live run._

---

## Subsequent phases

- **Phase 3 — Real-world overlays.** Replace the abstract NetworkX layout
  with georeferenced top-down images of the road segments.
- **Phase 4 — ML.** Train recovery-time models on the 120-scenario
  telemetry. `cumulative_delay` is the supervised target; features are
  derived from the per-second occupancy / speed timeseries.
