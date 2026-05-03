# IAPT-ARI2201 — Malta Road Recovery

Main repository for my IAPT Project.

## Phase 1 — Manual Ground-Truth Extraction

**Phase 1 is now conducted via Manual Ground-Truth Extraction. Automated YOLO
processing has been archived.**

The original plan was an automated computer-vision pipeline (YOLOv8 +
ByteTrack) that read 10-minute road clips and produced per-minute flow and
speed bins. That pipeline was scrapped after persistent perspective and
tracking inconsistencies corrupted the downstream SUMO calibration. All CV
code and supporting calibration files have been moved to the top-level
`Archive/` folder; they are **not** part of the active pipeline and are not
imported anywhere in `malta-road-recovery/`.

Phase 1 now runs on manual ground-truth extraction: a human watches each
10-minute clip per segment-and-direction, fills one CSV, and
`data_processing/manual_to_sumo.py` drives the rest of the pipeline.

### Directions

The project covers two segments, each with two directions:

- **Regional Road** — `regional_north`, `regional_south`
- **Valley Road** — `valley_east`, `valley_west`

### Manual CSV schema

One row per 1-minute bin (already aggregated by the observer):

| column               | type  | description                                                         |
| -------------------- | ----- | ------------------------------------------------------------------- |
| timestamp_bin        | int   | 1-minute bin index (0, 1, 2, …).                                    |
| car_count            | int   | Cars crossing the counting line in this bin.                        |
| heavy_vehicle_count  | int   | Buses + trucks combined in this bin.                                |
| avg_speed_kph        | float | Mean speed observed for the bin (km/h).                             |
| incident_status      | str   | One of: `clear`, `slow`, `blocked`, `incident`.                     |

Files live at `malta-road-recovery/data/raw/manual_counts/<direction>/<slot>.csv`.
See the README in that folder for full details.

### Prerequisites

1. Install Python dependencies:

```bash
cd /Users/kurtabela/IAPT-ARI2201
pip install -r requirements.txt
```

2. Prepare manual count CSVs in `malta-road-recovery/data/raw/manual_counts/<direction>/`.
   Each file must follow the schema described in `malta-road-recovery/data/raw/manual_counts/README.md`.

3. Verify `config/roads.yaml` against OSM before building the network:
   - confirm `osm_way_ids` match the filmed road segment
   - confirm `speed_limit_kph`, `lanes_per_direction`, and `highway_class`
   - confirm the `direction` → `edge_id` mapping corresponds to the camera direction

4. Install SUMO separately if you want to run simulations.
   Set `SUMO_HOME` and add `$SUMO_HOME/tools` to `PYTHONPATH`.

5. If you want to build the road networks from OSM, run:

```bash
cd malta-road-recovery
python simulation/build_network.py
```

### Running the pipeline

```bash
cd malta-road-recovery
python main.py                                                          # every direction / slot
python main.py --input data/raw/manual_counts/valley_east/1pm.csv       # single file
python main.py --no-metrics                                             # skip extract_metrics
```

### Outputs

For each `<direction>/<slot>.csv`, Phase 1 writes:

```
data/processed/aggregated/<direction>/<slot>.csv           # validated + enriched bins
simulation/demand/<direction>_<slot>.rou.xml               # SUMO routes (vType + route)
simulation/demand/<direction>_<slot>.flow.xml              # SUMO flows (per bin × class)
data/processed/calibration/<direction>_<slot>_metrics.csv  # calibration metrics
```

Because validation + enrichment happen up front, the aggregated and metrics
CSVs are 100% trustworthy ground truth — exactly what SUMO needs.

### Calibrating roads with OSM statistics

1. Use `config/roads.yaml` to identify the filmed segment in OpenStreetMap.
   Confirm the listed `osm_way_ids` and the bounding boxes before downloading.

2. Inspect the OSM tags for each way:
   - `lanes` / `lanes:forward` / `lanes:backward`
   - `maxspeed`
   - `highway`
   - `oneway`
   - `junction`

3. Run `python simulation/build_network.py` to download the OSM extract and convert it to SUMO `.net.xml`.
   This uses the road geometry and speed data from OSM, so the simulation network matches the real-world segment.

4. Compare the OSM-derived road geometry to the manual counts:
   - `speed_limit_kph` should match the posted `maxspeed` tag in OSM
   - `lanes_per_direction` should match the number of traffic lanes in the filmed direction
   - the SUMO edge IDs in `config/roads.yaml` should correspond to the direction observed in your manual counts

5. Use the generated calibration metrics in `data/processed/calibration/<direction>_<slot>_metrics.csv` to validate that the manual counts reflect the same road characteristics as OSM.
   - peak hourly flow should be plausible for the OSM road class and lane count
   - mean speed should be consistent with the posted speed limit
   - incident bins should map to low-speed / high-density outputs

If the OSM road tags have changed or if the filmed segment is no longer aligned with the current OSM extract, update `config/roads.yaml` and re-run `simulation/build_network.py`.

## Phase 2 — SUMO Network Build & Visual Simulation

Phase 2 turns OSM Way IDs and the manual CSVs from Phase 1 into a working
SUMO network you can watch traffic move on. **One script does everything:**
[`malta-road-recovery/simulation/run_sumo.py`](malta-road-recovery/simulation/run_sumo.py).

### What you'll do

1. **Paste OSM Way IDs into a config file** — no XML, no manual stitching.
2. **Build the network** — one command runs `netconvert` for you.
3. **Preview the network** — open SUMO-GUI / netedit and check the road looks right.
4. **Simulate traffic** — run a 10-minute clip with vehicles that move and are
   color-coded by class.

### One-time prerequisite — install SUMO

SUMO ships `netconvert`, `sumo`, `sumo-gui`, and `netedit`. After installing,
make sure `netconvert --version` works in your terminal.

- macOS: `brew install --cask sumo` (or download from <https://sumo.dlr.de>)
- Linux: `sudo apt install sumo sumo-tools sumo-doc`
- Windows: installer at <https://sumo.dlr.de/docs/Downloads.php>

### Step 1 — Tell the project which roads to use

Open [`malta-road-recovery/config/roads.yaml`](malta-road-recovery/config/roads.yaml).
At the top there's a **PHASE 2 QUICK-START** comment block explaining the
schema. Each direction declares a `topology` block:

- `topology.type: simple` — one OSM way per direction (Valley Road).
- `topology.type: merge_split` — two inflows merge into a 3-lane trunk that
  splits into two outflows (Triq Regionali, both directions).

The merge_split topology models Triq Regionali realistically:

```
NORTHBOUND
──────────
  [Tunnel inflow,  2 lanes]──┐
                              ├──[Trunk, 3 lanes]──┬──[Flyover,   2 lanes]
  [Side inflow,    1 lane ]──┘                     └──[Underpass, 1 lane ]
                              ↑                    ↑
                        Priority merge       Diverge / split

SOUTHBOUND  (mirror of NB — inflows and outflows swap)
──────────
  [Flyover,   2 lanes]──┐
                         ├──[Trunk, 3 lanes]──┬──[Tunnel,   2 lanes]
  [Underpass, 1 lane ]──┘                     └──[Escape,   1 lane ]
```

You paste the OSM Way ID for each named edge directly into the matching slot:

```yaml
regional_north:
  topology:
    type: merge_split
    inflow_a:    { label: tunnel_n,    way_id: 999000010, lanes: 2 }
    inflow_b:    { label: inflow_n,    way_id: 999000011, lanes: 1 }
    trunk:       { label: trunk_n,     way_id: 29980850,  lanes: 3 }
    outflow_main:{ label: flyover_n,   way_id: 517103377, lanes: 2 }
    outflow_div: { label: underpass_n, way_id: 999000012, lanes: 1 }
    merge_node:  merge_node_N
    split_node:  split_node_N
```

To find Way IDs:

1. Go to <https://www.openstreetmap.org>, locate the road.
2. Click each way segment — the URL becomes `…/way/<ID>`.
3. Paste the `<ID>` into the matching `way_id:` slot.

Placeholder IDs starting with `999` are stand-ins. The build script warns
loudly on every run until you replace them, but the rest of the pipeline
still works (the network geometry comes from YAML, not from OSM, for the
merge_split topology).

### Step 2 — Build the SUMO network

```bash
cd malta-road-recovery

# Both segments
python simulation/run_sumo.py --build

# Just one segment
python simulation/run_sumo.py --build --segment regional_road

# See what it would do without running netconvert
python simulation/run_sumo.py --build --dry-run
```

For `merge_split` segments this generates three input files from `roads.yaml`:

- `<segment>.nod.xml` — junction nodes (priority merges, priority splits)
- `<segment>.edg.xml` — edges with explicit lane counts and labels
- `<segment>.con.xml` — explicit lane-to-lane connections (trunk lanes
  feed `outflow_main`; the rightmost trunk lane feeds `outflow_div`)

`netconvert --lefthand` reads them and produces the compiled `.net.xml`. The
script then post-processes the `.net.xml` to inject XML comments above each
labelled edge so you can see at a glance which edge is the Flyover, Trunk,
Tunnel, etc.

```
simulation/networks/regional_road/regional_road.net.xml
simulation/networks/valley_road/valley_road.net.xml
```

### Step 3 — Preview the empty network (with coloured edges)

Look at the road **without any traffic** to verify the geometry is correct
before running vehicles. The preview command writes a `view_settings.xml`
that paints each labelled edge a different colour:

| Edge label                   | Colour   |
| ---------------------------- | -------- |
| Flyover (`flyover_n/_s`)     | Orange   |
| Trunk (`trunk_n/_s`)         | Green    |
| Tunnel (`tunnel_n/_s`)       | Magenta  |
| Underpass (`underpass_n/_s`) | Cyan     |
| Side inflow (`inflow_n`)     | Cyan     |
| Escape (`escape_s`)          | Red      |

```bash
# Open in SUMO-GUI with the colour scheme applied
python simulation/run_sumo.py --preview --segment regional_road

# Open in netedit instead (editor — useful if you need to tweak nodes/edges)
python simulation/run_sumo.py --preview --segment regional_road --netedit
```

### Step 4 — Run a visual simulation

```bash
# Watch traffic for one direction × time slot
python simulation/run_sumo.py --simulate --direction regional_north --slot 10am

# Headless run (no GUI — useful for CI / batch)
python simulation/run_sumo.py --simulate --direction valley_east --slot 1pm --no-gui
```

This will:

1. Run `manual_to_sumo.py` for the chosen `direction/slot.csv` if the demand
   files don't already exist (so you don't have to remember to run it first).
2. Write a `.sumocfg` linking the network, routes, and flows.
3. Launch `sumo-gui` with vehicles you can see and follow.

**Vehicle colors:**

| Class      | Color  | PCE weight |
| ---------- | ------ | ---------- |
| Car        | Yellow | 1.0        |
| Motorcycle | Blue   | 0.5        |
| Heavy      | Red    | 2.5        |

**Demand split for merge_split topologies.** The manual CSV gives one count
per bin for the full filmed direction. `manual_to_sumo.py` reads
`merge_split_demand` from `roads.yaml` and splits each bin's count across
**four** routes:

| Route          | Edges                                    | Default share |
| -------------- | ---------------------------------------- | ------------- |
| `via_a_main`   | `inflow_a → trunk → outflow_main`        | 0.49 (70×70)  |
| `via_a_div`    | `inflow_a → trunk → outflow_div`         | 0.21 (70×30)  |
| `via_b_main`   | `inflow_b → trunk → outflow_main`        | 0.21 (30×70)  |
| `via_b_div`    | `inflow_b → trunk → outflow_div`         | 0.09 (30×30)  |

Defaults: `inflow_a_share = 0.70` and `outflow_main_share = 0.70` (the wider
road carries more traffic). Tune them under `merge_split_demand:` in
`roads.yaml`. Largest-remainder rounding makes the per-bin counts sum exactly
to the CSV count, so the trunk total is preserved.

Background traffic on extra slip-roads (those you did **not** count) lives
under `merge_ways:` in `roads.yaml` — each entry gets a constant `<flow>`
covering the whole simulation at `background_flow_veh_hr` (≈ 5–10 % of
mainline). Empty by default.

### What the SUMO files mean (for the curious)

| File         | Purpose                                                                                  |
| ------------ | ---------------------------------------------------------------------------------------- |
| `.osm`       | Raw OpenStreetMap export — informational only for `merge_split` segments                 |
| `.nod.xml`   | Junction nodes (priority merges, priority splits) generated from `roads.yaml`            |
| `.edg.xml`   | Edges with explicit lane counts and labels (Flyover, Trunk, …) generated from YAML       |
| `.con.xml`   | Lane-to-lane connections — controls how trunk lanes diverge across the split             |
| `.net.xml`   | Compiled network. `netconvert` reads .nod + .edg + .con and writes this.                 |
|              | Carries inline `<!-- LABEL: ... -->` comments so each edge is identifiable on inspection |
| `.rou.xml`   | Vehicle-type definitions (car/motorcycle/heavy + colors) plus the four route variants    |
| `.flow.xml`  | Per-bin traffic demand split across the four routes by `merge_split_demand` shares       |
| `.sumocfg`   | Master config tying net + routes + flows + run params                                    |
| `view_settings.xml` | Per-edge colour overrides used by the `--preview` mode                            |

### Troubleshooting

- **"netconvert not found"** — install SUMO and verify `netconvert --version`
  in a fresh terminal.
- **"requests not installed"** — `pip install -r requirements.txt`.
- **Network looks wrong / discontinuous** — re-check the order of Way IDs in
  `roads.yaml` (must be in driving order). Then `--build --force-download`.
- **No vehicles appear in `--simulate`** — check that
  `data/raw/manual_counts/<direction>/<slot>.csv` exists and has at least one
  bin with non-zero counts.
- **Way IDs starting with `999`** — those are placeholders; the warning is
  expected until you replace them with real OSM measurements.

## Subsequent phases

- **Phase 3 — TraCI**: incident control + per-step metrics logging
  (consumes the `.net.xml` / `.rou.xml` / `.flow.xml` produced above).
- **Phase 4 — ML**: feature matrices live in `data/features/`.

## Archive

`Archive/` at the repo root contains the scrapped CV pipeline for reference
only — `detect.py`, `virtual_gates.py`, `utils.py`, `config.py`,
`video_config.py`, the historical `yolo_output/` directory, the per-clip
calibration YAMLs, and the CV-specific unit tests.
