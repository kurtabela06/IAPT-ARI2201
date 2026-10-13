# Malta Road Recovery

**A custom NetworkX-based traffic digital-twin and Random-Forest
recovery-time predictor for the Triq Regionali and Valley Road corridors
in Malta.**

*IAPT submission, course IAPT-ARI2201. Kurt Abela, University of Malta.*

---

## 1. Project Abstract

Maltese arterial roads — in particular Triq Regionali (Regional Road) and
the Valley Road / Naxxar corridor — suffer disproportionate, persistent
congestion whenever a lane is closed by an incident. There is currently
no lightweight tool that a transport planner can run on a laptop to
**estimate how long the road will take to recover** once the closure is
lifted.

This project delivers an end-to-end pipeline that:

1. **Ingests manual vehicle counts** from filmed 10-minute observation
   clips (per direction × time slot — 10:00 / 13:00 / 17:00).
2. **Simulates** the closure and subsequent recovery on a calibrated
   cell-transmission model of four distinct road sub-networks (Regional
   N/S, Valley E/W), using a custom engine built on `networkx`, `pandas`
   and `matplotlib`.
3. **Generates a 5×-multi-seed synthetic dataset** of 810 scenario
   snapshots taken at the exact moment the road reopens — the "ground
   truth" for ML training.
4. **Trains a `RandomForestRegressor`** (`sklearn`) to predict
   `recovery_time_seconds` from the traffic state at reopening.
5. **Serves the trained model** via a step-by-step interactive
   prediction dashboard.

### Architecture overview

```
┌──────────────────────────────────────────────────────────────────────┐
│           data/raw/manual_counts/<direction>/<slot>.csv              │
│                            (field counts)                            │
└──────────────────┬───────────────────────────────────────────────────┘
                   │   data_processing/manual_to_pulse.py
                   ▼
        data/processed/pulses/<direction>/<slot>.csv     (1-min demand pulses)
                   │   simulation/calibrator.py
                   ▼
        data/processed/calibration/<…>.json              (per-slot scaling)
                   │
                   │   simulation/engine.py  (cell-transmission Greenshields engine)
                   │   simulation/networks.py (4 hardcoded DiGraphs: RNB/RSB/VEB/VWB)
                   │   simulation/scenarios.py (Mode A standard / Mode B 5× factory)
                   ▼
   data/synthetic/scenarios/<H>s/<scenario>.{csv,json,particles.csv}   (playback)
                   │
                   │   simulation/build_features.py
                   ▼
        data/ml_training/master_traffic_features.csv     (810 rows × 10 cols)
                   │   ml/train_model.py
                   ▼
        models/recovery_rf_model.pkl                     (RF + grid search)
                   │   ml/predict.py
                   ▼
   [ML INFERENCE PREDICTION SUITE]
   * Query Parameters: RSB | 17:00 | 20-min Horizon | 2->1 State
   * Predicted Post-Incident Recovery Time: 842 Seconds (14.0 Minutes)
```

### Historical pivot — Why no SUMO?

This project originally targeted SUMO (Simulator of Urban Mobility) as
its physics engine. The pivot to a custom `networkx`-based engine was
driven by:

- **macOS / XQuartz environment friction** — SUMO's GUI binaries depend
  on X11, which is awkward on Apple Silicon. Repeated GUI-render
  failures during early calibration runs made iteration painfully slow.
- **Transparency over abstraction** — for a dissertation it is more
  defensible to write and explain the cell-transmission and merge-
  friction physics than to wrap them in a third-party simulator whose
  behaviour we can only treat as a black box.
- **Zero-overhead install** — the final deliverable runs from
  `pip install -r requirements.txt` and **needs nothing else** (no SUMO
  binaries, no XQuartz, no `SUMO_HOME`).

The pivot is fully documented in the git history.

---

## 2. Installation and Quickstart

### Requirements

- Python 3.10+ (tested on 3.12)
- macOS / Linux / Windows — no platform-specific dependencies

### One-command install

```bash
git clone <repo-url>
cd IAPT-ARI2201
pip install -r requirements.txt
```

`requirements.txt` brings in the entire dependency tree:

| package | role |
| --- | --- |
| `numpy`, `pandas`, `PyYAML` | data handling & YAML config parsing |
| `networkx` | directed graph models for road networks |
| `matplotlib` | live playback animation + heatmap + prediction dashboard |
| `scikit-learn` ≥ 1.3.0 | OneHotEncoder, RandomForestRegressor, GridSearchCV |
| `joblib` | model serialization & parallel jobs |

There are **no external binaries or platform-specific dependencies**.
Unlike SUMO, this runs cleanly on macOS (including Apple Silicon), Linux, and Windows
right out of `pip install -r requirements.txt`.

### Quickstart

```bash
cd malta-road-recovery
python main.py
```

This opens the persistent CLI menu (Section 3).

---

## 3. Step-by-Step User Guide

### The interactive suite

```
=========================================================
      MALTA DIGITAL TWIN - BATCH SIMULATION SUITE
=========================================================
[1] View a Single Scenario (Static Closure Preview)
[2] View a Single Scenario (Closure & Opening Recovery Preview)
    --> sub-menu offers transitions (2->1, 2->0, 1->0) by road
[3] RUN STANDARD BATCH   (1x — visual precomputations only)
[4] RUN ML FACTORY BATCH (5x multi-seed — purges + 5x ML data)
[5] TRAIN ML MODEL       (RandomForest grid-search → .pkl)
[6] PREDICT RECOVERY     (live inference dashboard)
[7] Quit
=========================================================
```

### The recommended first-run order

**Note:** Manual counts and pulses are already included in the repository.
If you want to regenerate them from scratch, run:

```bash
python data_processing/manual_to_pulse.py --all
```

Otherwise, proceed directly to step 2 below:

1. *(Optional)* **Convert manual counts to demand pulses**:
   Reads every `data/raw/manual_counts/<direction>/<slot>.csv` and produces
   1-minute pulse CSVs in `data/processed/pulses/`.

2. *(Required)* **Generate calibration factors**:
   Before running scenarios, the engine requires per-slot injection scaling
   factors. These are computed on-demand by `calibrator.py` during the first
   scenario run and cached in `data/processed/calibration/`. This is automatic.
   The first scenario takes slightly longer (~30s vs. ~3s thereafter).

3. **Generate the 162-scenario playback library** — menu option **[3]**.
   Iterates the full scenario matrix once (seed = 42), writes JSON / CSV /
   particles files into `data/synthetic/scenarios/<H>s/`, and generates a
   162-row `master_traffic_features.csv`. Takes ≈ 5 minutes.

4. **Generate the 810-row multi-seed ML training set** — menu option **[4]**.
   Runs the scenario matrix 5 times with distinct random seeds. Writes the
   full recovery playback only on the last seed; earlier iterations are
   *headless*. Final output: 810 rows in `master_traffic_features.csv`
   (162 scenarios × 5 seeds). Takes ≈ 5 minutes.

5. **Train the RandomForest** — menu option **[5]**. Loads the 810-row
   dataset, runs an 80/20 split (≈ 648 training rows), uses `GridSearchCV`
   over `n_estimators × max_depth × min_samples_split` minimizing MAE on
   the one-hot-encoded features, and serializes the fitted pipeline to
   `models/recovery_rf_model.pkl`. Prints R² / MAE / feature importances.
   Takes ≈ 10–15 seconds.

6. **Live prediction dashboard** — menu option **[6]**. Step-by-step prompts:
   - Select Road & Direction (RNB / RSB / VEB / VWB)
   - Select Time Slot (10:00 / 13:00 / 17:00)
   - Select Incident Duration (600 / 900 / 1200 s or 10 / 15 / 20 min)
   - Select Transition Type (Closure Only, or valid recovery transitions)

   The system extracts baseline traffic conditions from the precomputed
   `master_traffic_features.csv`, feeds them through the trained Random
   Forest, and outputs the predicted recovery time in seconds and minutes.

### Visual scenario playback (options [1] and [2])

Options [1] (static closure) and [2] (closure + opening recovery) drill
through Direction → Slot → Horizon → Variant/Transition and open a live
Matplotlib animation with:

- Heatmap on every edge (Green < 40 % occupancy, Orange 40–80 %, Red > 80 %).
- Line styles: **Solid** = open or partial closure, **Dotted** = full
  closure.
- Phase indicator (`STATUS: ACCIDENT PHASE` / `STATUS: RECOVERY PHASE`).
- Continuous simulation clock and `Time Since Reopening` counter.
- Recovery progress bar at the bottom (V_avg vs. target 60 / 45 km/h).
- Vehicle dots colour-coded by class (cars / HGV / motorcycles) moving
  along each edge at the calculated flow speed.

Closing the Matplotlib window returns control to the menu — no Python
restart needed.

---

## 4. Repository Layout

```
IAPT-ARI2201/
├── README.md                        ← (this file)
├── requirements.txt
└── malta-road-recovery/
    ├── main.py                      ← entry point: launches the CLI suite
    ├── config/
    │   └── simulation.yaml          ← engine parameters (Greenshields,
    │                                  merge friction, closure caps,
    │                                  recovery thresholds, extrapolation)
    ├── data_processing/
    │   └── manual_to_pulse.py       ← field counts → 1-minute pulses
    ├── simulation/
    │   ├── networks.py              ← 4 hardcoded DiGraphs (RNB/RSB/VEB/VWB)
    │   ├── engine.py                ← cell-transmission Greenshields engine
    │   ├── calibrator.py            ← per-slot demand scaling
    │   ├── scenarios.py             ← Mode A / Mode B batch drivers
    │   ├── build_features.py        ← T = base_horizon snapshots → CSV
    │   ├── visualizer.py            ← Matplotlib heatmap + HUD + animation
    │   └── cli.py                   ← persistent 7-option menu
    ├── ml/
    │   ├── train_model.py           ← RandomForest + GridSearchCV + joblib
    │   └── predict.py               ← real-time inference dashboard
    ├── models/
    │   └── recovery_rf_model.pkl    ← trained RF pipeline (≈ 200 KB)
    ├── data/
    │   ├── raw/
    │   │   ├── manual_counts/       ← per-(direction, slot) field CSVs
    │   │   ├── video/               ← source 10-min clips (.mp4)
    │   │   └── osm/                 ← OpenStreetMap extract
    │   ├── processed/
    │   │   ├── pulses/              ← 1-min demand pulses
    │   │   └── calibration/         ← per-slot scaling factors
    │   ├── synthetic/
    │   │   └── scenarios/
    │   │       ├── 600s/   ← Mode A H=600
    │   │       ├── 900s/   ← Mode A H=900
    │   │       ├── 1200s/  ← Mode A H=1200 + Mode B H=600
    │   │       ├── 1800s/  ← Mode B H=900
    │   │       └── 2400s/  ← Mode B H=1200
    │   └── ml_training/
    │       └── master_traffic_features.csv   ← 810-row training set
    └── reports/
        ├── validation_notes.md      ← physics audit + formulas
        ├── report_draft.md          ← dissertation skeleton
        └── figures/{RNB,RSB,VEB,VWB}.png
```

---

## 5. The Four Road Sub-Networks

Each is a `networkx.DiGraph` with hard-coded geometry calibrated against
the filmed corridor.

| Key | Display name | Topology |
| --- | --- | --- |
| **RNB** | Regional Road Northbound | Two inflows (Main + Underpass) → 3-lane trunk → diverge → Final Main (2L 465 m) / Escape (1L 465 m) |
| **RSB** | Regional Road Southbound | Two inflows (Flyover + Underpass) → 3-lane trunk → diverge → Tunnel (2L 175 m) / Escape (1L 175 m) |
| **VEB** | Valley Road Eastbound | Single inflow → branch → Valley Continuation (1L 350 m) / Naxxar Road escape (1L 400 m) |
| **VWB** | Valley Road Westbound | Three inflows (Valley + Psaila + Fleur-de-lys) → merge → 1-lane exit |

See `reports/figures/{RNB,RSB,VEB,VWB}.png` for the abstract layouts used
by the visualiser.

---

## 6. The 162-Scenario Grid

```
4 directions × 3 time slots × 3 horizons × variants =  162 scenarios
                                                   ×  5 seeds (Mode B)
                                                  = 810 ML training rows
```

Variants per road type:

| Road type | Closure-only variants | Recovery transitions |
| --- | --- | --- |
| Regional | `all_open`, `1_lane_closed`, `2_lanes_closed` | `2closed_to_1closed`, `2closed_to_0closed`, `1closed_to_0closed` |
| Valley | `all_open`, `1_lane_closed` | `1closed_to_0closed` |

Cache layout uses **duration-specific subfolders** (file isolation —
playback JSON lives here, never mixed with the ML CSV):

```
data/synthetic/scenarios/600s/    Mode A, H=600    (30 scenarios)
data/synthetic/scenarios/900s/    Mode A, H=900    (30 scenarios)
data/synthetic/scenarios/1200s/   Mode A, H=1200 + Mode B, H=600  (54)
data/synthetic/scenarios/1800s/   Mode B, H=900    (24 scenarios)
data/synthetic/scenarios/2400s/   Mode B, H=1200   (24 scenarios)
```

---

## 7. ML Training Set Contract

`data/ml_training/master_traffic_features.csv` — one row per
(scenario × seed), schema:

| column | type | description |
| --- | --- | --- |
| `run_id` | int | seed index (1 for Mode A; 1–5 for Mode B) |
| `road_id` | string | direction (regional_north / …) |
| `time_slot` | string | 10am / 1pm / 5pm |
| `incident_duration` | int | closure length in seconds (600 / 900 / 1200) |
| `transition_type` | string | e.g. `2closed_to_1closed` |
| `spatial_occupancy_ratio` | float | network `occupied_m / capacity_m` at T |
| `hgv_ratio` | float | HGV ÷ (cars + HGV) on the network at T |
| `backpressure_segment_count` | int | # of cells in the "red" (>80 %) band at T |
| `escape_flow_rate` | float | veh/s exiting via escape edges at T |
| `recovery_time_seconds` | float | **target** — observed OR final-120 s slope extrapolation, capped 3600 s. `null` only for closure-only rows. |

Snapshot time T = base_horizon (the moment of reopening for recovery
scenarios; the end of closure for static-closure rows).

---

## 8. Physics & ML Validation

For the formal write-up of the audit items required by the dissertation —
Greenshields per-network limits, the 20–30 % stochastic merge friction,
the 50 % partial-closure bottleneck cap, the dynamic 5 %–40 % escape
divergence curve, and the linear-extrapolation rule for censored recovery
times — check the docstrings in `simulation/engine.py` and
`simulation/calibrator.py`, and the inline comments in `config/simulation.yaml`.

Headline model performance on the 360 recovery-labelled rows:

| Metric | Value |
| --- | --- |
| Test R² | **0.979** |
| Test MAE | **127 seconds** |
| Best hyper-parameters | `n_estimators=100, max_depth=None, min_samples_split=10` |

Top feature importances:

| feature | importance |
| --- | --- |
| `backpressure_segment_count` | 0.72 |
| `spatial_occupancy_ratio` | 0.27 |
| (all others) | < 0.01 |

---

## 9. Pre-Submission Checklist

Before submitting, verify:

- [ ] **Dependencies installed**: `pip install -r requirements.txt` runs cleanly.
- [ ] **CLI menu works**: `python main.py` launches without errors.
- [ ] **First scenario runs**: Select [1], drill to RSB / 5pm / 600s / all_open.
      Should see live animation with no crashes (calibration caches on first run).
- [ ] **Synthetic data generation**: Run [3] (≈5 min) and [4] (≈5 min) to populate
      `data/ml_training/master_traffic_features.csv`.
- [ ] **Model training works**: Run [5] to train the RandomForest (≈10–15 sec).
      Verify `models/recovery_rf_model.pkl` is created and R² ≈ 0.98.
- [ ] **Inference dashboard works**: Run [6] and make a prediction for any road/slot.
      Should output recovery time without errors.
- [ ] **Archive is clean**: The `Archive/` folder contains legacy code (YOLO pipeline)
      and is not part of the active submission.
- [ ] **README is accurate**: All paths, dependency versions, and CLI descriptions match.
- [ ] **Git history is clean**: Commit message describes the project clearly.
      *(Optional but recommended for dissertations.)*

The size of the jam at the moment of reopening dominates the prediction
— physically faithful.

---

## 9. Reproducibility

Every numeric constant the engine relies on lives in
`config/simulation.yaml`. Random seeds are explicit:

- **Mode A** uses the fixed default `rng_seed = 42`.
- **Mode B** seeds run `i ∈ {1..5}` deterministically as `1000 + i × 7919`.

Re-running the same batch on the same machine produces identical
`master_traffic_features.csv` files.

---

## 10. Citation and Licence

If you use this work academically:

> Abela, K. (2026). *Malta Road Recovery: a custom NetworkX digital-twin
> simulator and Random Forest predictor for post-incident traffic
> recovery on Maltese arterials.* University of Malta — FYP IAPT-ARI2201.

License: see `LICENSE` (FYP work; institutional terms apply).
