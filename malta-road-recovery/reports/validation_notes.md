# Validation Notes — Malta Road Recovery Engine

Companion document to the FYP report. Records the hard-coded mathematical
constants and behavioural rules used by the simulator, with the precise
file paths and line ranges at which each rule is implemented. The audit
items required by the master code-freeze prompt (merge friction, partial
bottleneck cap, escape divergence curve, censored extrapolation) are each
documented here with their formula, defaults, and provenance.

---

## 1. Greenshields Limits per Road

The engine uses the linear **Greenshields** fundamental diagram for the
class-aware speed-density relation. Per-edge free-flow speed (V_free) is
set from the posted limit; jam density (K_jam) is implicit in the
spatial-occupancy formulation.

| Network | V_free (m/s) | V_free (km/h) | K_jam (5-m car eq.) |
| ------- | ------------ | ------------- | ------------------- |
| Regional Road (RNB, RSB) | 22.22 | **80** | length × lanes / 5 m |
| Valley Road (VEB main, VWB) | 16.67 | **60** | length × lanes / 5 m |
| Naxxar Road slip (VEB escape) | 13.89 | **50** | 1 × 400 / 5 m = 80 cars |

**Formula** (cell-level, in `simulation/engine.py:_cell_speed`):

```
ratio   = occupied_m / (cell_length_m × effective_lanes)
v_cars  = V_free × (1 − ratio / jam_ratio_cap)
v_motos = V_free                              # queue-filtering
v       = max(v_…, min_speed_mps)             # 0.5 m/s creep floor
```

Spatial occupancy uses class-specific lengths: **car = 5 m**, **HGV = 8 m**,
**motorcycle = 0 m** (queue-filtering — bikes pass through congestion).

**Source:** `simulation/engine.py` (vehicle lengths in `_DEFAULT_CFG`,
Greenshields in `_greenshields_speed`, free-flow constants in
`simulation/networks.py`).

---

## 2. Stochastic Merge Friction (20–30 % over first 30 m)

Drivers slow at every junction as they negotiate the change of geometry.
The engine applies a multiplicative speed penalty to the first
`merge_zone_m` metres of any edge whose source node is a **merge** (≥ 2
inbound edges) **or** a **diverge** (≥ 2 outbound edges). The multiplier
is drawn once per cell from a uniform distribution and reused for that
scenario (deterministic per seed, stochastic across seeds).

**Defaults** (`config/simulation.yaml :: merge_friction`):

```
merge_zone_m:      30.0
speed_factor_min:  0.70   # → 30 % maximum reduction
speed_factor_max:  0.80   # → 20 % minimum reduction
```

**Per-junction coverage (verified):**

| Network | Edges that receive friction |
| ------- | --------------------------- |
| RNB | `mergeway_N_b` (trunk after merge), `mainway_N` & `escape_N` (after diverge) |
| RSB | `mergeway_S` (trunk after merge), `mainway_S` & `escape_S` (after diverge) |
| VEB | `valley_cont_E` & `naxxar_E` (after Naxxar diverge) |
| VWB | `fleur_b_W` (after Psaila/Fleur merge), `valley_W_post` (after Valley/Fleur merge) |

**Source:** `simulation/engine.py:_init_state` (cell identification),
`_cell_speed` (penalty application).

---

## 3. Edge-Closure Constraints

### 3.1 Visual rules (lane state → line style)

| Lane state | Line style | Where enforced |
| ---------- | ---------- | -------------- |
| **Open** (all lanes active) | Solid | `simulation/visualizer.py:frame()` |
| **Partial closure** (2-lane edge, 1 closed) | Solid, plus `½cap` text tag | same — `partial = (eff_lanes < orig_lanes) and (eff_lanes > 0)` |
| **Full closure** (single-lane edge closed, OR all lanes of a multi-lane edge closed) | Dashed/dotted (`linestyle=(0,(4,4))`) | same — `fully_closed = (eff_lanes == 0)` |

### 3.2 Bottleneck-capacity penalty (50 % fixed cap)

When a multi-lane edge has at least one lane closed but is not fully
closed, the engine caps the per-second acceptance rate at its first cell
at **50 % of the original free-flow throughput**:

```
partial_cap = original_lanes × per_lane_acceptance_per_s × partial_throughput_factor
            = 2          × 0.6                          × 0.5
            = 0.6 vehicles / second
```

Implemented via an **acceptance-credit accumulator** so the cap works for
fractional rates without `int()` quantisation pinning throughput at zero:

```
each second:  credit ← min(credit + partial_cap, credit_buffer_s × partial_cap)
each veh:     if credit ≥ 1, accept + credit ← credit − 1
```

The buffer (default 3 s) prevents unlimited credit accumulation when the
lane is empty, so the same lane behaves identically whether it has always
been partially closed or has just been reopened to that state mid-run.

**Defaults** (`config/simulation.yaml :: closure`):

```
partial_throughput_factor:  0.5
credit_buffer_s:            3.0
per_lane_acceptance_per_s:  0.6
```

**Source:** `simulation/engine.py:_resolve_edge_arrivals` and the
`_DEFAULT_CFG[closure]` block.

---

## 4. Dynamic Escape-Way Diversion Curve (5–10 % base → 40 % cap)

At every diverge node where an Escape Way exists, traffic chooses Main vs.
Escape stochastically. The fraction picking the Escape Way is:

```
ratio = main_way_whole-edge_occupancy            # 0..1+

if ratio ≤ saturation_threshold (0.80):
    share = baseline_share           = 0.07     # 5–10 % local exiters
elif ratio ≥ 1.0:
    share = max_divert_share         = 0.40     # cap under heavy queue
else:
    share = baseline + (max − baseline) × (ratio − 0.80) / 0.20
            # linear ramp from 0.07 → 0.40 as backpressure fills the trunk
```

Three stages, as specified:

| Stage | Trigger | Share |
| ----- | ------- | ----- |
| **Normal** | Main Way occupancy ≤ 80 % | 7 % (within the 5–10 % band) |
| **Queue Forming** | 80 % < occupancy < 100 % | linear interp 7 % → 40 % |
| **High Pressure** | occupancy ≥ 100 % | 40 % (capped) |

Motorcycles bypass diversion entirely (they pick Main Way; they don't
experience the queue).

**Defaults** (`config/simulation.yaml :: diversion`):

```
baseline_share:        0.07
saturation_threshold:  0.80
max_divert_share:      0.40
```

**Source:** `simulation/engine.py:_choose_next_edge`.

---

## 5. Censored-Recovery Linear Extrapolation (capped 3600 s)

If a recovery scenario reaches its horizon without the rolling V_avg
crossing the threshold (60 km/h Regional, 45 km/h Valley), the engine
fits a slope to the **final 120 s** of the rolling-speed curve and
projects how long it would take to cross — capped safely at 3600 s.

```
tail        = speed_history[-120:]             # final-120s rolling V_avg
slope (m)   = least-squares fit of tail vs. time
current_v   = tail[-1]
gap         = threshold − current_v

if gap ≤ 0:                             # already at/above threshold
    recovery_time_s = horizon − recovery_trigger_s
elif slope > 1e-6:                      # still improving
    recovery_time_s = min(
        (horizon − recovery_trigger_s) + gap / slope,
        3600                            # extrapolation_cap_s
    )
else:                                    # flat or worsening
    recovery_time_s = 3600              # cap
```

Outputs are flagged in the engine summary:

| Field | Meaning |
| ----- | ------- |
| `recovery_observed`     | `True` if the threshold was crossed before horizon |
| `recovery_extrapolated` | `True` if the value was projected (not observed) |
| `recovery_time_s`       | numeric (≤ 3600); `None` only for closure-only previews |

**Defaults** (`config/simulation.yaml :: recovery`):

```
rolling_window_s:         30
regional_threshold_kph:   60.0
valley_threshold_kph:     45.0
extrapolation_window_s:  120
extrapolation_cap_s:    3600
```

**Source:** `simulation/engine.py:simulate()` (slope fit + cap + summary
fields).

---

## 6. Dataset Validation Summary

The 5× multi-seed Mode-B factory batch produces 810 training rows
(`data/ml_training/master_traffic_features.csv`). Of those:

- **162 scenarios × 5 seeds** — each scenario simulated independently with
  a distinct random seed, varying spawn second + Car/HGV order + the
  merge-friction draw.
- **450 closure-only** rows (`recovery_time_seconds = null` — no reopening
  event; correctly excluded from training).
- **360 recovery** rows, all carrying a numeric label in [30, 3600] s
  (observed or extrapolated per Section 5).
- **Mass conservation** verified to 0 vehicles lost across every scenario
  (audit run; injected = exit + escape + queued + in-network at horizon).

Trained model (`models/recovery_rf_model.pkl`):

- **Algorithm:** `RandomForestRegressor` with `OneHotEncoder` for
  categorical descriptors, raw numerics passed through.
- **Grid-search test scores:** R² = 0.979, MAE = 127 s.
- **Feature importances** (in order): `backpressure_segment_count` 0.72,
  `spatial_occupancy_ratio` 0.27, all others < 0.01 — i.e. the size of the
  jam at the moment of reopening is the dominant predictor.

---

## 7. Reproducibility

Every constant in this document lives in `config/simulation.yaml` and can
be overridden at run time. Random seeds are explicit at the
`simulate(scenario, rng_seed=…)` boundary; Mode B fixes them
deterministically as `1000 + run_id × 7919`. Running the same batch on the
same machine twice produces identical CSVs.
