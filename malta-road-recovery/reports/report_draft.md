<!--
============================================================================
FYP REPORT — SKELETON / DRAFT
Malta Road Recovery: A NetworkX Digital-Twin Simulator and Machine-Learning
Predictor for Post-Incident Traffic Recovery on Maltese Arterial Roads.

This is a structural skeleton. Each section lists what to write and points
at the concrete project artifacts that supply the content. Replace the
[bracketed prompts] with prose. Word-count guidance is indicative.
============================================================================
-->

# 1. Title Page

> *(Front matter — typically a fixed institutional template. Fill in.)*

- **Project title:** Malta Road Recovery — A Custom Digital-Twin Traffic
  Simulator and Random-Forest Predictor for Post-Incident Recovery Time
- **Author:** Kurt Abela
- **Supervisor:** [name]
- **Course / Unit:** IAPT — ARI2201
- **Institution / Department:** University of Malta, [Faculty/Dept]
- **Submission date:** [month year]
- **Degree for which submitted:** [B.Sc. / …]
- *(Plagiarism / authenticity declaration as required by the institution.)*

---

# 2. Abstract

> *(~250–300 words, single paragraph or two. Written LAST.)*

[State the problem: Maltese arterial roads (Triq Regionali / Valley Road)
suffer disproportionate congestion after lane-closure incidents, and there
is no lightweight tool to estimate how long recovery takes. State the
approach: a custom cell-transmission traffic engine built on NetworkX
(replacing an abandoned SUMO pipeline), calibrated from manually-collected
ground-truth counts, used to generate a synthetic scenario library, from
which a Random-Forest regressor predicts recovery time. State the headline
results: R² ≈ 0.98 / MAE ≈ 127 s on held-out scenarios; the dominant
predictors are backpressure-segment-count and spatial-occupancy-ratio.
State the contribution: an end-to-end, dependency-light digital twin +
predictive dashboard runnable on a laptop.]

**Keywords:** traffic simulation, cell-transmission model, Greenshields,
NetworkX, incident recovery, Random Forest, digital twin, Malta.

---

# 3. Acknowledgements

> *(Half a page. Personal.)*

[Supervisor; anyone who helped with the manual field counts; family;
open-source maintainers (NetworkX, scikit-learn, pandas, Matplotlib).]

---

# 4. Table of Contents

> *(Auto-generated at typeset time. Placeholder below.)*

```
1  Introduction ............................................. p
2  Background and Literature Review ......................... p
3  Specification and Design ................................. p
4  Implementation ........................................... p
5  Evaluation and Testing ................................... p
6  Future Work .............................................. p
7  Conclusions .............................................. p
   Appendices ............................................... p
   Bibliography and References .............................. p
```

---

# 5. Lists

## 5.1 List of Figures

> *(Auto-generated. Seed entries — the four network layouts already exist
> as `reports/figures/{RNB,RSB,VEB,VWB}.png`.)*

- Fig. 2.1 — The fundamental diagram (Greenshields speed–density relation).
- Fig. 3.1 — RNB network skeleton (`reports/figures/RNB.png`).
- Fig. 3.2 — RSB network skeleton (`reports/figures/RSB.png`).
- Fig. 3.3 — VEB network skeleton (`reports/figures/VEB.png`).
- Fig. 3.4 — VWB network skeleton (`reports/figures/VWB.png`).
- Fig. 4.1 — Cell-transmission decomposition of an edge.
- Fig. 4.2 — Visualizer HUD during the RECOVERY PHASE (heatmap + progress bar).
- Fig. 5.1 — Occupancy ripple (backpressure propagation) over time.
- Fig. 5.2 — Random-Forest feature-importance bar chart.
- Fig. 5.3 — Predicted vs. actual recovery time (test set).

## 5.2 List of Tables

- Table 3.1 — Per-network edge geometry (length, lanes, capacity).
- Table 3.2 — Recovery transitions per road type.
- Table 4.1 — Engine configuration parameters (`config/simulation.yaml`).
- Table 4.2 — ML feature contract (`master_traffic_features.csv` schema).
- Table 5.1 — Scenario-grid composition (162 scenarios; 810 multi-seed rows).
- Table 5.2 — Model performance (R², MAE, best hyper-parameters).

## 5.3 List of Abbreviations

| Abbrev. | Expansion |
| --- | --- |
| RNB / RSB | Regional Road Northbound / Southbound |
| VEB / VWB | Valley Road Eastbound / Westbound |
| CTM | Cell-Transmission Model |
| HGV | Heavy Goods Vehicle |
| PCE | Passenger-Car Equivalent |
| RF | Random Forest |
| MAE | Mean Absolute Error |
| OSM | OpenStreetMap |
| GUI | Graphical User Interface |
| CLI | Command-Line Interface |
| V_avg | Network vehicle-weighted mean speed |
| H | Base incident horizon (600 / 900 / 1200 s) |

---

# 6. Main Content

## 6.a Introduction

> *(~3–5 pages.)*

### 6.a.1 Problem context and motivation
[Malta's road density and dependence on a few arterials; the outsized
impact of lane-closure incidents; absence of a cheap recovery-estimation
tool.]

### 6.a.2 Aim and objectives
- **Aim:** predict post-incident recovery time on Maltese arterials from
  the traffic state at the moment a lane reopens.
- **Objectives:** (i) build a calibrated micro/macro traffic engine for
  four real road sub-networks; (ii) generate a synthetic, multi-seed
  scenario library as ground truth; (iii) train and validate a recovery-time
  predictor; (iv) deliver an interactive simulator + prediction dashboard.

### 6.a.3 Scope and limitations
[Two real corridors, three filmed time slots (10am/1pm/5pm), three
incident durations; macroscopic cell model rather than per-vehicle
microsimulation; manually-collected counts as the calibration source.]

### 6.a.4 The methodological pivot (SUMO → custom engine)
[Briefly: the original SUMO/XQuartz plan was abandoned for a self-contained
NetworkX engine — justify the trade-off (portability, transparency,
control over merge/diverge physics) and note this is documented in the
project README and git history.]

### 6.a.5 Report structure
[One sentence per chapter.]

## 6.b Background and Literature Review

> *(~6–10 pages. This is the citation-heavy chapter.)*

### 6.b.1 Traffic-flow theory
- Fundamental diagram; **Greenshields** linear speed–density model (the
  engine's core: `v = v_free·(1 − ρ/ρ_jam)`).
- **Cell-Transmission Model (Daganzo)** — the discretisation the engine uses.
- Spatial occupancy vs. vehicle-count density; PCE weighting.
- Queue/backpressure propagation and shockwave theory.

### 6.b.2 Incident impact and recovery
[Capacity-drop literature; recovery-time / resilience metrics; "time to
return to free-flow" definitions — relate to the project's V_avg threshold
(60 km/h Regional, 45 km/h Valley).]

### 6.b.3 Microsimulation tooling
[SUMO and why it was ultimately not used here; lightweight graph-based
alternatives; NetworkX as a modelling substrate.]

### 6.b.4 Machine learning for traffic prediction
[Tree ensembles vs. deep models for tabular traffic features; Random Forest
regression; handling censored targets; one-hot vs. embedding for
categorical road descriptors.]

### 6.b.5 Gap analysis
[What existing work does *not* offer: a laptop-runnable, transparent,
incident-recovery digital twin tailored to specific Maltese geometry.]

## 6.c Specification and Design

> *(~5–8 pages.)*

### 6.c.1 Requirements
- Functional: ingest manual counts → simulate closure & recovery → predict
  recovery time; interactive CLI; visual playback.
- Non-functional: runs without SUMO/heavy GPU deps; reproducible (seeded);
  ground-truth-calibrated.

### 6.c.2 System architecture
[Data-flow diagram: `manual_counts/` → `manual_to_pulse` → calibrated demand
→ engine → scenario cache (JSON/CSV) → feature snapshots
(`master_traffic_features.csv`) → RF model → prediction dashboard.]

### 6.c.3 Road-network modelling
- Four `networkx.DiGraph` skeletons (RNB, RSB, VEB, VWB) with the
  merge-then-split topology of Triq Regionali and the branch/merge of
  Valley Road. Reference Fig. 3.1–3.4 and Table 3.1.
- Edge attributes: `length_m`, `lanes`, `max_occupancy_m`, `v_free_mps`,
  `is_inflow/exit/escape`, `display_name`.

### 6.c.4 Incident and recovery design
[Static closure vs. closure+recovery modes; transition states
(2→1, 2→0, 1→0 closed) per Table 3.2; horizons H ∈ {600, 900, 1200}s,
recovery runs to 2H.]

### 6.c.5 Physics design decisions
- Stochastic merge friction (20–30 % over first 30 m at every junction).
- 50 % bottleneck-capacity penalty on partial lane closure.
- Dynamic escape-way diversion curve (5–10 % base → 40 % under backpressure).
- Censored-recovery extrapolation (final-120 s slope, capped 3600 s).

### 6.c.6 ML design
[Feature/label contract (Table 4.2); choice of explicitly-categorical
one-hot RF over per-road sub-models; rationale.]

## 6.d Implementation

> *(~8–12 pages. Map to the actual modules.)*

### 6.d.1 Data pipeline — manual counts to demand pulses
[`data_processing/manual_to_pulse.py`; per-minute bins; calibration to
match observed exit flow (`simulation/calibrator.py`).]

### 6.d.2 The simulation engine
[`simulation/engine.py`: cell-transmission state, per-cell Greenshields
speed, source-queue + acceptance-credit backpressure, looped CSV inflow,
seed-driven stochastic spawn timing/order, recovery trigger + rolling-V_avg
detection and extrapolation.]

### 6.d.3 Networks and configuration
[`simulation/networks.py` hardcoded geometry + layouts;
`config/simulation.yaml` parameter block (Table 4.1).]

### 6.d.4 Scenario generation (decoupled batch modes)
[`simulation/scenarios.py`: Mode A standard 1× batch (visual playback +
run_id=1) vs. Mode B 5× ML-factory (headless seeds 1–4, JSON on the 5th,
every run logs a snapshot). Duration-specific subfolders; file isolation.]

### 6.d.5 Feature extraction
[`simulation/build_features.py`: T = H snapshot, the eight features +
target, purge routine, `master_traffic_features.csv` contract.]

### 6.d.6 ML training and inference
[`ml/train_model.py` (ColumnTransformer + GridSearchCV + joblib export to
`models/recovery_rf_model.pkl`); `ml/predict.py` baseline-lookup inference.]

### 6.d.7 Visualisation and CLI
[`simulation/visualizer.py` heatmap + particle animation + phase HUD +
recovery progress bar; `simulation/cli.py` 7-option suite, drill-downs,
predictive dashboard.]

## 6.e Evaluation and Testing

> *(~6–10 pages — the results chapter.)*

### 6.e.1 Physics validation (face validity)
[Audit results: merge friction on all junctions; 50 % partial cap; escape
diversion curve; backpressure ripple (Fig. 5.1). Reference
`reports/validation_notes.md`.]

### 6.e.2 Calibration accuracy
[How closely simulated exit flow matched manual counts per slot/direction.]

### 6.e.3 Scenario library
[Composition: 162 scenarios, 810 multi-seed feature rows (Table 5.1);
distribution of observed vs. extrapolated recovery times.]

### 6.e.4 Model performance
[Table 5.2: R² ≈ 0.98, MAE ≈ 127 s, best hyper-parameters; predicted-vs-
actual scatter (Fig. 5.3); feature importances (Fig. 5.2) —
backpressure_segment_count and spatial_occupancy_ratio dominate.]

### 6.e.5 Sensitivity and robustness
[Effect of incident_duration on recovery; per-network behaviour;
stochastic-seed variance across the 5 runs.]

### 6.e.6 Threats to validity / limitations
[Synthetic-ground-truth circularity (model trained on simulator output, not
field recovery measurements); 3 time slots; macroscopic abstraction;
3600 s cap on censored targets.]

## 6.f Future Work

> *(~1–2 pages.)*

- Field-measure real recovery times to break the synthetic-circularity loop.
- Real-world georeferenced map overlays (the visualiser currently uses
  abstract layouts).
- More corridors / dynamic OD demand; weather and signal effects.
- Alternative learners (gradient boosting, quantile regression for the
  censored tail); online prediction from live sensor feeds.

## 6.g Conclusions

> *(~1–2 pages.)*

[Restate aim and whether it was met; summarise the engine, dataset, and
predictor; the single most useful finding (queue size at reopening
dominates recovery time); the practical artefact delivered.]

---

# 7. Appendices

### Appendix A — User Guide
[Install (`pip install -r requirements.txt`, no SUMO needed); run
`python main.py`; the 7 menu options; how to regenerate data and retrain.]

### Appendix B — Contents of the Digital Media
[Repository tree; key files: `simulation/`, `ml/`, `data/`, `config/`,
`models/recovery_rf_model.pkl`; how to reproduce every figure/table.]

### Appendix C — Road-Network Schematics
[Full-resolution `reports/figures/{RNB,RSB,VEB,VWB}.png` with annotated
edge labels, lengths, lane counts.]

### Appendix D — Manual Data-Collection Instrument
[The CSV schema and field-counting protocol from
`data/raw/manual_counts/README.md`; sample filled sheets.]

### Appendix E — Complete Data Samples
[Excerpts of `master_traffic_features.csv`; one full scenario telemetry CSV;
calibration JSON examples.]

### Appendix F — Configuration Reference
[Full `config/simulation.yaml` with every parameter explained.]

---

# 8. Bibliography and References

> *(Use the institution's required style — IEEE / Harvard / APA. Seed list
> of the works the Background chapter must cite.)*

1. Greenshields, B. D. (1935). *A study of traffic capacity.* Highway
   Research Board Proceedings.
2. Daganzo, C. F. (1994). *The cell transmission model: A dynamic
   representation of highway traffic…* Transportation Research Part B.
3. Lighthill, M. J., & Whitham, G. B. (1955). *On kinematic waves II: A
   theory of traffic flow on long crowded roads.* Proc. Royal Society A.
4. Richards, P. I. (1956). *Shock waves on the highway.* Operations Research.
5. Breiman, L. (2001). *Random forests.* Machine Learning, 45(1).
6. Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR.
7. Hagberg, Schult, & Swart (2008). *Exploring network structure… with
   NetworkX.* Proc. 7th Python in Science Conf.
8. Lopez et al. (2018). *Microscopic Traffic Simulation using SUMO.* IEEE ITSC.
9. [Maltese transport / Transport Malta statistics source.]
10. [Incident-recovery / network-resilience reference.]

<!-- TODO before submission:
     - Replace every [bracket] with prose.
     - Generate Fig 5.1–5.3 from the telemetry CSVs and the trained model.
     - Convert to the institution's typesetting template (Word/LaTeX).
     - Auto-build ToC / lists; finalise reference style. -->
