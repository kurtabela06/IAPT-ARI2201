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

## Subsequent phases

- **Phase 2/3 — SUMO & TraCI**: consume the `.rou.xml` / `.flow.xml` under
  `simulation/demand/` and the aggregated bins under
  `data/processed/aggregated/`.
- **Phase 4 — ML**: feature matrices live in `data/features/`.

## Archive

`Archive/` at the repo root contains the scrapped CV pipeline for reference
only — `detect.py`, `virtual_gates.py`, `utils.py`, `config.py`,
`video_config.py`, the historical `yolo_output/` directory, the per-clip
calibration YAMLs, and the CV-specific unit tests.
