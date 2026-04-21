# Manual Counts — Ground Truth Data Entry

This folder holds the manually-filled CSVs that feed Phase 1. The automated
YOLO pipeline has been archived (see `Archive/` at the repo root).

## Folder layout

One subfolder per segment × direction:

```
regional_north/
regional_south/
valley_east/
valley_west/
```

Inside each direction, one CSV per time slot, named `<slot>.csv`:

```
regional_north/10am.csv
regional_north/1pm.csv
regional_north/5pm.csv
regional_south/10am.csv
...
valley_west/5pm.csv
```

The `slot` is free-text (e.g. `10am`, `1pm`, `5pm`) — it becomes part of the
SUMO demand file names downstream.

## CSV schema — one row per 1-minute bin

| column               | type  | description                                                         |
| -------------------- | ----- | ------------------------------------------------------------------- |
| timestamp_bin        | int   | 1-minute bin index (0, 1, 2, …).                                    |
| car_count            | int   | Number of cars that crossed the counting line in this bin.          |
| heavy_vehicle_count  | int   | Buses + trucks combined in this bin.                                |
| avg_speed_kph        | float | Mean speed observed for the bin (km/h).                             |
| incident_status      | str   | One of: `clear`, `slow`, `blocked`, `incident`.                     |

The observer has already done the aggregation per bin — this CSV is the
1-minute bin file, not a per-vehicle log.

## Example

```csv
timestamp_bin,car_count,heavy_vehicle_count,avg_speed_kph,incident_status
0,12,2,52.3,clear
1,10,1,50.1,clear
2,8,3,38.7,slow
3,5,0,12.0,blocked
4,7,1,44.8,clear
```

## Running

```
python main.py                                                   # every direction, every slot
python main.py --input data/raw/manual_counts/valley_east/1pm.csv  # single
```
