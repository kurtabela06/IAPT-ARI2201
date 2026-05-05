# Manual Counts — Ground Truth Data Entry

This folder holds the manually-filled CSVs that feed Phase 1. The automated
YOLO pipeline has been archived (see `Archive/` at the repo root). The Phase 2
SUMO pipeline has been replaced by a custom NetworkX engine (see
[`malta-road-recovery/simulation/`](../../../simulation/)).

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
…
valley_west/5pm.csv
```

The `slot` is free-text (`10am`, `1pm`, `5pm`) — it shows up in the demand
pulse filenames downstream.

## CSV schema (one row per 1-minute bin)

| column                | type  | required | description                                                  |
| --------------------- | ----- | -------- | ------------------------------------------------------------ |
| timestamp_bin         | int   | yes      | 1-minute bin index (0, 1, 2, …)                              |
| car_count             | int   | yes      | Cars crossing the counting line in this bin                  |
| motorcycle_count      | int   | no       | Motorcycles in this bin (defaults to 0 if column is absent)  |
| heavy_vehicle_count   | int   | yes      | Buses + trucks combined                                      |
| avg_car_speed_kph     | float | yes      | Mean car speed observed for the bin (informational only)     |
| incident_status       | str   | yes      | `clear` / `slow` / `blocked` / `incident`                    |

The observer has already aggregated per bin — this CSV is the 1-minute bin
file, not a per-vehicle log.

## What the counts represent

The counts you record are the **total number of vehicles passing the camera
in that minute**, regardless of which lane or which entry road they came
from. The Phase 2 NetworkX engine apportions them across the inflow edges
of the matching road skeleton (e.g. RNB has Mergeway A and Underpass; RSB
has Flyover and Underpass; VWB has Valley + Psaila + Fleur-de-lys).

You do **not** need to count entry roads separately. The engine apportions
by edge capacity (lanes × length), then runs Greenshields on the result.

**No PCE / density / flow-rate columns.** Those derivations live in the
engine and are computed from spatial occupancy on each graph edge:

```
occupied_m = car_count × 5m + hgv_count × 8m       (motorcycles excluded)
ratio      = occupied_m / (edge_length_m × lanes)
```

## What the counts represent (Triq Regionali, merge_split topology)

For `regional_north` and `regional_south` the filmed section is the **3-lane
trunk** in the middle of a merge-then-split structure (see the diagram at
the top of `config/roads.yaml` and the project README). The counts you
record represent the **total flow across the trunk** — every vehicle passing
the camera, regardless of whether it came from the Flyover or the Underpass
and regardless of whether it leaves via the Tunnel/Flyover or the
Escape/Underpass.

`manual_to_sumo.py` automatically splits each bin's count across the four
plausible routes (wider inflow / narrower inflow × wider outflow / narrower
outflow) using the shares in `merge_split_demand:` (defaults 70 % wider
inflow, 70 % wider outflow). The sum of the four split counts equals the
original CSV count exactly, so trunk totals are preserved.

You do **not** need to count Flyover and Underpass separately. If you have
time-of-day evidence (e.g. "at 5 pm 80 % uses the Flyover"), edit
`inflow_a_share` / `outflow_main_share` in `roads.yaml` rather than
splitting the counts in the CSV.

Valley Road (`valley_east` / `valley_west`) is single-lane and uses the
`simple` topology — the count goes directly to the one and only route.

## Example

```csv
timestamp_bin,car_count,motorcycle_count,heavy_vehicle_count,avg_car_speed_kph,incident_status
0,12,3,2,52.3,clear
1,10,1,1,50.1,clear
2,8,0,3,38.7,slow
3,5,2,0,12.0,blocked
4,7,1,1,44.8,clear
```

## Running

```
# Convert all 12 manual CSVs into 1-minute pulse files
python data_processing/manual_to_pulse.py --all

# Or one at a time
python data_processing/manual_to_pulse.py \
    --input data/raw/manual_counts/valley_east/1pm.csv

# Then launch the persistent menu
python main.py
```
