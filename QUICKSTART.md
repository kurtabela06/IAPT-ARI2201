# QUICKSTART

**Test this in < 5 minutes**

```bash
# 1. Install
pip install -r requirements.txt

# 2. Launch interactive menu
cd malta-road-recovery
python main.py

# 3. In the menu, select [1] to view a single scenario
#    Drill through: any road → any time slot → any horizon → any variant
#    A Matplotlib window should pop up showing live animation

# 4. Verify calibration works (will take ~30s on first run)
#    After first scenario, subsequent runs cache and are instant (~3s)

# 5. To regenerate the full ML pipeline (optional):
#    [3] RUN STANDARD BATCH     (~5 min: 162 scenarios)
#    [4] RUN ML FACTORY BATCH   (~5 min: 810 scenarios × 5 seeds)
#    [5] TRAIN ML MODEL         (~15 sec: GridSearchCV)
#    [6] PREDICT RECOVERY       (interactive inference dashboard)
```

## Expected Outputs

- **Scenario animation** (option [1]): Heatmap of road network with green/orange/red occupancy. Particles move along edges. Animation loops until window closes.
- **Trained model** (after [5]): `models/recovery_rf_model.pkl` (~200 KB). Should see R² ≈ 0.98 on console.
- **Prediction** (option [6]): Prompts for road/slot/duration/transition, returns recovery time in seconds.

## No Manual Configuration Needed

All parameters (Greenshields limits, merge friction, calibration bounds, recovery thresholds) are pre-configured in `config/simulation.yaml`. The manual CSV counts are already in `data/raw/manual_counts/`. Just run the menu.

---

See `README.md` for full documentation.
