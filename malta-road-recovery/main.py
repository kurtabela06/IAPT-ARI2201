"""
main.py
=======
Malta Road Recovery — Phase 2 entry point.

Launches the persistent CLI menu defined in simulation/cli.py.
The CLI never exits when a Matplotlib window closes; it loops back to the
main menu so you can drill into another scenario.

If you need the legacy non-interactive batch entry point used to convert
manual CSVs into 1-minute pulses, run:

    python data_processing/manual_to_pulse.py --all
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from simulation.cli import run_menu     # noqa: E402


if __name__ == "__main__":
    run_menu()
