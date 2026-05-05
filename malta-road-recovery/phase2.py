#!/usr/bin/env python3
"""
Phase 2 SUMO wrapper for macOS M3 — Fixed paths + XQuartz + Anaconda compatibility.

Features:
  • Hardcoded SUMO .app bundle paths (no discovery needed)
  • XQuartz display support (DISPLAY=:0)
  • python3 enforcement for helper scripts
  • Subprocess.Popen for GUI apps (non-blocking)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SIM_RUN_SCRIPT = PROJECT_ROOT / "simulation" / "run_sumo.py"

# Hardcoded SUMO paths for M3 Mac
SUMO_HOME = Path("/Applications/SUMO sumo-gui.app/Contents/share/sumo")

# GUI binaries in their respective .app bundles
NETEDIT_BINARY = Path("/Applications/SUMO netedit.app/Contents/MacOS/SUMO netedit")
SUMO_GUI_BINARY = Path("/Applications/SUMO sumo-gui.app/Contents/MacOS/SUMO sumo-gui")

# Netconvert might be in PATH or bundled (location varies by installation)
SUMO_BIN_DIR = Path("/Applications/SUMO sumo-gui.app/Contents/MacOS")
NETCONVERT_BINARY = SUMO_BIN_DIR / "netconvert"

# XQuartz display for macOS
XQUARTZ_DISPLAY = ":0"


def setup_environment() -> None:
    """Configure SUMO_HOME, PATH, and XQuartz display."""
    os.environ["SUMO_HOME"] = str(SUMO_HOME)
    os.environ["SUMO_BIN_DIR"] = str(SUMO_BIN_DIR)
    os.environ["DISPLAY"] = XQUARTZ_DISPLAY

    # Add both GUI app bin directories to PATH
    bin_paths = [
        str(Path("/Applications/SUMO netedit.app/Contents/MacOS")),
        str(Path("/Applications/SUMO sumo-gui.app/Contents/MacOS")),
        str(SUMO_BIN_DIR),
    ]
    path = os.environ.get("PATH", "")
    new_path = os.pathsep.join(bin_paths) + os.pathsep + path
    os.environ["PATH"] = new_path

    print("✓ Successfully linked to SUMO engine at /Applications/SUMO sumo-gui.app")
    print(f"  SUMO_HOME     = {SUMO_HOME}")
    print(f"  SUMO_BIN_DIR  = {SUMO_BIN_DIR}")
    print(f"  DISPLAY       = {XQUARTZ_DISPLAY}")


def validate_binaries() -> None:
    """Check that GUI apps exist. netconvert will be checked at build time."""
    gui_required = {
        "SUMO netedit": NETEDIT_BINARY,
        "SUMO sumo-gui": SUMO_GUI_BINARY,
    }
    missing_gui = [name for name, path in gui_required.items() if not path.exists()]

    if missing_gui:
        sys.exit(
            f"Missing SUMO GUI apps: {', '.join(missing_gui)}\n"
            "Please ensure SUMO apps are in /Applications/SUMO\n"
            "and not still in the Downloads folder."
        )


def find_python3() -> str:
    """Resolve python3 executable, preferring conda environment."""
    # Check if running in conda environment
    if "CONDA_PREFIX" in os.environ:
        conda_python = Path(os.environ["CONDA_PREFIX"]) / "bin" / "python"
        if conda_python.exists():
            return str(conda_python)
    
    # Fall back to system python3
    for candidate in ["python3", "python", "/usr/bin/python3"]:
        try:
            result = subprocess.run([candidate, "--version"], capture_output=True, timeout=2)
            if result.returncode == 0:
                return candidate
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return sys.executable


def run_build(segment: str = "regional_road") -> None:
    """Build network from roads.yaml using netconvert."""
    validate_binaries()

    if not SIM_RUN_SCRIPT.exists():
        raise FileNotFoundError(f"Missing: {SIM_RUN_SCRIPT}")

    python_exe = find_python3()
    cmd = [python_exe, str(SIM_RUN_SCRIPT), "--build", "--segment", segment]

    print(f"\n[phase2] Building {segment} network…")
    print(f"$ {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=os.environ)
    if result.returncode != 0:
        raise RuntimeError(f"Build failed with exit code {result.returncode}")


def run_preview_netedit(segment: str = "regional_road") -> None:
    """Preview road network in netedit.app (XQuartz)."""
    validate_binaries()

    if not SIM_RUN_SCRIPT.exists():
        raise FileNotFoundError(f"Missing: {SIM_RUN_SCRIPT}")

    python_exe = find_python3()
    cmd = [python_exe, str(SIM_RUN_SCRIPT), "--preview", "--segment", segment, "--netedit"]

    print(f"\n[phase2] Launching netedit…")
    print(f"$ {' '.join(cmd)}\n")

    subprocess.Popen(cmd, cwd=PROJECT_ROOT, env=os.environ)


def check_road(segment: str = "regional_road") -> None:
    """Alias for run_preview_netedit — launches SUMO netedit to preview the network."""
    run_preview_netedit(segment)


def run_simulation_gui(direction: str = "regional_south", slot: str = "10am") -> None:
    """Run traffic simulation in sumo-gui.app (XQuartz)."""
    validate_binaries()

    if not SIM_RUN_SCRIPT.exists():
        raise FileNotFoundError(f"Missing: {SIM_RUN_SCRIPT}")

    python_exe = find_python3()
    cmd = [python_exe, str(SIM_RUN_SCRIPT), "--simulate", "--direction", direction, "--slot", slot]

    print(f"\n[phase2] Launching SUMO simulation…")
    print(f"  Direction: {direction}")
    print(f"  Time slot: {slot}")
    print(f"$ {' '.join(cmd)}\n")

    subprocess.Popen(cmd, cwd=PROJECT_ROOT, env=os.environ)


def run_sim(direction: str = "regional_south", slot: str = "10am") -> None:
    """Alias for run_simulation_gui — launches SUMO sumo-gui to run traffic simulation."""
    run_simulation_gui(direction, slot)


def main() -> None:
    setup_environment()

    parser = argparse.ArgumentParser(
        description="Phase 2 SUMO — XQuartz-ready network builder & simulator for Triq Regionali.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--build",
        action="store_true",
        help="Build network only (no GUI)",
    )
    mode_group.add_argument(
        "--preview",
        action="store_true",
        help="Build & open netedit (network editor)",
    )
    mode_group.add_argument(
        "--simulate",
        action="store_true",
        help="Build & launch sumo-gui with traffic",
    )

    parser.add_argument(
        "--segment",
        default="regional_road",
        choices=["regional_road", "valley_road"],
        help="Road segment to build/preview",
    )
    parser.add_argument(
        "--direction",
        default="regional_south",
        choices=["regional_north", "regional_south", "valley_east", "valley_west"],
        help="Traffic direction (for --simulate)",
    )
    parser.add_argument(
        "--slot",
        default="10am",
        help="Time slot for traffic CSV (e.g., 10am, 1pm, 5pm)",
    )

    args = parser.parse_args()

    try:
        if args.build:
            run_build(args.segment)
        elif args.preview:
            run_build(args.segment)
            run_preview_netedit(args.segment)
        elif args.simulate:
            run_build(args.segment)
            run_simulation_gui(args.direction, args.slot)
    except Exception as exc:
        sys.exit(f"ERROR: {exc}")


if __name__ == "__main__":
    main()
