#!/usr/bin/env python3
"""macOS SUMO wrapper for Triq Regionali Phase 2.

This script discovers SUMO app bundles under /Applications, sets SUMO_HOME and
PATH for the duration of the run, and launches the existing pipeline from
simulation/run_sumo.py with beginner-friendly commands.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SIM_RUN_SCRIPT = PROJECT_ROOT / "simulation" / "run_sumo.py"

SUMO_APP_ROOTS = [Path("/Applications/SUMO"), Path("/Applications")]
SUMO_EXECUTABLES = ["sumo-gui", "netedit", "sumo", "netconvert"]
APP_BUNDLES = ["sumo-gui.app", "netedit.app"]


class SumAppDiscoveryError(RuntimeError):
    pass


def _binary_candidates(root: Path) -> dict[str, Path]:
    """Find SUMO binaries under a known application-folder root."""
    found: dict[str, Path] = {}
    for app in APP_BUNDLES:
        app_mac = root / app / "Contents" / "MacOS"
        if not app_mac.exists():
            continue
        for exe in SUMO_EXECUTABLES:
            candidate = app_mac / exe
            if candidate.exists() and candidate.is_file():
                found[exe] = candidate.resolve()
    return found


def _scan_applications() -> dict[str, Path]:
    """Fallback search for SUMO binaries inside /Applications."""
    found: dict[str, Path] = {}
    applications = Path("/Applications")
    if not applications.exists():
        return found

    for exe in SUMO_EXECUTABLES:
        if exe in found:
            continue
        for match in applications.rglob(exe):
            if match.is_file() and os.access(match, os.X_OK):
                found[exe] = match.resolve()
                break
    return found


def discover_sumo_binaries() -> dict[str, Path]:
    """Discover SUMO binaries on macOS by checking known .app bundle locations."""
    found: dict[str, Path] = {}
    for root in SUMO_APP_ROOTS:
        if not root.exists():
            continue
        discovered = _binary_candidates(root)
        found.update(discovered)
        if found:
            break

    if not found:
        found = _scan_applications()

    return found


def infer_sumo_home(found: dict[str, Path]) -> Path | None:
    """Infer SUMO_HOME from a discovered binary path."""
    if not found:
        return None
    first_binary = next(iter(found.values()))
    # Expect /Applications/SUMO/<app>.app/Contents/MacOS/<binary>
    if len(first_binary.parents) >= 4:
        return first_binary.parents[3]
    return None


def configure_sumo_environment(found: dict[str, Path]) -> None:
    """Set SUMO_HOME, add tool paths to sys.path, and inject binary paths into PATH."""
    if not found:
        raise SumAppDiscoveryError(
            "Unable to discover SUMO app bundles in /Applications. "
            "Please ensure SUMO apps are in your Applications folder and not still in the Downloads folder."
        )

    binary_dirs = {str(path.parent) for path in found.values()}
    existing_path = os.environ.get("PATH", "")
    os.environ["PATH"] = os.pathsep.join(sorted(binary_dirs)) + os.pathsep + existing_path

    sumo_home = infer_sumo_home(found)
    if sumo_home is not None:
        os.environ["SUMO_HOME"] = str(sumo_home)
        tools_path = sumo_home / "tools"
        if tools_path.exists() and str(tools_path) not in sys.path:
            sys.path.insert(0, str(tools_path))


def _run_subprocess(command: list[str]) -> int:
    print("$ " + " ".join(command))
    return subprocess.run(command, cwd=PROJECT_ROOT, env=os.environ).returncode


def _require_executables(found: dict[str, Path], required: list[str]) -> None:
    missing = [exe for exe in required if exe not in found]
    if missing:
        raise SumAppDiscoveryError(
            "Missing SUMO executables: " + ", ".join(missing) + ".\n"
            "Please ensure the SUMO app bundles are installed under /Applications, "
            "and not still in the Downloads folder."
        )


def build_network(segment: str = "regional_road") -> None:
    if not SIM_RUN_SCRIPT.exists():
        raise FileNotFoundError(f"Missing simulation script: {SIM_RUN_SCRIPT}")
    command = [sys.executable, str(SIM_RUN_SCRIPT), "--build", "--segment", segment]
    rc = _run_subprocess(command)
    if rc != 0:
        raise RuntimeError("Build failed")


def preview_road(segment: str = "regional_road") -> None:
    command = [sys.executable, str(SIM_RUN_SCRIPT), "--preview", "--segment", segment, "--netedit"]
    rc = _run_subprocess(command)
    if rc != 0:
        raise RuntimeError("Preview failed")


def simulate_traffic(direction: str = "regional_south", slot: str = "10am") -> None:
    command = [sys.executable, str(SIM_RUN_SCRIPT), "--simulate", "--direction", direction, "--slot", slot]
    rc = _run_subprocess(command)
    if rc != 0:
        raise RuntimeError("Simulation failed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase 2 SUMO workflows on macOS with SUMO .app discovery.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--view-road", action="store_true",
                       help="Build the regional_road network and open it in NetEdit.app")
    group.add_argument("--view-sim", action="store_true",
                       help="Build the network and launch sumo-gui with regional_south traffic")
    group.add_argument("--build", action="store_true",
                       help="Build the regional_road network only")

    parser.add_argument("--segment", default="regional_road",
                        help="Segment to build/preview")
    parser.add_argument("--direction", default="regional_south",
                        help="Direction to simulate when using --view-sim")
    parser.add_argument("--slot", default="10am",
                        help="Time slot to simulate when using --view-sim")
    args = parser.parse_args()

    found = discover_sumo_binaries()
    configure_sumo_environment(found)

    if args.build:
        print("[run.py] Building network...")
        build_network(args.segment)
        return

    if args.view_road:
        print("[run.py] Building and previewing the road in NetEdit...")
        build_network(args.segment)
        preview_road(args.segment)
        return

    if args.view_sim:
        print("[run.py] Building and launching SUMO-GUI simulation...")
        build_network(args.segment)
        simulate_traffic(args.direction, args.slot)
        return


if __name__ == "__main__":
    try:
        main()
    except SumAppDiscoveryError as exc:
        sys.exit(str(exc))
    except Exception as exc:
        sys.exit(f"ERROR: {exc}")
