"""Unified app configuration — CLI args, paths, and constants."""

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EX15_DIR = os.path.join(PROJECT_ROOT, "examples", "ex15")
EX16_DIR = os.path.join(PROJECT_ROOT, "examples", "ex16")
EX17_DIR = os.path.join(PROJECT_ROOT, "examples", "ex17")
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
XENREAL_SDK = "/workspace/xenreal_001d/src"
XENREAL_ROOT = "/workspace/xenreal_001d"

# DVS sensor dimensions
DVS_WIDTH = 164
DVS_HEIGHT = 160

# DVS camera config files
DVS_ONLY_CONFIG = (
    "/workspace/xenreal_001d/ESC001D_DV_RAW4_200FPS_20260204_modify.cfg"
)
HYBRID_CONFIG = "/workspace/xenreal_001d/ESC001D_2D_RAW8_DV_RAW2.cfg"

# Default calibration / profile paths
DEFAULT_DVS_CAL_PATH = os.path.join(EX15_DIR, "dvs_calibration.json")
DEFAULT_LASER_PROFILE = os.path.join(EX16_DIR, "laser_profile.json")


def setup_sys_path() -> None:
    """Add required paths for importing existing modules."""
    for p in [EX15_DIR, EX16_DIR, EX17_DIR, SRC_DIR, XENREAL_SDK, XENREAL_ROOT]:
        if p not in sys.path:
            sys.path.insert(0, p)


def parse_args() -> argparse.Namespace:
    """Parse unified CLI arguments."""
    p = argparse.ArgumentParser(description="Unified GUI")

    # Camera
    p.add_argument("--dvs-camera", type=int, default=2,
                    help="DVS /dev/video index")
    p.add_argument("--rgb-camera", default="0",
                    help="RGB camera index or path")
    p.add_argument("--rgb-rotate", type=int, choices=[0, 90, 180, 270],
                    default=90, help="RGB frame rotation")
    p.add_argument("--scale", type=int, default=3,
                    help="DVS preview scale factor")

    # Calibration
    p.add_argument("--dvs-cal", default=DEFAULT_DVS_CAL_PATH,
                    help="DVS calibration JSON path")
    p.add_argument("--noise-mask", default=None,
                    help="DVS noise mask path")
    p.add_argument("--load-profile", default=DEFAULT_LASER_PROFILE,
                    help="RGB laser profile JSON path")

    # Tracking
    p.add_argument("--idle-clear", type=float, default=1.0,
                    help="Auto-clear canvas after N seconds idle")
    p.add_argument("--canvas-size", type=int, default=400,
                    help="Trajectory canvas size in pixels")

    # Arm
    p.add_argument("--can", default="can0", help="CAN bus interface name")
    p.add_argument("--speed", type=float, default=0.3,
                    help="Arm movement speed")
    p.add_argument("--no-arm", action="store_true",
                    help="Disable arm hardware")

    return p.parse_args()
