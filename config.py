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
DVS_SCALE = 3           # DVS preview magnification factor
CANVAS_SIZE = 400       # Trajectory canvas size in pixels
RGB_DISPLAY_ROTATE = 90 # CW rotation to correct camera mounting
IDLE_CLEAR = 1.0        # Auto-clear canvas after N seconds idle

# DVS camera config files
DVS_ONLY_CONFIG = (
    "/workspace/xenreal_001d/ESC001D_DV_RAW4_200FPS_20260204_modify.cfg"
)
HYBRID_CONFIG = "/workspace/xenreal_001d/ESC001D_2D_RAW8_DV_RAW2.cfg"

# Default calibration / profile paths
DEFAULT_DVS_CAL_PATH = os.path.join(EX15_DIR, "dvs_calibration.json")
DEFAULT_LASER_PROFILE = os.path.join(EX16_DIR, "laser_profile.json")

# ---------------------------------------------------------------------------
# Gesture recognition constants
# ---------------------------------------------------------------------------
DVS_GESTURE_MODEL = (
    "/workspace/gest/xenreal_001d/gest/models/dvs_20260223_105731/best_loss_model.pth"
)
MEDIAPIPE_MODEL = (
    "/workspace/gest/rgb/models/20260119_150909/gesture_recognizer.task"
)
HAND_SDK_PATH = "/workspace/gest/rgb"

# Gesture inference defaults (hardcoded — no CLI)
GESTURE_CONF = 0.5
DVS_HOLD_FRAMES = 10
RGB_HOLD_FRAMES = 2
GESTURE_VOTE_MODE = "none"
DVS_NORMALIZE_CENTER = 9
DVS_NORMALIZE_STEEPNESS = 3.0
HAND_TYPE = "right"
HAND_JOINT = "O6"

# Arrival detection — wait for hand to reach target position
GESTURE_ARRIVAL_THRESHOLD = 15   # max per-joint error to consider "arrived" (0-255 scale)
GESTURE_ARRIVAL_TIMEOUT = 2.0    # fallback timeout in seconds
GESTURE_ARRIVAL_POLL = 0.05      # poll interval in seconds


def setup_sys_path() -> None:
    """Add required paths for importing existing modules."""
    for p in [EX15_DIR, EX16_DIR, EX17_DIR, SRC_DIR, XENREAL_SDK, XENREAL_ROOT]:
        if p not in sys.path:
            sys.path.insert(0, p)


def parse_args() -> argparse.Namespace:
    """Parse unified CLI arguments."""
    p = argparse.ArgumentParser(description="Unified GUI")

    # Camera
    p.add_argument("--dvs-camera", type=int, default=None,
                    help="DVS /dev/video index (auto-detect if omitted)")
    p.add_argument("--rgb-camera", default=None,
                    help="RGB camera index or path (auto-detect if omitted)")

    # Calibration
    p.add_argument("--dvs-cal", default=DEFAULT_DVS_CAL_PATH,
                    help="DVS calibration JSON path")
    p.add_argument("--noise-mask", default=None,
                    help="DVS noise mask path")
    p.add_argument("--load-profile", default=DEFAULT_LASER_PROFILE,
                    help="RGB laser profile JSON path")

    # Arm
    p.add_argument("--can", default="can0", help="CAN bus interface name")
    p.add_argument("--speed", type=float, default=0.3,
                    help="Arm movement speed")
    p.add_argument("--usb-port", default=None,
                    help="USB port address for CAN adapter (auto-detect if omitted)")
    p.add_argument("--no-arm", action="store_true",
                    help="Disable arm hardware")

    # LinkerHand
    p.add_argument("--no-hand", action="store_true",
                    help="Disable LinkerHand hardware")
    p.add_argument("--hand-sdk", default=HAND_SDK_PATH,
                    help="Path to dir containing LinkerHand/ package")

    # CAN shared
    p.add_argument("--can-warmup", type=float, default=3.0,
                    help="Seconds to wait after CAN activation before hardware init")

    return p.parse_args()
