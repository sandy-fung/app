"""Shared calibration data — written by CalibrationDemo, read by TrackingDemo."""

import json
import os
from typing import Optional

import cv2
import numpy as np


class CalibrationStore:
    """Thread-safe store for DVS and RGB calibration data.

    CalibrationDemo writes calibration results here;
    TrackingDemo reads homography matrices for coordinate warping.
    """

    def __init__(self):
        self.dvs_corners: Optional[np.ndarray] = None    # (4,2) float32
        self.dvs_homography: Optional[np.ndarray] = None  # 3x3
        self.rgb_quad = None           # QuadTarget
        self.rgb_homography: Optional[np.ndarray] = None  # 3x3

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def dvs_calibrated(self) -> bool:
        return self.dvs_homography is not None

    @property
    def rgb_calibrated(self) -> bool:
        return self.rgb_homography is not None

    # ------------------------------------------------------------------
    # DVS calibration
    # ------------------------------------------------------------------

    def set_dvs(self, corners: np.ndarray) -> None:
        """Set DVS calibration from interactive corner selection."""
        from quad_calibrator import compute_homography
        self.dvs_corners = corners
        self.dvs_homography = compute_homography(corners)

    def save_dvs(self, path: str) -> bool:
        """Persist DVS corners to JSON file. Returns True on success."""
        from quad_calibrator import save_calibration
        if self.dvs_corners is None:
            return False
        return save_calibration(self.dvs_corners, path)

    def load_dvs(self, path: str) -> bool:
        """Load saved DVS corners and recompute homography.

        Returns True if loaded successfully.
        """
        from quad_calibrator import load_calibration, compute_homography
        if not os.path.isfile(path):
            return False
        try:
            corners = load_calibration(path)
        except Exception as e:
            print(f"[CAL] Failed to load {path}: {e}")
            return False
        if corners is not None:
            self.dvs_corners = corners
            self.dvs_homography = compute_homography(corners)
            return True
        return False

    # ------------------------------------------------------------------
    # RGB calibration
    # ------------------------------------------------------------------

    def set_rgb(self, quad) -> None:
        """Set RGB calibration from detected QuadTarget."""
        self.rgb_quad = quad
        src = quad.corners.astype(np.float32)
        dst = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32)
        self.rgb_homography = cv2.getPerspectiveTransform(src, dst)

    def save_rgb(self, path: str) -> bool:
        """Persist RGB quad corners to JSON file."""
        if self.rgb_quad is None:
            return False
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            data = {"corners": self.rgb_quad.corners.tolist()}
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            return True
        except OSError as e:
            print(f"[CAL] Failed to save RGB calibration: {e}")
            return False

    def load_rgb(self, path: str) -> bool:
        """Load saved RGB quad corners and recompute homography."""
        if not os.path.isfile(path):
            return False
        try:
            with open(path) as f:
                data = json.load(f)
            corners = np.array(data["corners"], dtype=np.float32)
            if corners.shape != (4, 2):
                raise ValueError(f"Invalid corners shape: {corners.shape}")
        except Exception as e:
            print(f"[CAL] Failed to load {path}: {e}")
            return False
        contour = corners.reshape(-1, 1, 2).astype(np.float32)
        from quad_detector import QuadTarget
        quad = QuadTarget(
            corners=corners,
            area=float(cv2.contourArea(contour)),
            perimeter=float(cv2.arcLength(contour, True)),
            contour=contour,
        )
        self.set_rgb(quad)
        return True
