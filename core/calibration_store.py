"""Shared calibration data — written by CalibrationDemo, read by TrackingDemo."""

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

    def save_dvs(self, path: str) -> None:
        """Persist DVS corners to JSON file."""
        from quad_calibrator import save_calibration
        if self.dvs_corners is not None:
            save_calibration(self.dvs_corners, path)

    def load_dvs(self, path: str) -> bool:
        """Load saved DVS corners and recompute homography.

        Returns True if loaded successfully.
        """
        from quad_calibrator import load_calibration, compute_homography
        if not os.path.isfile(path):
            return False
        corners = load_calibration(path)
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
