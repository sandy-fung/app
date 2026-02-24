"""Centralized camera manager — init once, always running."""

from typing import Optional, Tuple

import cv2
import numpy as np


class CameraManager:
    """Manages DVS and RGB cameras with mode switching.

    DVS supports two modes:
      - "tracking": DVS-only ~200fps for laser tracking
      - "hybrid": RGB+DVS for calibration preview
    """

    def __init__(self, dvs_device: int, rgb_device: str, rgb_rotate: int = 90):
        self._dvs_device = dvs_device
        self._rgb_device = rgb_device
        self._rgb_rotate = rgb_rotate
        self._rgb_cap: Optional[cv2.VideoCapture] = None
        self._xe_cam = None  # XenReal module reference
        self._dvs_mode: Optional[str] = None

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def init_dvs(self) -> None:
        """Open DVS camera in tracking mode."""
        from app.config import DVS_ONLY_CONFIG

        import example_open_xe_001d_laser as xe_cam
        self._xe_cam = xe_cam
        xe_cam.DEVICE = f"/dev/video{self._dvs_device}"
        xe_cam.CONFIG_ABS_PATH = DVS_ONLY_CONFIG
        xe_cam.start_camera_laser()
        self._dvs_mode = "tracking"

    def init_rgb(self) -> None:
        """Open RGB camera."""
        dev: object
        if self._rgb_device.isdigit():
            dev = int(self._rgb_device)
        else:
            dev = self._rgb_device
        self._rgb_cap = cv2.VideoCapture(dev)
        if isinstance(dev, int):
            self._rgb_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self._rgb_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # ------------------------------------------------------------------
    # Frame reading
    # ------------------------------------------------------------------

    @property
    def xe_cam(self):
        """Direct access to XenReal module (for DVSReaderThread etc.)."""
        return self._xe_cam

    def read_dvs_frame(self) -> Optional[np.ndarray]:
        """Read one DVS event frame. Returns (160, 164) uint8 or None."""
        if self._xe_cam is None:
            return None
        return self._xe_cam.get_frame_laser_nparray()

    def read_rgb_frame(self) -> Optional[np.ndarray]:
        """Read one RGB frame with rotation applied. Returns BGR or None."""
        if self._rgb_cap is None or not self._rgb_cap.isOpened():
            return None
        ret, frame = self._rgb_cap.read()
        if not ret:
            return None
        if self._rgb_rotate and self._rgb_rotate != 0:
            from laser_tracker import ROTATE_FLAGS
            flag = ROTATE_FLAGS.get(self._rgb_rotate)
            if flag is not None:
                frame = cv2.rotate(frame, flag)
        return frame

    # ------------------------------------------------------------------
    # DVS mode switching
    # ------------------------------------------------------------------

    def switch_dvs_to_hybrid(self) -> None:
        """Switch DVS to hybrid RGB+DVS mode (for calibration)."""
        if self._dvs_mode == "hybrid":
            return
        from app.config import HYBRID_CONFIG
        self._xe_cam.close_camera(self._xe_cam.g_cap)
        self._xe_cam.CONFIG_ABS_PATH = HYBRID_CONFIG
        self._xe_cam.start_camera_laser()
        self._dvs_mode = "hybrid"

    def switch_dvs_to_tracking(self) -> None:
        """Switch DVS to DVS-only tracking mode (~200fps)."""
        if self._dvs_mode == "tracking":
            return
        from app.config import DVS_ONLY_CONFIG
        self._xe_cam.close_camera(self._xe_cam.g_cap)
        self._xe_cam.CONFIG_ABS_PATH = DVS_ONLY_CONFIG
        self._xe_cam.start_camera_laser()
        self._dvs_mode = "tracking"

    @property
    def dvs_mode(self) -> Optional[str]:
        """Current DVS mode: 'tracking', 'hybrid', or None."""
        return self._dvs_mode

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Release all camera resources."""
        if self._xe_cam is not None:
            try:
                self._xe_cam.close_camera(self._xe_cam.g_cap)
            except Exception:
                pass
        if self._rgb_cap is not None:
            self._rgb_cap.release()
