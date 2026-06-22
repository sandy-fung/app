"""Centralized camera manager — init once, always running."""

import threading
from typing import Optional

import cv2
import numpy as np


class CameraManager:
    """Manages DVS and RGB cameras with mode switching.

    DVS supports two modes:
      - "tracking": DVS-only ~200fps for laser tracking
      - "hybrid": RGB+DVS for calibration preview
    """

    def __init__(self, dvs_device: int, rgb_device: str):
        self._dvs_device = dvs_device
        self._rgb_device = rgb_device
        self._rgb_cap: Optional[cv2.VideoCapture] = None
        self._rgb_lock = threading.Lock()
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
        """Read one DVS event frame, rotated for display. Returns (164, 160) uint8 or None."""
        if self._xe_cam is None:
            return None
        frame = self._xe_cam.get_frame_laser_nparray()
        if frame is not None:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            frame = cv2.flip(frame, 1)
        return frame

    def read_rgb_frame(self, rotate: bool = True) -> Optional[np.ndarray]:
        """Read one RGB frame. Returns BGR or None.

        Args:
            rotate: Apply ``RGB_DISPLAY_ROTATE`` rotation (default True).
                    Pass False to get the raw camera frame.

        Thread-safe: guarded by ``_rgb_lock`` so background inference threads
        and the main thread can call this without racing on VideoCapture.
        """
        with self._rgb_lock:
            if self._rgb_cap is None or not self._rgb_cap.isOpened():
                return None
            ret, frame = self._rgb_cap.read()
        if not ret:
            return None
        if rotate:
            from app.config import RGB_DISPLAY_ROTATE
            _ROTATE_FLAGS = {
                90: cv2.ROTATE_90_CLOCKWISE,
                180: cv2.ROTATE_180,
                270: cv2.ROTATE_90_COUNTERCLOCKWISE,
            }
            flag = _ROTATE_FLAGS.get(RGB_DISPLAY_ROTATE)
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
        self._run_oneshot_ae()

    def _run_oneshot_ae(self) -> None:
        """Converge auto-exposure once, then leave settings locked in the sensor.

        Runs after a hybrid-mode switch so the 2D preview is bright enough to
        see in the current lighting. Bounded by AE_MAX_FRAMES to avoid hanging
        if the scene never converges (e.g. fully dark or clipped). Failures
        are logged but non-fatal — the user still gets the un-adjusted frame.
        """
        from app.config import (
            AE_MAX_FRAMES,
            AE_TARGET_BRIGHTNESS,
        )

        try:
            from auto_exposure import AEConfig, AutoExposure
        except Exception as exc:
            print(f"[CAM] one-shot AE skipped: import failed ({exc})")
            return

        xe = self._xe_cam
        cap = xe.g_cap
        ae = AutoExposure(cap, AEConfig(target_brightness=AE_TARGET_BRIGHTNESS))

        for i in range(AE_MAX_FRAMES):
            _dvs, gray = cap.XeGetFrame(xe.g_xereal_mode, xe.g_xereal_bit_depth)
            if gray is None:
                continue
            state = ae.update(gray)
            if state.converged:
                print(f"[CAM] one-shot AE converged in {i + 1} frames: "
                      f"expo={state.exposure}, gain=0x{state.gain_index:02X}, "
                      f"brt={state.brightness:.1f}")
                return
        print(f"[CAM] one-shot AE hit frame limit ({AE_MAX_FRAMES}) without "
              f"converging; current brt={ae.state.brightness:.1f}")

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
