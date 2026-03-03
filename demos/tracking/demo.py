"""Tracking demo — dual DVS + RGB laser tracking.

Reuses existing components:
  - DVSReaderThread / DVSDrawingThread (from ex17)
  - LaserTracker / LaserProfile (from ex16)
  - DVSLaserTracker (from ex15)
  - TrajectoryCanvas (from ex16)
"""

import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from app.config import DVS_WIDTH, DVS_HEIGHT, DVS_SCALE, CANVAS_SIZE, IDLE_CLEAR
from app.core.demo import Demo


@dataclass
class TrackingResult:
    """Snapshot of one frame's tracking results for output modes."""
    dvs_display: Optional[np.ndarray] = None   # DVS BGR preview (scaled)
    dvs_target: object = None                   # DVSTarget or None
    dvs_warped: Optional[Tuple[float, float]] = None
    dvs_fps: float = 0.0
    rgb_frame: Optional[np.ndarray] = None      # RGB BGR frame
    rgb_target: object = None                   # LaserTarget or None
    rgb_warped: Optional[Tuple[float, float]] = None


class TrackingDemo(Demo):
    """Dual DVS + RGB laser tracking demo tab."""

    def __init__(self, cal_store, args):
        super().__init__("Tracking")
        self._store = cal_store
        self._args = args
        # Components (created in activate)
        self._dvs_reader = None       # DVSReaderThread or DVSDrawingThread
        self._dvs_tracker = None      # DVSLaserTracker
        self._rgb_tracker = None      # LaserTracker
        self._rgb_canvas = None       # TrajectoryCanvas
        self._dvs_canvas = None       # TrajectoryCanvas (persistent across mode switches)
        self._dvs_canvas_lock = None  # threading.Lock
        self._quad_detector = None    # QuadDetector
        self._tracking_enabled = True
        # Latest tracking result for output modes
        self._result = TrackingResult()
        # RGB state
        self._rgb_frame = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def activate(self, camera_mgr) -> None:
        camera_mgr.switch_dvs_to_tracking()

        # Store xe_cam reference for physical output modes
        self._xe_cam = camera_mgr.xe_cam

        # Lazy imports
        from dvs_laser_tracker import DVSLaserTracker
        from dual_tracker_compare import DVSReaderThread
        from laser_tracker import LaserTracker, LaserProfile
        from trajectory_canvas import TrajectoryCanvas
        from quad_detector import QuadDetector
        import os

        # DVS tracker + background reader thread
        self._dvs_tracker = DVSLaserTracker(
            width=DVS_WIDTH, height=DVS_HEIGHT,
            noise_mask_path=self._args.noise_mask,
        )

        # DVS canvas — owned by demo, survives output mode switches
        self._dvs_canvas = TrajectoryCanvas(
            size=CANVAS_SIZE, idle_clear=IDLE_CLEAR, write_confirm=3)
        self._dvs_canvas_lock = threading.Lock()

        self._dvs_reader = DVSReaderThread(
            camera_mgr.xe_cam, self._dvs_tracker,
            self._store.dvs_homography,
            scale=DVS_SCALE,
            canvas=self._dvs_canvas,
            canvas_lock=self._dvs_canvas_lock,
        )
        self._dvs_reader.start()

        # RGB tracker
        if os.path.isfile(self._args.load_profile):
            profile = LaserProfile.load(self._args.load_profile)
            self._rgb_tracker = LaserTracker.from_profile(profile)
        else:
            self._rgb_tracker = LaserTracker()

        # Set RGB ROI from calibration if available
        if self._store.rgb_calibrated and self._store.rgb_quad:
            self._rgb_tracker.roi = self._store.rgb_quad.as_xyxy()

        # RGB canvas
        self._rgb_canvas = TrajectoryCanvas(
            size=CANVAS_SIZE,
            idle_clear=IDLE_CLEAR,
        )

        # Quad detector for re-detection
        if self._quad_detector is None:
            self._quad_detector = QuadDetector()

        # Activate current output mode
        if self.active_output:
            self.active_output.activate()

    def deactivate(self) -> None:
        if self.active_output:
            self.active_output.deactivate()
        if self._dvs_reader:
            self._dvs_reader.stop()
            self._dvs_reader = None

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------

    def process_frame(self, camera_mgr) -> None:
        from main_dvs_tracking import dvs_frame_to_bgr, draw_dvs_target_scaled
        from main_laser_drawing import warp_point as rgb_warp_point

        # --- DVS: get latest from background thread ---
        dvs_frame, dvs_target, dvs_warped, dvs_fps = (
            self._dvs_reader.get_latest()
        )

        # Build DVS display image
        dvs_display = None
        if dvs_frame is not None:
            dvs_display = dvs_frame_to_bgr(dvs_frame, DVS_SCALE)
            if dvs_target is not None:
                draw_dvs_target_scaled(dvs_display, dvs_target, DVS_SCALE)

        # --- RGB: read + detect in main thread ---
        rgb_frame = camera_mgr.read_rgb_frame()
        self._rgb_frame = rgb_frame
        rgb_target = None
        rgb_warped = None

        if rgb_frame is not None and self._rgb_tracker is not None:
            rgb_target = self._rgb_tracker.detect(rgb_frame)

            if rgb_target and self._store.rgb_homography is not None:
                nx, ny = rgb_warp_point(
                    self._store.rgb_homography,
                    rgb_target.cx, rgb_target.cy,
                )
                if 0 <= nx <= 1 and 0 <= ny <= 1:
                    rgb_warped = (nx, ny)
                    self._rgb_canvas.update(True, nx, ny)
                else:
                    self._rgb_canvas.update(False, 0, 0)
            elif rgb_target:
                # No calibration — fallback to simple normalization
                h, w = rgb_frame.shape[:2]
                nx, ny = rgb_target.cx / w, 1.0 - (rgb_target.cy / h)
                rgb_warped = (nx, ny)
                self._rgb_canvas.update(True, nx, ny)
            else:
                self._rgb_canvas.update(False, 0, 0)

        # Build result snapshot
        self._result = TrackingResult(
            dvs_display=dvs_display,
            dvs_target=dvs_target,
            dvs_warped=dvs_warped,
            dvs_fps=dvs_fps,
            rgb_frame=rgb_frame,
            rgb_target=rgb_target,
            rgb_warped=rgb_warped,
        )

        # Forward to active output
        if self.active_output:
            self.active_output.process(self._result)

    def render(self) -> np.ndarray:
        if self.active_output:
            return self.active_output.render()
        # Fallback
        return np.zeros((480, 640, 3), dtype=np.uint8)

    # ------------------------------------------------------------------
    # Key handling
    # ------------------------------------------------------------------

    def handle_key(self, key: int) -> bool:
        if key == ord(' '):
            self.tracking_enabled = not self._tracking_enabled
            if self.active_output:
                self.active_output.on_tracking_changed(self._tracking_enabled)
            print(f"[TRACK] Tracking {'ON' if self._tracking_enabled else 'OFF'}")
            return True

        if key == ord('c'):
            self.clear_dvs_canvas()
            if self._rgb_canvas:
                self._rgb_canvas.clear()
            return True

        if key in (ord('d'), ord('D')):
            # Re-detect RGB quad
            if self._rgb_frame is not None and self._quad_detector is not None:
                quad = self._quad_detector.detect(self._rgb_frame)
                if quad:
                    self._store.set_rgb(quad)
                    if self._rgb_tracker:
                        self._rgb_tracker.roi = quad.as_xyxy()
                    print("[TRACK] RGB quad re-detected")
                else:
                    print("[TRACK] RGB quad detection failed")
            return True

        return super().handle_key(key)

    # ------------------------------------------------------------------
    # Mouse handling
    # ------------------------------------------------------------------

    def mouse_callback(self, event: int, x: int, y: int,
                       flags: int, param) -> None:
        """Forward left-button clicks to the active output mode."""
        import cv2
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.active_output:
            self.active_output.mouse_callback(x, y)

    # ------------------------------------------------------------------
    # Accessors for output modes
    # ------------------------------------------------------------------

    @property
    def dvs_reader(self):
        return self._dvs_reader

    @property
    def dvs_canvas(self):
        return self._dvs_canvas

    @property
    def rgb_canvas(self):
        return self._rgb_canvas

    def render_dvs_canvas(self) -> np.ndarray:
        """Thread-safe render of the persistent DVS canvas."""
        with self._dvs_canvas_lock:
            return self._dvs_canvas.render()

    def clear_dvs_canvas(self) -> None:
        """Thread-safe clear of the persistent DVS canvas."""
        if self._dvs_canvas is not None:
            with self._dvs_canvas_lock:
                self._dvs_canvas.clear()

    @property
    def result(self) -> TrackingResult:
        return self._result

    @property
    def tracking_enabled(self) -> bool:
        return self._tracking_enabled

    @tracking_enabled.setter
    def tracking_enabled(self, value: bool) -> None:
        self._tracking_enabled = value
        # Also controls DVSReaderThread's bridge push (PHYS_DVS relies on this)
        if self._dvs_reader:
            self._dvs_reader.tracking_enabled = value
