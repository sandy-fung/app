"""Physical DVS output — DVS tracking drives the robotic arm.

When active, DVS tracking results are pushed to CommandBridge at ~200fps
via DVSDrawingThread (replaces DVSReaderThread).
Display shows single DVS canvas + arm status.
"""

import cv2
import numpy as np

from app.config import DVS_WIDTH, DVS_HEIGHT, DVS_SCALE, CANVAS_SIZE, IDLE_CLEAR
from app.core.demo import OutputMode
from app.core.display import draw_hint_bar, draw_paused_overlay


class TrackingPhysDVSOutput(OutputMode):
    """DVS tracking → arm drawing via CommandBridge."""

    def __init__(self, tracking_demo, bridge, arm_thread):
        self._demo = tracking_demo
        self._bridge = bridge
        self._arm = arm_thread
        self._result = None
        self._drawing_thread = None

    def activate(self) -> None:
        """Replace DVSReaderThread with DVSDrawingThread for 200fps bridge push."""
        from main_dvs_drawing import DVSDrawingThread
        from dvs_laser_tracker import DVSLaserTracker

        # Stop the existing reader thread
        if self._demo.dvs_reader:
            self._demo.dvs_reader.stop()

        # Create fresh tracker and start drawing thread
        tracker = DVSLaserTracker(
            width=DVS_WIDTH, height=DVS_HEIGHT,
            noise_mask_path=self._demo._args.noise_mask,
        )
        # xe_cam is stored by TrackingDemo during activate()
        self._drawing_thread = DVSDrawingThread(
            self._demo._xe_cam, tracker,
            self._demo._store.dvs_homography,
            self._bridge,
            scale=DVS_SCALE,
            canvas_size=CANVAS_SIZE,
            idle_clear=IDLE_CLEAR,
            write_confirm=3,
        )
        self._drawing_thread.start()
        # Start paused for safety — user must press Space to begin
        self._drawing_thread.tracking_enabled = False
        self._demo.tracking_enabled = False
        print("[PHYS_DVS] Activated — PAUSED (press Space to start tracking)")

    def deactivate(self) -> None:
        """Stop DVSDrawingThread and return arm to center."""
        if self._drawing_thread:
            self._drawing_thread.stop()
            self._drawing_thread = None
        # Drain residual high-freq commands, then go center (pen up)
        self._bridge.clear()
        self._bridge.put(False, 0.5, 0.5)
        print("[PHYS_DVS] Deactivated — arm returning to center")

    def on_tracking_changed(self, enabled: bool) -> None:
        if self._drawing_thread:
            self._drawing_thread.tracking_enabled = enabled

    def process(self, result) -> None:
        self._result = result

    def render(self) -> np.ndarray:
        # Single DVS canvas + arm status
        if self._drawing_thread:
            canvas = self._drawing_thread.render_canvas()
        elif self._demo.dvs_reader:
            canvas = self._demo.dvs_reader.render_canvas()
        else:
            canvas = np.zeros((400, 400, 3), dtype=np.uint8)

        # Arm status hint bar
        canvas = canvas.copy()
        if self._arm:
            pending = self._bridge.pending if self._bridge else 0
            move_c = self._arm.move_count
            fail_c = self._arm.fail_count
            ready = self._arm.is_ready.is_set() if self._arm.is_ready else False
            status = (f"ARM: {'RDY' if ready else 'INIT'} | "
                      f"moves: {move_c} | fails: {fail_c} | queue: {pending}")
            color = (0, 200, 0) if ready else (0, 140, 255)
            hints = [(status, color)]
            if self._arm.error:
                hints.insert(0, (f"ERR: {self._arm.error}", (0, 0, 255)))
            draw_hint_bar(canvas, hints)

        cv2.putText(canvas, "Physical DVS", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 1)

        if not self._demo.tracking_enabled:
            draw_paused_overlay(canvas)

        return canvas
