"""Physical RGB output — RGB tracking drives the robotic arm.

RGB tracking results are pushed to CommandBridge at main-loop rate (~30fps).
Display shows RGB canvas (+ optional DVS canvas side-by-side).
"""

import cv2
import numpy as np

from app.core.demo import OutputMode
from app.core.display import (
    draw_active_border,
    draw_hint_bar,
    draw_paused_overlay,
    draw_view_toggle,
    view_toggle_from_click,
)


class TrackingPhysRGBOutput(OutputMode):
    """RGB tracking → arm drawing via CommandBridge."""

    def __init__(self, tracking_demo, bridge, arm_thread):
        self._demo = tracking_demo
        self._bridge = bridge
        self._arm = arm_thread
        self._result = None
        self._dual = True  # default: dual canvas view
        self._frame_w = 0  # cached rendered frame width

    def activate(self) -> None:
        # Start paused for safety — user must press Space to begin
        self._demo.tracking_enabled = False
        self._bridge.put(False, 0.5, 0.5)
        print("[PHYS_RGB] Activated — PAUSED (press Space to start tracking)")

    def deactivate(self) -> None:
        """Return arm to center on mode exit."""
        self._bridge.clear()
        print("[PHYS_RGB] Deactivated — arm returning to center")

    def on_tracking_changed(self, enabled: bool) -> None:
        if not enabled:
            self._bridge.clear()

    def process(self, result) -> None:
        self._result = result
        if not self._demo.tracking_enabled:
            return
        # Push RGB warped coordinates to arm bridge
        if result.rgb_warped:
            nx, ny = result.rgb_warped
            # Undo 90° CW frame rotation for arm coords (coupled to RGB_DISPLAY_ROTATE=90)
            arm_nx, arm_ny = 1 - ny, nx
            self._bridge.put(True, arm_nx, arm_ny)
        else:
            self._bridge.put(False, 0, 0)

    def render(self) -> np.ndarray:
        rgb = self._rgb_canvas()

        if self._dual:
            # Side-by-side: DVS + RGB (active, orange border)
            dvs = self._demo.render_dvs_canvas()
            rgb = rgb.copy()
            draw_active_border(rgb)
            # Match heights before hstack
            if dvs.shape[0] != rgb.shape[0]:
                dvs = cv2.resize(dvs, (dvs.shape[1], rgb.shape[0]),
                                 interpolation=cv2.INTER_LINEAR)
            canvas = np.hstack([dvs, rgb])
        else:
            canvas = rgb.copy()

        # Arm status hint bar
        self._draw_arm_hints(canvas)

        cv2.putText(canvas, "Physical RGB", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 1)

        if not self._demo.tracking_enabled:
            draw_paused_overlay(canvas)

        self._frame_w = canvas.shape[1]
        draw_view_toggle(canvas, self._dual)
        return canvas

    def mouse_callback(self, x: int, y: int) -> bool:
        """Handle toggle button click."""
        result = view_toggle_from_click(x, y, self._frame_w, self._dual)
        if result is not None:
            self._dual = result
            return True
        return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _rgb_canvas(self) -> np.ndarray:
        """Get RGB canvas frame."""
        if self._demo.rgb_canvas:
            return self._demo.rgb_canvas.render()
        return np.zeros((400, 400, 3), dtype=np.uint8)

    def _draw_arm_hints(self, canvas: np.ndarray) -> None:
        """Draw arm status hint bar on *canvas*."""
        if not self._arm:
            return
        pending = self._bridge.pending if self._bridge else 0
        move_c = self._arm.move_count
        fail_c = self._arm.fail_count
        ready = self._arm.is_ready.is_set() if self._arm.is_ready else False
        status = f"ARM: {'RDY' if ready else 'INIT'} | " \
                 f"moves: {move_c} | fails: {fail_c} | queue: {pending}"
        color = (0, 200, 0) if ready else (0, 140, 255)
        hints = [(status, color)]
        if self._arm.error:
            hints.insert(0, (f"ERR: {self._arm.error}", (0, 0, 255)))
        draw_hint_bar(canvas, hints)
