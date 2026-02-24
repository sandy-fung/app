"""Physical RGB output — RGB tracking drives the robotic arm.

RGB tracking results are pushed to CommandBridge at main-loop rate (~30fps).
Display shows single RGB canvas + arm status.
"""

import cv2
import numpy as np

from app.core.demo import OutputMode
from app.core.display import draw_hint_bar


class TrackingPhysRGBOutput(OutputMode):
    """RGB tracking → arm drawing via CommandBridge."""

    def __init__(self, tracking_demo, bridge, arm_thread):
        self._demo = tracking_demo
        self._bridge = bridge
        self._arm = arm_thread
        self._result = None

    def activate(self) -> None:
        # Re-center arm at workspace midpoint before tracking starts
        self._bridge.put(False, 0.5, 0.5)
        print("[PHYS_RGB] Activated — arm follows RGB tracking")

    def deactivate(self) -> None:
        self._bridge.put_safe_home()
        print("[PHYS_RGB] Deactivated")

    def process(self, result) -> None:
        self._result = result
        # Push RGB warped coordinates to arm bridge
        if result.rgb_warped:
            nx, ny = result.rgb_warped
            self._bridge.put(True, nx, ny)
        else:
            self._bridge.put(False, 0, 0)

    def render(self) -> np.ndarray:
        # Single RGB canvas + arm status
        if self._demo.rgb_canvas:
            canvas = self._demo.rgb_canvas.render()
        else:
            canvas = np.zeros((400, 400, 3), dtype=np.uint8)

        canvas = canvas.copy()

        # Arm status hint bar
        if self._arm:
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

        cv2.putText(canvas, "Physical RGB", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 1)

        return canvas
