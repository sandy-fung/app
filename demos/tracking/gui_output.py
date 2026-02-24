"""GUI output mode — dual DVS + RGB side-by-side display.

Shows camera preview + trajectory canvas for both DVS and RGB.
Supports three layouts: FULL, TRAJECTORY, PIP.
"""

import cv2
import numpy as np

from app.core.demo import OutputMode
from app.core.display import (
    compose_full, compose_trajectory, compose_pip, draw_hint_bar, draw_status_on,
    LAYOUT_FULL, LAYOUT_TRAJECTORY, LAYOUT_PIP, LAYOUT_NAMES,
)


class TrackingGUIOutput(OutputMode):
    """Dual canvas display for GUI-only tracking."""

    def __init__(self, tracking_demo):
        self._demo = tracking_demo
        self._layout = LAYOUT_FULL
        self._result = None

    def activate(self) -> None:
        pass

    def deactivate(self) -> None:
        pass

    def process(self, result) -> None:
        self._result = result

    def render(self) -> np.ndarray:
        result = self._result
        if result is None:
            return np.zeros((480, 800, 3), dtype=np.uint8)

        # DVS display with status overlay
        dvs_display = result.dvs_display
        if dvs_display is None:
            dvs_display = np.zeros((480, 492, 3), dtype=np.uint8)
            cv2.putText(dvs_display, "DVS: waiting...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)
        dvs_display = dvs_display.copy()
        draw_status_on(dvs_display, "DVS",
                       self._demo.tracking_enabled,
                       result.dvs_fps,
                       result.dvs_warped)

        # RGB display with status overlay
        rgb_display = result.rgb_frame
        if rgb_display is None:
            rgb_display = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(rgb_display, "RGB: waiting...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)
        else:
            rgb_display = rgb_display.copy()
            # Draw target marker on RGB frame
            if result.rgb_target:
                from main_laser_drawing import draw_target, draw_quad
                draw_target(rgb_display, result.rgb_target)
                if self._demo._store.rgb_quad:
                    draw_quad(rgb_display, self._demo._store.rgb_quad)
        draw_status_on(rgb_display, "RGB",
                       result.rgb_target is not None,
                       30.0,  # RGB runs at main loop rate
                       result.rgb_warped)

        # Canvas images
        dvs_canvas = self._demo.dvs_reader.render_canvas()
        rgb_canvas = self._demo.rgb_canvas.render()

        # Compose based on layout
        if self._layout == LAYOUT_FULL:
            composed = compose_full(dvs_display, rgb_display,
                                    dvs_canvas, rgb_canvas)
        elif self._layout == LAYOUT_TRAJECTORY:
            composed = compose_trajectory(dvs_canvas, rgb_canvas)
        elif self._layout == LAYOUT_PIP:
            composed = compose_pip(dvs_display, rgb_display,
                                   dvs_canvas, rgb_canvas)
        else:
            composed = compose_full(dvs_display, rgb_display,
                                    dvs_canvas, rgb_canvas)

        # Hint bar
        draw_hint_bar(composed, [
            f"[v] layout: {LAYOUT_NAMES[self._layout]}  "
            f"[space] track  [c] clear  [d] re-detect",
        ])

        return composed

    def handle_key(self, key: int) -> bool:
        if key == ord('v'):
            self._layout = (self._layout + 1) % 3
            print(f"[GUI] Layout: {LAYOUT_NAMES[self._layout]}")
            return True
        return False
