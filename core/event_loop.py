"""Main event loop with mouse-click tab switching."""

import cv2
import numpy as np
from typing import Dict, Optional

from app.core.demo import Demo, OutputModeType
from app.core.camera import CameraManager
from app.core.display import (
    render_tab_bar, tab_index_from_click, TAB_BAR_HEIGHT,
    render_mode_buttons, mode_button_from_click, mode_buttons_width,
    MODE_ORDER,
)

WINDOW_NAME = "Demo"


class MainLoop:
    """Top-level event loop managing tabs and display."""

    def __init__(self, camera_mgr: CameraManager, demos: Dict[str, Demo]):
        self._camera_mgr = camera_mgr
        self._demos = demos  # ordered dict: {"calibration": ..., "tracking": ...}
        self._demo_names = list(demos.keys())
        self._active_name = ""
        self._active_demo: Optional[Demo] = None
        self._running = False
        self._frame_width = 800  # updated dynamically after first render
        self._shown_modes = []   # mode buttons currently displayed

    def run(self) -> None:
        """Start the main loop (blocking)."""
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self._mouse_callback)

        # Start on first tab
        self._switch_demo(self._demo_names[0])
        self._running = True

        while self._running:
            self._active_demo.process_frame(self._camera_mgr)
            frame = self._active_demo.render()
            self._frame_width = frame.shape[1]

            # Determine mode buttons for active demo
            outputs = self._active_demo._outputs
            if outputs:
                self._shown_modes = MODE_ORDER
                available = set(outputs.keys())
                active_mode = self._active_demo._active_output_type
                btn_w = mode_buttons_width(len(self._shown_modes))
            else:
                self._shown_modes = []
                btn_w = 0

            # Compose tab bar + demo frame
            tabs = [(str(i + 1), name) for i, name in enumerate(self._demo_names)]
            tab_bar = render_tab_bar(tabs, self._active_name, frame.shape[1],
                                     reserved_right=btn_w)
            if self._shown_modes:
                mode_bar = render_mode_buttons(
                    self._shown_modes, active_mode, available, btn_w)
                tab_bar[:, frame.shape[1] - btn_w:] = mode_bar
            composed = np.vstack([tab_bar, frame])
            cv2.imshow(WINDOW_NAME, composed)

            key = cv2.waitKey(1) & 0xFF
            if key == 255:
                continue
            if not self._handle_key(key):
                self._active_demo.handle_key(key)

        # Cleanup
        if self._active_demo:
            self._active_demo.deactivate()
        self._camera_mgr.shutdown()
        cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # Key handling
    # ------------------------------------------------------------------

    def _handle_key(self, key: int) -> bool:
        """Handle global key events. Return True if consumed."""
        if key == ord('q'):
            self._running = False
            return True
        # Output mode switching (only meaningful for tracking demo)
        if key == ord('g'):
            self._active_demo.switch_output(OutputModeType.GUI)
            return True
        if key == ord('e'):
            self._active_demo.switch_output(OutputModeType.PHYS_DVS)
            return True
        if key == ord('r'):
            self._active_demo.switch_output(OutputModeType.PHYS_RGB)
            return True
        return False

    # ------------------------------------------------------------------
    # Mouse handling (tab clicks)
    # ------------------------------------------------------------------

    def _mouse_callback(self, event: int, x: int, y: int,
                        flags: int, param) -> None:
        """Handle mouse events for tab bar clicks."""
        if event != cv2.EVENT_LBUTTONDOWN:
            # Also forward mouse events to the active demo (for calibration
            # corner dragging etc.) with y offset adjusted for tab bar.
            if self._active_demo and hasattr(self._active_demo, 'mouse_callback'):
                self._active_demo.mouse_callback(
                    event, x, y - TAB_BAR_HEIGHT, flags, param)
            return

        # Check if click is in tab bar area
        btn_w = mode_buttons_width(len(self._shown_modes))
        idx = tab_index_from_click(x, y, len(self._demo_names),
                                   self._frame_width, reserved_right=btn_w)
        if idx is not None:
            self._switch_demo(self._demo_names[idx])
            return

        # Check mode button click
        if self._shown_modes:
            mode = mode_button_from_click(
                x, y, self._shown_modes, self._frame_width)
            if mode is not None:
                if mode in self._active_demo._outputs:
                    self._active_demo.switch_output(mode)
                return

        # Click in demo area — forward to demo with adjusted y
        if self._active_demo and hasattr(self._active_demo, 'mouse_callback'):
            self._active_demo.mouse_callback(
                event, x, y - TAB_BAR_HEIGHT, flags, param)

    # ------------------------------------------------------------------
    # Tab switching
    # ------------------------------------------------------------------

    def _switch_demo(self, name: str) -> None:
        """Switch to a different demo tab."""
        if name == self._active_name:
            return
        if self._active_demo:
            self._active_demo.deactivate()
        self._active_demo = self._demos[name]
        self._active_name = name
        self._active_demo.activate(self._camera_mgr)
