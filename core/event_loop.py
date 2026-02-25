"""Main event loop with mouse-click tab switching."""

import cv2
import numpy as np
from typing import Dict, Optional

from app.core.demo import Demo, OutputModeType
from app.core.camera import CameraManager
from app.core.display import (
    render_tab_bar, tab_index_from_click, TAB_BAR_HEIGHT,
    render_mode_buttons, mode_button_from_click, mode_buttons_width,
    render_arm_buttons, arm_button_from_click, arm_buttons_width,
    MODE_ORDER,
)

WINDOW_NAME = "Demo"


class MainLoop:
    """Top-level event loop managing tabs and display."""

    def __init__(self, camera_mgr: CameraManager, demos: Dict[str, Demo],
                 bridge=None, arm_thread=None):
        self._camera_mgr = camera_mgr
        self._demos = demos  # ordered dict: {"calibration": ..., "tracking": ...}
        self._demo_names = list(demos.keys())
        self._active_name = ""
        self._active_demo: Optional[Demo] = None
        self._running = False
        self._frame_width = 800  # updated dynamically after first render
        self._shown_modes = []   # mode buttons currently displayed
        self._bridge = bridge
        self._arm_thread = arm_thread

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
                mode_w = mode_buttons_width(len(self._shown_modes))
            else:
                self._shown_modes = []
                mode_w = 0

            # Arm buttons (right-most, only when bridge exists)
            arm_w = arm_buttons_width() if self._bridge else 0
            reserved_right = mode_w + arm_w

            # Compose tab bar + demo frame
            tabs = [(str(i + 1), name) for i, name in enumerate(self._demo_names)]
            tab_bar = render_tab_bar(tabs, self._active_name, frame.shape[1],
                                     reserved_right=reserved_right)
            if self._shown_modes:
                mode_bar = render_mode_buttons(
                    self._shown_modes, active_mode, available, mode_w)
                tab_bar[:, frame.shape[1] - reserved_right:
                        frame.shape[1] - arm_w] = mode_bar
            if arm_w > 0:
                at_home = (self._arm_thread.at_home
                           if self._arm_thread else True)
                arm_bar = render_arm_buttons(at_home, arm_w)
                tab_bar[:, frame.shape[1] - arm_w:] = arm_bar
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
        # Arm control (global — works from any tab)
        if key == ord('h') and self._bridge:
            self._bridge.put_safe_home()
            return True
        if key == ord('w') and self._bridge:
            self._bridge.put(False, 0.5, 0.5)
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
        arm_w = arm_buttons_width() if self._bridge else 0
        mode_w = mode_buttons_width(len(self._shown_modes))
        reserved_right = mode_w + arm_w

        idx = tab_index_from_click(x, y, len(self._demo_names),
                                   self._frame_width, reserved_right=reserved_right)
        if idx is not None:
            self._switch_demo(self._demo_names[idx])
            return

        # Check arm button click (right-most area)
        if arm_w > 0:
            btn = arm_button_from_click(x, y, self._frame_width, arm_w)
            if btn == "HOME":
                self._bridge.put_safe_home()
                return
            if btn == "DRAW":
                self._bridge.put(False, 0.5, 0.5)
                return

        # Check mode button click (between tabs and arm buttons)
        if self._shown_modes:
            # Shift frame_width so mode_button_from_click sees its area
            mode = mode_button_from_click(
                x, y, self._shown_modes, self._frame_width - arm_w)
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
