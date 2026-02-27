"""Main event loop with mouse-click tab switching."""

import cv2
import numpy as np
from typing import Dict, Optional

from app.core.demo import Demo, OutputModeType
from app.core.camera import CameraManager
from app.config import DISPLAY_W, DISPLAY_H
from app.core.display import (
    render_tab_bar, tab_index_from_click, TAB_BAR_HEIGHT,
    render_mode_row, mode_row_click, MODE_ROW_HEIGHT,
    render_arm_buttons, arm_button_from_click, arm_buttons_width,
    MODE_ORDER, normalize_frame,
)
from app.core.memory_monitor import MemoryMonitor

WINDOW_NAME = "Demo"

# Fixed content area height (total window minus UI chrome)
_CONTENT_H = DISPLAY_H - TAB_BAR_HEIGHT - MODE_ROW_HEIGHT  # 700


def _draw_mem_bar(frame, rss_mb, peak_mb, warning=False):
    """Draw a small memory-usage bar at the bottom-right of frame."""
    if peak_mb <= 0:
        return
    BAR_W, BAR_H, MARGIN = 120, 14, 8
    h, w = frame.shape[:2]
    x1 = w - BAR_W - MARGIN
    y1 = h - BAR_H - MARGIN
    x2, y2 = x1 + BAR_W, y1 + BAR_H

    # Gray border frame
    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 80), 1)

    # Filled portion (proportional to peak)
    ratio = min(rss_mb / peak_mb, 1.0)
    fill_w = int((BAR_W - 2) * ratio)
    color = (0, 140, 255) if warning else (100, 100, 100)  # orange vs gray
    cv2.rectangle(frame, (x1 + 1, y1 + 1),
                  (x1 + 1 + fill_w, y2 - 1), color, -1)

    # Text label
    label = f"{rss_mb:.0f}M"
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, label, (x1 + 4, y2 - 3),
                font, 0.35, (255, 255, 255), 1, cv2.LINE_AA)


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
        self._frame_width = DISPLAY_W
        self._shown_modes = []   # mode buttons currently displayed
        self._mode_row_h = MODE_ROW_HEIGHT  # always rendered
        self._bridge = bridge
        self._arm_thread = arm_thread
        self._mem_monitor = MemoryMonitor()
        self._content_scale = 1.0
        self._content_pad_x = 0
        self._content_pad_y = 0

    @property
    def _pen_down(self) -> bool:
        """Query actual pen state from arm thread."""
        if self._arm_thread is None:
            return False
        return self._arm_thread.pen_down

    def run(self) -> None:
        """Start the main loop (blocking)."""
        cv2.namedWindow(WINDOW_NAME,
                        cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, DISPLAY_W, DISPLAY_H)
        cv2.setMouseCallback(WINDOW_NAME, self._mouse_callback)

        # Start on first tab
        self._switch_demo(self._demo_names[0])
        self._running = True

        while self._running:
            self._active_demo.process_frame(self._camera_mgr)
            frame = self._active_demo.render()
            frame, self._content_scale, self._content_pad_x, \
                self._content_pad_y = normalize_frame(
                    frame, DISPLAY_W, _CONTENT_H)

            # Determine mode buttons for active demo
            outputs = self._active_demo._outputs
            if outputs:
                self._shown_modes = MODE_ORDER
                available = set(outputs.keys())
                active_mode = self._active_demo._active_output_type
            else:
                self._shown_modes = []

            # Arm buttons (right-most on tab bar, only when bridge exists)
            arm_w = arm_buttons_width() if self._bridge else 0
            reserved_right = arm_w

            # Compose tab bar (width always DISPLAY_W)
            tabs = [(str(i + 1), name) for i, name in enumerate(self._demo_names)]
            tab_bar = render_tab_bar(tabs, self._active_name, DISPLAY_W,
                                     reserved_right=reserved_right)
            if arm_w > 0:
                at_home = (self._arm_thread.at_home
                           if self._arm_thread else True)
                arm_bar = render_arm_buttons(at_home, arm_w,
                                             pen_down=self._pen_down)
                tab_bar[:, DISPLAY_W - arm_w:] = arm_bar

            # Mode row — always rendered (empty gray bar when no modes)
            if self._shown_modes:
                mode_row = render_mode_row(
                    self._shown_modes, active_mode, available, DISPLAY_W)
            else:
                mode_row = render_mode_row([], None, set(), DISPLAY_W)
            self._mode_row_h = mode_row.shape[0]  # always MODE_ROW_HEIGHT
            composed = np.vstack([tab_bar, mode_row, frame])

            self._mem_monitor.tick()
            _draw_mem_bar(composed, self._mem_monitor.rss_mb,
                          self._mem_monitor.peak_mb, self._mem_monitor.warning)

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
        # Pen control (global — works from any tab, only when not at home)
        if key == ord('p') and self._bridge and self._arm_thread \
                and not self._arm_thread.at_home:
            self._bridge.put_pen_down()
            return True
        if key == ord('u') and self._bridge and self._arm_thread \
                and not self._arm_thread.at_home:
            self._bridge.put_pen_up()
            return True
        return False

    # ------------------------------------------------------------------
    # Mouse handling (tab clicks)
    # ------------------------------------------------------------------

    def _to_demo_coords(self, x: int, y: int, header_h: int):
        """Reverse normalize_frame transform for mouse coordinates."""
        dx = x - self._content_pad_x
        dy = (y - header_h) - self._content_pad_y
        if self._content_scale > 0:
            dx = int(dx / self._content_scale)
            dy = int(dy / self._content_scale)
        return dx, dy

    def _mouse_callback(self, event: int, x: int, y: int,
                        flags: int, param) -> None:
        """Handle mouse events for tab bar and mode row clicks."""
        header_h = TAB_BAR_HEIGHT + self._mode_row_h

        if event != cv2.EVENT_LBUTTONDOWN:
            # Forward non-click events to demo with reverse-mapped coords
            if self._active_demo and hasattr(self._active_demo, 'mouse_callback'):
                dx, dy = self._to_demo_coords(x, y, header_h)
                self._active_demo.mouse_callback(
                    event, dx, dy, flags, param)
            return

        # --- Region 1: Tab bar (0 ~ TAB_BAR_HEIGHT) ---
        if y < TAB_BAR_HEIGHT:
            arm_w = arm_buttons_width() if self._bridge else 0

            # Arm button click (right side of tab bar)
            if arm_w > 0:
                btn = arm_button_from_click(x, y, self._frame_width, arm_w)
                if btn == "HOME":
                    self._bridge.put_safe_home()
                    return
                if btn == "DRAW":
                    self._bridge.put(False, 0.5, 0.5)
                    return
                at_home = (self._arm_thread.at_home
                           if self._arm_thread else True)
                if btn == "PEN v" and not at_home:
                    self._bridge.put_pen_down()
                    return
                if btn == "PEN ^" and not at_home:
                    self._bridge.put_pen_up()
                    return

            # Tab click
            reserved_right = arm_w
            idx = tab_index_from_click(x, y, len(self._demo_names),
                                       self._frame_width,
                                       reserved_right=reserved_right)
            if idx is not None:
                self._switch_demo(self._demo_names[idx])
            return

        # --- Region 2: Mode row (TAB_BAR_HEIGHT ~ TAB_BAR_HEIGHT + mode_h) ---
        if self._mode_row_h > 0 and y < header_h:
            mode = mode_row_click(
                x, y - TAB_BAR_HEIGHT, self._shown_modes)
            if mode is not None and self._active_demo._outputs:
                if mode in self._active_demo._outputs:
                    self._active_demo.switch_output(mode)
            return

        # --- Region 3: Demo content ---
        if self._active_demo and hasattr(self._active_demo, 'mouse_callback'):
            dx, dy = self._to_demo_coords(x, y, header_h)
            self._active_demo.mouse_callback(
                event, dx, dy, flags, param)

    # ------------------------------------------------------------------
    # Tab switching
    # ------------------------------------------------------------------

    def _switch_demo(self, name: str) -> None:
        """Switch to a different demo tab."""
        if name == self._active_name:
            return
        if self._active_demo:
            self._active_demo.deactivate()
        self._mem_monitor.collect()
        self._active_demo = self._demos[name]
        self._active_name = name
        self._active_demo.activate(self._camera_mgr)
