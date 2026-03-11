"""Calibration tab — DVS quad calibration + RGB quad detection.

Embeds the same drag-corner logic from quad_calibrator.py into the
unified MainLoop (no separate window).

Sub-tabs: Page Calibration (default) and Arm Calibration (when arm present).
"""

import threading
from typing import Optional

import cv2
import numpy as np

from cv2_like_xe_sdk import dvs_normalize

from app.config import DVS_WIDTH, DVS_HEIGHT, DVS_SCALE
from app.core.demo import Demo
from app.core.display import (
    draw_hint_bar, draw_hint_buttons, hint_button_from_click,
    resize_to_height,
    render_sub_tab_bar, sub_tab_from_click, SUB_TAB_BAR_HEIGHT,
)

# Sub-tab definitions
_SUB_TABS = [("page", "Page Cal"), ("arm", "Arm Cal")]

# DVS config mode toggle buttons
_DVS_CFG_BTNS = [("dvs_only", "DVS Only"), ("hybrid", "Hybrid")]


class CalibrationDemo(Demo):
    """Side-by-side DVS corner calibration + RGB quad detection."""

    def __init__(self, cal_store, args, bridge=None, arm_thread=None):
        super().__init__("Calibration")
        self._store = cal_store
        self._args = args
        # DVS corner state
        self._dvs_corners: Optional[np.ndarray] = None
        self._dragging_idx: Optional[int] = None
        self._hit_radius = 15  # px in display space
        # RGB quad state
        self._rgb_quad = None
        self._quad_detector = None
        self._rgb_dragging_idx: Optional[int] = None
        self._rgb_panel_offset_x: int = 0
        self._rgb_scale: float = 1.0
        self._rgb_cam_size: tuple = (640, 480)
        # Frames
        self._camera_mgr = None
        self._dvs_panel_w = DVS_WIDTH * DVS_SCALE
        self._dvs_panel_h = DVS_HEIGHT * DVS_SCALE

        # Sub-mode: "page" or "arm"
        self._has_arm = bridge is not None and arm_thread is not None
        self._sub_mode = "page"
        self._arm_panel = None  # lazy ArmCalibrationPanel
        self._bridge = bridge
        self._arm_thread = arm_thread
        self._content_w = 0  # updated each render()
        self._dvs_config_mode = "dvs_only"  # "dvs_only" or "hybrid"
        self._hint_bar_h = 50  # matches draw_hint_bar default
        self._page_w = 0  # actual composed width, updated each render

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def activate(self, camera_mgr) -> None:
        self._camera_mgr = camera_mgr
        if self._dvs_config_mode == "hybrid":
            camera_mgr.switch_dvs_to_hybrid()
        else:
            camera_mgr.switch_dvs_to_tracking()

        # Lazy import (needs sys.path setup)
        from quad_calibrator import default_corners
        from quad_detector import QuadDetector

        # Load saved DVS calibration or use defaults
        if self._store.dvs_corners is not None:
            self._dvs_corners = self._store.dvs_corners.copy()
        elif self._store.load_dvs(self._args.dvs_cal):
            self._dvs_corners = self._store.dvs_corners.copy()
        else:
            self._dvs_corners = default_corners()

        # RGB quad detector
        if self._quad_detector is None:
            self._quad_detector = QuadDetector()

        # Load saved RGB calibration
        if self._store.rgb_quad is None:
            self._store.load_rgb(self._args.rgb_cal)
        self._rgb_quad = self._store.rgb_quad  # may still be None

        # Try initial RGB quad detection if no saved calibration
        if self._rgb_quad is None:
            rgb_frame = camera_mgr.read_rgb_frame()
            if rgb_frame is not None:
                self._rgb_quad = self._quad_detector.detect(rgb_frame)
                if self._rgb_quad:
                    self._store.set_rgb(self._rgb_quad)

    def deactivate(self) -> None:
        # Auto-save DVS calibration
        if self._dvs_corners is not None:
            self._store.set_dvs(self._dvs_corners)
            if self._store.save_dvs(self._args.dvs_cal):
                print("[CAL] DVS corners saved")
            else:
                print("[CAL] DVS save failed")
        # Auto-save RGB calibration
        if self._rgb_quad:
            self._store.set_rgb(self._rgb_quad)
            if self._store.save_rgb(self._args.rgb_cal):
                print("[CAL] RGB quad saved")
            else:
                print("[CAL] RGB save failed")
        if self._camera_mgr:
            self._camera_mgr.switch_dvs_to_tracking()

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------

    def process_frame(self, camera_mgr) -> None:
        # Nothing heavy to do; frames are read in render() for simplicity
        pass

    def _grab_gray_safe(self, timeout: float = 2.0,
                        max_retries: int = 3) -> Optional[np.ndarray]:
        """Grab a gray frame with timeout to avoid XeGetFrame() hangs.

        Runs grab_gray_frame() in a daemon thread so the main thread is
        never blocked indefinitely after a DVS mode switch.
        """
        from quad_calibrator import grab_gray_frame

        xe_cam = self._camera_mgr.xe_cam
        for attempt in range(max_retries):
            result = [None]

            def _grab():
                result[0] = grab_gray_frame(xe_cam)

            t = threading.Thread(target=_grab, daemon=True)
            t.start()
            t.join(timeout=timeout)
            if t.is_alive():
                print(f"[CAL] WARNING: grab_gray_frame hung "
                      f"(attempt {attempt + 1}/{max_retries})")
                continue
            return result[0]

        print("[CAL] ERROR: grab_gray_frame failed after "
              f"{max_retries} retries, returning black frame")
        return None

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self) -> np.ndarray:
        if self._has_arm and self._sub_mode == "arm":
            content = self._render_arm()
        else:
            content = self._render_page()

        available = self._available_sub_tabs()
        tab_bar = render_sub_tab_bar(
            _SUB_TABS, self._sub_mode, content.shape[1], available=available,
        )
        content = np.vstack([tab_bar, content])

        self._content_w = content.shape[1]
        return content

    def _available_sub_tabs(self) -> set:
        """Return the set of enabled sub-tab keys."""
        avail = {"page"}
        if self._has_arm:
            avail.add("arm")
        return avail

    def _render_page(self) -> np.ndarray:
        """Original page calibration render."""
        from quad_calibrator import draw_overlay
        from main_laser_drawing import draw_quad

        scale = DVS_SCALE

        # --- DVS panel ---
        if self._dvs_config_mode == "hybrid":
            gray = self._grab_gray_safe()
            if gray is None:
                gray = np.zeros((DVS_HEIGHT, DVS_WIDTH), dtype=np.uint8)
            bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            # dvs_only — render like gesture display (full dynamic range)
            xe = self._camera_mgr.xe_cam
            dvs_raw, _ = xe.g_cap.XeGetFrame(
                xe.g_xereal_mode, xe.g_xereal_bit_depth,
            )
            if dvs_raw is not None:
                raw_2d = dvs_raw.reshape((DVS_HEIGHT, DVS_WIDTH))
                gray = dvs_normalize(raw_2d, xe.g_xereal_bit_depth)
            else:
                gray = None
            if gray is None:
                gray = np.zeros((DVS_HEIGHT, DVS_WIDTH), dtype=np.uint8)
            bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        dvs_panel = cv2.resize(
            bgr, (self._dvs_panel_w, self._dvs_panel_h),
            interpolation=cv2.INTER_NEAREST,
        )
        if self._dvs_corners is not None:
            draw_overlay(dvs_panel, self._dvs_corners, scale, self._dragging_idx)
        cv2.putText(dvs_panel, "DVS", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # --- RGB panel ---
        rgb_frame = self._camera_mgr.read_rgb_frame()
        if rgb_frame is not None:
            self._rgb_cam_size = (rgb_frame.shape[1], rgb_frame.shape[0])
            self._rgb_scale = self._dvs_panel_h / rgb_frame.shape[0]
            rgb_display = rgb_frame.copy()
            if self._rgb_quad:
                draw_quad(rgb_display, self._rgb_quad,
                          active_idx=self._rgb_dragging_idx)
            rgb_panel = resize_to_height(rgb_display, self._dvs_panel_h)
        else:
            rgb_panel = np.zeros((self._dvs_panel_h, self._dvs_panel_w, 3),
                                dtype=np.uint8)
            cv2.putText(rgb_panel, "RGB: no frame", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        cv2.putText(rgb_panel, "RGB", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # --- Compose side-by-side ---
        sep = np.zeros((self._dvs_panel_h, 2, 3), dtype=np.uint8)
        self._rgb_panel_offset_x = self._dvs_panel_w + 2
        composed = np.hstack([dvs_panel, sep, rgb_panel])
        self._page_w = composed.shape[1]

        # Hint bar at bottom
        dvs_ok = "OK" if self._store.dvs_calibrated else "--"
        rgb_ok = "OK" if self._store.rgb_calibrated else "--"
        hints = [
            f"DVS: {dvs_ok}  |  RGB: {rgb_ok}",
            "[R] reset DVS  [D] re-detect RGB  (auto-save on exit)",
        ]
        if self._has_arm:
            hints.append("[Tab] switch to Arm Cal")
        self._hint_bar_h = draw_hint_bar(composed, hints)
        draw_hint_buttons(composed, _DVS_CFG_BTNS, self._dvs_config_mode,
                          bar_h=self._hint_bar_h)

        return composed

    def _render_arm(self) -> np.ndarray:
        """Render the arm calibration sub-panel."""
        if self._arm_panel is None:
            from app.demos.calibration.arm_panel import ArmCalibrationPanel
            self._arm_panel = ArmCalibrationPanel(
                self._bridge, self._arm_thread,
            )

        w = self._dvs_panel_w * 2 + 2  # match page width
        h = self._dvs_panel_h
        return self._arm_panel.render(w, h)

    # ------------------------------------------------------------------
    # Key handling
    # ------------------------------------------------------------------

    def handle_key(self, key: int) -> bool:
        # Tab toggles sub-mode when arm is available
        if self._has_arm and key == 9:  # Tab
            self._sub_mode = "arm" if self._sub_mode == "page" else "page"
            return True

        # Delegate to active sub-mode
        if self._has_arm and self._sub_mode == "arm":
            if self._arm_panel is not None and self._arm_panel.handle_key(key):
                return True
            return super().handle_key(key)

        # Page mode — original logic
        if key in (ord('r'), ord('R')):  # Reset DVS corners
            from quad_calibrator import default_corners
            self._dvs_corners = default_corners()
            print("[CAL] DVS corners reset to default")
            return True

        if key in (ord('d'), ord('D')):  # Re-detect RGB quad
            rgb_frame = self._camera_mgr.read_rgb_frame()
            if rgb_frame is not None and self._quad_detector is not None:
                quad = self._quad_detector.detect(rgb_frame)
                if quad:
                    self._rgb_quad = quad
                    self._store.set_rgb(quad)
                    print("[CAL] RGB quad re-detected")
                else:
                    print("[CAL] RGB quad detection failed")
            return True

        return super().handle_key(key)

    # ------------------------------------------------------------------
    # Mouse handling (corner dragging)
    # ------------------------------------------------------------------

    def mouse_callback(self, event: int, x: int, y: int,
                       flags: int, param) -> None:
        """Handle mouse events for DVS / RGB corner dragging.

        Note: x, y are already adjusted for tab bar offset by MainLoop.
        DVS panel (left) and RGB panel (right) both accept drag events.
        """
        # Sub-tab bar click detection (always shown)
        if event == cv2.EVENT_LBUTTONDOWN:
            content_w = self._content_w or (self._dvs_panel_w * 2 + 2)
            available = self._available_sub_tabs()
            clicked = sub_tab_from_click(
                x, y, _SUB_TABS, content_w, available=available,
            )
            if clicked is not None:
                self._sub_mode = clicked
                return

        # Offset y for sub-tab bar
        y -= SUB_TAB_BAR_HEIGHT
        if y < 0:
            return

        # Delegate to arm panel
        if self._has_arm and self._sub_mode == "arm":
            if self._arm_panel is not None:
                self._arm_panel.mouse_callback(event, x, y, flags, param)
            return

        # Page mode — check hint bar config buttons first
        if event == cv2.EVENT_LBUTTONDOWN:
            page_w = self._page_w or (self._dvs_panel_w * 2 + 2)  # fallback
            clicked_cfg = hint_button_from_click(
                x, y, page_w, self._dvs_panel_h,
                _DVS_CFG_BTNS, bar_h=self._hint_bar_h,
            )
            if clicked_cfg is not None and clicked_cfg != self._dvs_config_mode:
                self._dvs_config_mode = clicked_cfg
                if clicked_cfg == "hybrid":
                    self._camera_mgr.switch_dvs_to_hybrid()
                else:
                    self._camera_mgr.switch_dvs_to_tracking()
                print(f"[CAL] DVS config -> {self._dvs_config_mode}")
                return

        # Page mode — corner-drag logic for DVS (left) and RGB (right)

        # Global LBUTTONUP: reset drag state regardless of which panel the
        # cursor lands on.  Prevents stuck-drag when releasing across panels.
        if event == cv2.EVENT_LBUTTONUP:
            self._dragging_idx = None
            self._rgb_dragging_idx = None
            return

        scale = DVS_SCALE

        if x < self._dvs_panel_w:
            # --- DVS panel dragging ---
            if self._dvs_corners is None:
                return

            if event == cv2.EVENT_LBUTTONDOWN:
                pts_disp = self._dvs_corners * scale
                dists = np.sqrt(((pts_disp - [x, y]) ** 2).sum(axis=1))
                idx = int(np.argmin(dists))
                if dists[idx] < self._hit_radius:
                    self._dragging_idx = idx

            elif event == cv2.EVENT_MOUSEMOVE and self._dragging_idx is not None:
                nx = np.clip(x / scale, 0, DVS_WIDTH - 1)
                ny = np.clip(y / scale, 0, DVS_HEIGHT - 1)
                self._dvs_corners[self._dragging_idx] = [nx, ny]

        elif x >= self._rgb_panel_offset_x and self._rgb_quad is not None:
            # --- RGB panel dragging ---
            rgb_scale = self._rgb_scale
            cam_x = (x - self._rgb_panel_offset_x) / rgb_scale
            cam_y = y / rgb_scale

            if event == cv2.EVENT_LBUTTONDOWN:
                corners = self._rgb_quad.corners
                dists = np.sqrt(((corners - [cam_x, cam_y]) ** 2).sum(axis=1))
                idx = int(np.argmin(dists))
                # Hit test in display space
                if dists[idx] * rgb_scale < self._hit_radius:
                    self._rgb_dragging_idx = idx

            elif event == cv2.EVENT_MOUSEMOVE and self._rgb_dragging_idx is not None:
                cam_w, cam_h = self._rgb_cam_size
                cam_x = np.clip(cam_x, 0, cam_w - 1)
                cam_y = np.clip(cam_y, 0, cam_h - 1)
                self._rgb_quad.corners[self._rgb_dragging_idx] = [cam_x, cam_y]
