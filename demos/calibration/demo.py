"""Calibration tab — DVS quad calibration + RGB quad detection.

Embeds the same drag-corner logic from quad_calibrator.py into the
unified MainLoop (no separate window).

Sub-tabs: Page Calibration (default) and Arm Calibration (when arm present).
"""

from typing import Optional

import cv2
import numpy as np

from app.config import DVS_WIDTH, DVS_HEIGHT, DVS_SCALE
from app.core.demo import Demo
from app.core.display import (
    draw_hint_bar, resize_to_height,
    render_sub_tab_bar, sub_tab_from_click, SUB_TAB_BAR_HEIGHT,
)

# Sub-tab definitions
_SUB_TABS = [("page", "Page Cal"), ("arm", "Arm Cal")]


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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def activate(self, camera_mgr) -> None:
        self._camera_mgr = camera_mgr
        camera_mgr.switch_dvs_to_hybrid()

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
        from quad_calibrator import draw_overlay, grab_gray_frame
        from main_laser_drawing import draw_quad

        scale = DVS_SCALE

        # --- DVS panel ---
        gray = grab_gray_frame(self._camera_mgr.xe_cam)
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
            rgb_display = rgb_frame.copy()
            if self._rgb_quad:
                draw_quad(rgb_display, self._rgb_quad)
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
        composed = np.hstack([dvs_panel, sep, rgb_panel])

        # Hint bar at bottom
        dvs_ok = "OK" if self._store.dvs_calibrated else "--"
        rgb_ok = "OK" if self._store.rgb_calibrated else "--"
        hints = [
            f"DVS: {dvs_ok}  |  RGB: {rgb_ok}",
            "[R] reset DVS  [D] re-detect RGB  (auto-save on exit)",
        ]
        if self._has_arm:
            hints.append("[Tab] switch to Arm Cal")
        draw_hint_bar(composed, hints)

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
        """Handle mouse events for DVS corner dragging.

        Note: x, y are already adjusted for tab bar offset by MainLoop.
        Only DVS panel (left side) accepts drag events.
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

        # Page mode — original corner-drag logic
        if self._dvs_corners is None:
            return

        scale = DVS_SCALE

        # Ignore clicks on RGB panel
        if x >= self._dvs_panel_w:
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

        elif event == cv2.EVENT_LBUTTONUP:
            self._dragging_idx = None
