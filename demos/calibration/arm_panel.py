"""Arm Calibration panel — gripper control.

Renders arm status and gripper progress bar with buttons.
Pen up/down controls have been moved to global arm buttons (see display.py).
"""

import cv2
import numpy as np

from app.core.display import draw_hint_bar

# Layout constants
_SECTION_PAD = 12
_ROW_H = 32
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_S = 0.50
_FONT_XS = 0.42
_WHITE = (220, 220, 220)
_GRAY = (130, 130, 130)
_DARK = (40, 40, 40)
_ORANGE = (0, 140, 255)
_GREEN = (0, 200, 80)
_RED = (80, 80, 255)
_BTN_H = 26
_BTN_RADIUS = 4


# ------------------------------------------------------------------
# Lightweight button helpers
# ------------------------------------------------------------------

def _draw_button(img, x, y, w, label, active=False):
    """Draw a rounded-ish button; return (x1, y1, x2, y2) rect."""
    color = _ORANGE if active else (80, 80, 80)
    cv2.rectangle(img, (x, y), (x + w, y + _BTN_H), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + _BTN_H), (160, 160, 160), 1)
    ts = cv2.getTextSize(label, _FONT, _FONT_XS, 1)[0]
    tx = x + (w - ts[0]) // 2
    ty = y + (_BTN_H + ts[1]) // 2 - 1
    cv2.putText(img, label, (tx, ty), _FONT, _FONT_XS, (255, 255, 255), 1, cv2.LINE_AA)
    return (x, y, x + w, y + _BTN_H)


def _draw_badge(img, x, y, text, color):
    """Draw a small status badge."""
    ts = cv2.getTextSize(text, _FONT, _FONT_XS, 1)[0]
    bw = ts[0] + 12
    cv2.rectangle(img, (x, y), (x + bw, y + _BTN_H), color, -1)
    tx = x + (bw - ts[0]) // 2
    ty = y + (_BTN_H + ts[1]) // 2 - 1
    cv2.putText(img, text, (tx, ty), _FONT, _FONT_XS, (255, 255, 255), 1, cv2.LINE_AA)
    return bw


class ArmCalibrationPanel:
    """Self-contained panel for arm calibration inside the Calibration tab."""

    def __init__(self, bridge, arm_thread):
        self._bridge = bridge
        self._arm = arm_thread

        # Gripper (lazy)
        self._gripper = None
        self._grip_pos_mm: float = 40.0  # target

        # Click regions [(x1,y1,x2,y2, action), ...]
        self._buttons: list = []

    # ------------------------------------------------------------------
    # Gripper helpers
    # ------------------------------------------------------------------

    def _ensure_gripper(self):
        """Create GripperController once arm is ready."""
        if self._gripper is not None:
            return True
        if not self._arm.is_ready.is_set():
            return False
        piper = self._arm.piper
        if piper is None:
            return False
        from piper_demo.gripper import GripperController
        self._gripper = GripperController(piper)
        return True

    def _send_grip(self):
        pos = max(0.0, min(80.0, self._grip_pos_mm))
        self._grip_pos_mm = pos
        if self._ensure_gripper():
            self._gripper.set_position_mm(pos)

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self, width: int, height: int) -> np.ndarray:
        """Render the arm calibration panel as a BGR image."""
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self._buttons = []
        y = _SECTION_PAD

        # ---- ARM STATUS ----
        y = self._render_status(canvas, y, width)

        # ---- GRIPPER ----
        y += _SECTION_PAD
        y = self._render_gripper(canvas, y, width)

        # ---- Hint bar ----
        draw_hint_bar(canvas, [
            "[O] open  [C] close  [< >] +/-1mm  [[ ]] +/-10mm",
            "[Tab] switch sub-tab",
        ])

        return canvas

    def _render_status(self, img, y, w):
        """ARM STATUS section."""
        cv2.putText(img, "ARM STATUS", (_SECTION_PAD, y + 16),
                    _FONT, _FONT_S, _GRAY, 1, cv2.LINE_AA)
        y += 24

        # Status badge
        if self._arm.error:
            badge_text, badge_color = "ERR", _RED
        elif self._arm.is_ready.is_set():
            badge_text, badge_color = "RDY", _GREEN
        else:
            badge_text, badge_color = "INIT", _ORANGE

        _draw_badge(img, _SECTION_PAD, y, badge_text, badge_color)

        if self._arm.error:
            y += _ROW_H + 4
            err_text = f"Error: {self._arm.error[:60]}"
            cv2.putText(img, err_text, (_SECTION_PAD, y + 14),
                        _FONT, _FONT_XS, _RED, 1, cv2.LINE_AA)

        return y + _ROW_H

    def _render_gripper(self, img, y, w):
        """GRIPPER section with progress bar and buttons."""
        cv2.putText(img, "GRIPPER", (_SECTION_PAD, y + 16),
                    _FONT, _FONT_S, _GRAY, 1, cv2.LINE_AA)
        y += 24

        bar_x = _SECTION_PAD
        bar_w = min(w - 2 * _SECTION_PAD - 160, 300)
        bar_h = 18

        # Progress bar background
        cv2.rectangle(img, (bar_x, y), (bar_x + bar_w, y + bar_h), (60, 60, 60), -1)

        # Filled portion
        frac = self._grip_pos_mm / 80.0
        fill_w = int(bar_w * frac)
        if fill_w > 0:
            cv2.rectangle(img, (bar_x, y), (bar_x + fill_w, y + bar_h), _ORANGE, -1)

        # Border
        cv2.rectangle(img, (bar_x, y), (bar_x + bar_w, y + bar_h), (100, 100, 100), 1)

        # Value label
        val_text = f"{self._grip_pos_mm:.0f} mm"
        cv2.putText(img, val_text, (bar_x + bar_w + 8, y + 14),
                    _FONT, _FONT_XS, _WHITE, 1, cv2.LINE_AA)

        # HW readback
        hw_text = ""
        if self._gripper is not None:
            try:
                hw_pos = self._gripper.read_position_mm()
                hw_eff = self._gripper.read_effort()
                hw_text = f"(hw: {hw_pos:.1f}mm  eff: {hw_eff:.1f}Nm)"
            except Exception:
                hw_text = "(hw: read err)"
        cv2.putText(img, hw_text, (bar_x, y + bar_h + 16),
                    _FONT, _FONT_XS, _GRAY, 1, cv2.LINE_AA)

        # Buttons row
        btn_y = y + bar_h + 28
        btn_w = 60
        gap = 8

        bx = _SECTION_PAD
        r = _draw_button(img, bx, btn_y, btn_w, "Open")
        self._buttons.append((*r, "open"))

        bx += btn_w + gap
        r = _draw_button(img, bx, btn_y, btn_w, "Close")
        self._buttons.append((*r, "close"))

        return btn_y + _BTN_H + 4

    # ------------------------------------------------------------------
    # Key handling
    # ------------------------------------------------------------------

    def handle_key(self, key: int) -> bool:
        """Handle keyboard input. Return True if consumed."""
        if key in (ord('o'), ord('O')):
            self._grip_pos_mm = 80.0
            self._send_grip()
            return True

        if key in (ord('c'), ord('C')):
            self._grip_pos_mm = 0.0
            self._send_grip()
            return True

        if key == 81 or key == 2:  # Left arrow
            self._grip_pos_mm = max(0.0, self._grip_pos_mm - 1.0)
            self._send_grip()
            return True

        if key == 83 or key == 3:  # Right arrow
            self._grip_pos_mm = min(80.0, self._grip_pos_mm + 1.0)
            self._send_grip()
            return True

        if key == ord('['):
            self._grip_pos_mm = max(0.0, self._grip_pos_mm - 10.0)
            self._send_grip()
            return True

        if key == ord(']'):
            self._grip_pos_mm = min(80.0, self._grip_pos_mm + 10.0)
            self._send_grip()
            return True

        return False

    # ------------------------------------------------------------------
    # Mouse handling
    # ------------------------------------------------------------------

    def mouse_callback(self, event: int, x: int, y: int,
                       flags: int, param) -> None:
        """Handle mouse clicks on buttons."""
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        for (x1, y1, x2, y2, action) in self._buttons:
            if x1 <= x <= x2 and y1 <= y <= y2:
                if action == "open":
                    self._grip_pos_mm = 80.0
                    self._send_grip()
                elif action == "close":
                    self._grip_pos_mm = 0.0
                    self._send_grip()
                return
