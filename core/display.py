"""Display composition helpers.

Layout functions are adapted from examples/ex17/dual_tracker_compare.py.
"""

from typing import List, Optional, Set, Tuple

import cv2
import numpy as np

from app.core.demo import OutputModeType

# Layout constants
LAYOUT_FULL = 0
LAYOUT_TRAJECTORY = 1
LAYOUT_PIP = 2
LAYOUT_NAMES = ["FULL", "TRAJECTORY", "PIP"]

# Tab bar
TAB_BAR_HEIGHT = 36

# Mode buttons
MODE_BTN_W = 56
MODE_BTN_GAP = 6
MODE_BTN_PAD = 10
MODE_LABELS = {
    OutputModeType.GUI: "GUI",
    OutputModeType.PHYS_DVS: "DVS",
    OutputModeType.PHYS_RGB: "RGB",
}
MODE_ORDER = [OutputModeType.GUI, OutputModeType.PHYS_DVS, OutputModeType.PHYS_RGB]

# Separate mode-button row (below tab bar)
MODE_ROW_HEIGHT = 32

SUB_TAB_BAR_HEIGHT = 28
_SUB_BTN_W = 90
_SUB_BTN_GAP = 4


def mode_buttons_width(n_modes: int) -> int:
    """Total pixel width of the mode-button area."""
    if n_modes <= 0:
        return 0
    return MODE_BTN_PAD * 2 + n_modes * MODE_BTN_W + max(0, n_modes - 1) * MODE_BTN_GAP


def resize_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    """Resize image to target height while preserving aspect ratio."""
    h, w = img.shape[:2]
    if h == target_h:
        return img
    scale = target_h / h
    new_w = int(w * scale)
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_LINEAR)


def pad_to_width(img: np.ndarray, target_w: int) -> np.ndarray:
    """Pad image on the right with black to reach target_w."""
    h, w = img.shape[:2]
    if w >= target_w:
        return img[:, :target_w]
    pad = np.zeros((h, target_w - w, 3), dtype=np.uint8)
    return np.hstack([img, pad])


def normalize_frame(
    frame: np.ndarray, target_w: int, target_h: int,
) -> tuple:
    """Letterbox frame to target_w x target_h, centered.

    Returns (normalized, scale, pad_x, pad_y) for reverse mouse mapping.
    """
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_h, target_w, 3), (240, 240, 240), dtype=np.uint8)
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y


def make_label_bar(text: str, width: int, height: int = 28,
                   bg_color=(40, 40, 40),
                   fg_color=(220, 220, 220)) -> np.ndarray:
    """Create a thin label bar image."""
    bar = np.full((height, width, 3), bg_color, dtype=np.uint8)
    cv2.putText(bar, text, (8, height - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, fg_color, 1, cv2.LINE_AA)
    return bar


def draw_status_on(panel: np.ndarray, label: str, tracking: bool,
                   fps: float,
                   coord: Optional[Tuple[float, float]]) -> None:
    """Draw a simplified status line on top of a panel."""
    status = "ON" if tracking else "OFF"
    color = (0, 255, 0) if tracking else (0, 0, 255)
    text = f"{label} | Track: {status} | FPS: {fps:.0f}"
    if coord:
        text += f" | ({coord[0]:.2f}, {coord[1]:.2f})"
    cv2.putText(panel, text, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def draw_hint_bar(
    frame: np.ndarray,
    lines: list,
    font_scale: float = 0.5,
    thickness: int = 1,
    padding: int = 8,
    alpha: float = 0.6,
) -> None:
    """Draw a semi-transparent dark bar with hint text at the bottom of *frame*.

    Args:
        frame: BGR image to draw on (modified in-place).
        lines: List of hint strings or (str, bgr_tuple) for custom color.
               Last item renders at the very bottom.
        font_scale: cv2 font scale (0.5 matches draw_status_on).
        thickness: text stroke thickness.
        padding: pixel padding around text.
        alpha: darkening factor for the bar (0 = transparent, 1 = opaque black).
    """
    if not lines or frame.shape[0] < 60:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    default_color = (220, 220, 220)

    # Measure line heights
    parsed = []  # [(text, color, text_height), ...]
    for item in lines:
        if isinstance(item, str):
            text, color = item, default_color
        else:
            text, color = item[0], item[1]
        (_, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        parsed.append((text, color, th))

    line_gap = 6
    bar_h = padding * 2 + sum(th for _, _, th in parsed) + line_gap * max(0, len(parsed) - 1)
    # Clamp bar to at most 1/4 of frame height
    bar_h = min(bar_h, frame.shape[0] // 4)

    h, w = frame.shape[:2]

    # Darken the ROI
    roi = frame[h - bar_h:h, 0:w]
    cv2.addWeighted(roi, 1.0 - alpha, np.zeros_like(roi), alpha, 0, dst=roi)

    # Render text lines bottom-up
    y_cursor = h - padding
    for text, color, th in reversed(parsed):
        cv2.putText(frame, text, (padding, y_cursor),
                    font, font_scale, color, thickness, cv2.LINE_AA)
        y_cursor -= th + line_gap


def draw_paused_overlay(frame: np.ndarray) -> None:
    """Draw a semi-transparent dark overlay with PAUSED text on *frame* (in-place).

    Used by physical output modes to indicate tracking is paused.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)

    # Large "PAUSED" text centered
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = min(w, h) / 300.0  # adaptive size
    thickness = max(2, int(scale * 2))
    (tw, th), _ = cv2.getTextSize("PAUSED", font, scale, thickness)
    tx = (w - tw) // 2
    ty = (h + th) // 2 - 20
    cv2.putText(frame, "PAUSED", (tx, ty),
                font, scale, (0, 0, 255), thickness, cv2.LINE_AA)

    # Hint below
    hint = "Press [Space] to start tracking"
    hint_scale = scale * 0.4
    hint_thick = max(1, int(hint_scale * 2))
    (hw, hh), _ = cv2.getTextSize(hint, font, hint_scale, hint_thick)
    cv2.putText(frame, hint, ((w - hw) // 2, ty + th + 10),
                font, hint_scale, (220, 220, 220), hint_thick, cv2.LINE_AA)


def compose_full(
    dvs_display: np.ndarray,
    rgb_display: np.ndarray,
    dvs_canvas: np.ndarray,
    rgb_canvas: np.ndarray,
) -> np.ndarray:
    """Layout FULL: 4-panel grid.

    [ DVS camera ] [ RGB camera ]
    [ DVS canvas ] [ RGB canvas ]
    """
    cam_h = max(dvs_display.shape[0], rgb_display.shape[0])
    dvs_cam = resize_to_height(dvs_display, cam_h)
    rgb_cam = resize_to_height(rgb_display, cam_h)

    cam_row_w = dvs_cam.shape[1] + rgb_cam.shape[1]
    cam_row = np.hstack([dvs_cam, rgb_cam])

    canvas_h = dvs_canvas.shape[0]
    canvas_total_w = dvs_canvas.shape[1] + rgb_canvas.shape[1]

    canvas_row = np.hstack([dvs_canvas, rgb_canvas])
    if canvas_total_w != cam_row_w:
        canvas_row = cv2.resize(
            canvas_row,
            (cam_row_w, int(canvas_h * cam_row_w / canvas_total_w)),
            interpolation=cv2.INTER_LINEAR,
        )

    return np.vstack([cam_row, canvas_row])


def compose_trajectory(
    dvs_canvas: np.ndarray,
    rgb_canvas: np.ndarray,
) -> np.ndarray:
    """Layout TRAJECTORY: side-by-side canvases only."""
    return np.hstack([dvs_canvas, rgb_canvas])


def compose_pip(
    dvs_display: np.ndarray,
    rgb_display: np.ndarray,
    dvs_canvas: np.ndarray,
    rgb_canvas: np.ndarray,
    pip_h: int = 120,
) -> np.ndarray:
    """Layout PIP: trajectory canvases with small camera preview inset."""
    dvs_pip = resize_to_height(dvs_display, pip_h)
    rgb_pip = resize_to_height(rgb_display, pip_h)

    dvs_out = dvs_canvas.copy()
    rgb_out = rgb_canvas.copy()

    pip_margin = 6
    # DVS PIP
    dph, dpw = dvs_pip.shape[:2]
    if dpw < dvs_out.shape[1] - pip_margin and dph < dvs_out.shape[0] - pip_margin:
        dvs_out[pip_margin:pip_margin + dph, pip_margin:pip_margin + dpw] = dvs_pip
        cv2.rectangle(dvs_out, (pip_margin, pip_margin),
                      (pip_margin + dpw, pip_margin + dph), (100, 100, 100), 1)

    # RGB PIP
    rph, rpw = rgb_pip.shape[:2]
    if rpw < rgb_out.shape[1] - pip_margin and rph < rgb_out.shape[0] - pip_margin:
        rgb_out[pip_margin:pip_margin + rph, pip_margin:pip_margin + rpw] = rgb_pip
        cv2.rectangle(rgb_out, (pip_margin, pip_margin),
                      (pip_margin + rpw, pip_margin + rph), (100, 100, 100), 1)

    return np.hstack([dvs_out, rgb_out])


# ---------------------------------------------------------------------------
# Tab bar
# ---------------------------------------------------------------------------

def render_tab_bar(
    tabs: List[Tuple[str, str]],
    active: str,
    width: int,
    height: int = TAB_BAR_HEIGHT,
    reserved_right: int = 0,
) -> np.ndarray:
    """Render a clickable tab selector bar.

    Args:
        tabs: [(key, name), ...] e.g. [("1","Calibration"), ("2","Tracking")]
        active: currently active tab name
        width: pixel width of the bar
        height: pixel height of the bar
        reserved_right: pixels reserved on the right for mode buttons

    Returns:
        BGR image (height, width, 3)
    """
    bar = np.full((height, width, 3), (240, 240, 240), dtype=np.uint8)
    n = len(tabs)
    if n == 0:
        return bar

    tab_area_w = width - reserved_right
    tab_w = tab_area_w // n

    for i, (_key, name) in enumerate(tabs):
        x1 = i * tab_w
        x2 = (i + 1) * tab_w if i < n - 1 else tab_area_w
        is_active = (name == active)

        if is_active:
            # Active tab: white bg + orange accent bar at bottom
            cv2.rectangle(bar, (x1, 0), (x2, height), (255, 255, 255), -1)
            cv2.line(bar, (x1, height - 3), (x2, height - 3), (0, 140, 255), 3)
            text_color = (40, 40, 40)
        else:
            # Inactive tab: light gray
            cv2.rectangle(bar, (x1, 0), (x2, height), (225, 225, 225), -1)
            text_color = (120, 120, 120)

        # Tab label
        text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
        text_x = x1 + (x2 - x1 - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2 - 2
        cv2.putText(bar, name, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 1, cv2.LINE_AA)

        # Separator between tabs
        if i > 0:
            cv2.line(bar, (x1, 6), (x1, height - 6), (200, 200, 200), 1)

    return bar


def tab_index_from_click(x: int, y: int, num_tabs: int,
                         frame_width: int,
                         reserved_right: int = 0) -> Optional[int]:
    """Return tab index from mouse click coordinates, or None if outside.

    Args:
        x, y: mouse click position
        num_tabs: total number of tabs
        frame_width: width of the display frame
        reserved_right: pixels reserved on the right for mode buttons
    """
    if y >= TAB_BAR_HEIGHT or num_tabs == 0:
        return None
    tab_area_w = frame_width - reserved_right
    if x >= tab_area_w:
        return None
    tab_w = tab_area_w // num_tabs
    idx = x // tab_w
    return min(idx, num_tabs - 1)


# ---------------------------------------------------------------------------
# Output mode buttons
# ---------------------------------------------------------------------------

def render_mode_buttons(
    modes: List[OutputModeType],
    active: Optional[OutputModeType],
    available: Set[OutputModeType],
    width: int,
    height: int = TAB_BAR_HEIGHT,
) -> np.ndarray:
    """Render output-mode toggle buttons for the right side of the tab bar.

    Args:
        modes: ordered list of mode types to display
        active: currently active mode (orange highlight)
        available: set of clickable modes (white); others shown as gray
        width: pixel width of the button area
        height: pixel height
    """
    bar = np.full((height, width, 3), (240, 240, 240), dtype=np.uint8)
    n = len(modes)
    if n == 0:
        return bar

    btn_x = MODE_BTN_PAD
    y1, y2 = 6, height - 6
    for mode in modes:
        x1, x2 = btn_x, btn_x + MODE_BTN_W
        label = MODE_LABELS.get(mode, mode.value)

        if mode == active:
            # Active: orange background, white text
            cv2.rectangle(bar, (x1, y1), (x2, y2), (0, 140, 255), -1)
            text_color = (255, 255, 255)
        elif mode in available:
            # Available but inactive: white background, dark text
            cv2.rectangle(bar, (x1, y1), (x2, y2), (255, 255, 255), -1)
            cv2.rectangle(bar, (x1, y1), (x2, y2), (180, 180, 180), 1)
            text_color = (60, 60, 60)
        else:
            # Unavailable: gray
            cv2.rectangle(bar, (x1, y1), (x2, y2), (210, 210, 210), -1)
            text_color = (160, 160, 160)

        ts = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        tx = x1 + (MODE_BTN_W - ts[0]) // 2
        ty = (height + ts[1]) // 2 - 2
        cv2.putText(bar, label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)

        btn_x = x2 + MODE_BTN_GAP

    return bar


def render_sub_tab_bar(
    tabs: List[Tuple[str, str]],
    active: str,
    width: int,
    height: int = SUB_TAB_BAR_HEIGHT,
    available: Optional[Set[str]] = None,
) -> np.ndarray:
    """Render a centered sub-tab selector bar.

    Uses the same active/inactive/disabled color scheme as
    render_mode_buttons for visual consistency.

    Args:
        available: set of enabled tab keys. None means all enabled.
    """
    bar = np.full((height, width, 3), (240, 240, 240), dtype=np.uint8)
    n = len(tabs)
    if n == 0:
        return bar
    total_w = n * _SUB_BTN_W + (n - 1) * _SUB_BTN_GAP
    x_start = (width - total_w) // 2
    y1, y2 = 3, height - 3

    for i, (key, label) in enumerate(tabs):
        x = x_start + i * (_SUB_BTN_W + _SUB_BTN_GAP)
        enabled = available is None or key in available
        if key == active and enabled:
            # Active: orange fill, white text
            cv2.rectangle(bar, (x, y1), (x + _SUB_BTN_W, y2), (0, 140, 255), -1)
            text_color = (255, 255, 255)
        elif enabled:
            # Inactive: white fill + gray border
            cv2.rectangle(bar, (x, y1), (x + _SUB_BTN_W, y2), (255, 255, 255), -1)
            cv2.rectangle(bar, (x, y1), (x + _SUB_BTN_W, y2), (180, 180, 180), 1)
            text_color = (60, 60, 60)
        else:
            # Disabled: light gray, no border
            cv2.rectangle(bar, (x, y1), (x + _SUB_BTN_W, y2), (210, 210, 210), -1)
            text_color = (160, 160, 160)

        ts = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        tx = x + (_SUB_BTN_W - ts[0]) // 2
        ty = (height + ts[1]) // 2 - 1
        cv2.putText(bar, label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)

    return bar


def sub_tab_from_click(
    x: int, y: int,
    tabs: List[Tuple[str, str]],
    width: int,
    height: int = SUB_TAB_BAR_HEIGHT,
    available: Optional[Set[str]] = None,
) -> Optional[str]:
    """Hit-test a sub-tab bar. Return tab key or None.

    Disabled tabs (not in *available*) are ignored.
    """
    if y >= height:
        return None
    n = len(tabs)
    if n == 0:
        return None
    total_w = n * _SUB_BTN_W + (n - 1) * _SUB_BTN_GAP
    x_start = (width - total_w) // 2
    for i, (key, _label) in enumerate(tabs):
        bx = x_start + i * (_SUB_BTN_W + _SUB_BTN_GAP)
        if bx <= x <= bx + _SUB_BTN_W:
            if available is not None and key not in available:
                return None
            return key
    return None


# ---------------------------------------------------------------------------
# Mode-button row (separate row below tab bar)
# ---------------------------------------------------------------------------

def render_mode_row(
    modes: List[OutputModeType],
    active: Optional[OutputModeType],
    available: Set[OutputModeType],
    width: int,
    height: int = MODE_ROW_HEIGHT,
) -> np.ndarray:
    """Render a standalone mode-button row (below the tab bar).

    Buttons are left-aligned. A thin separator line is drawn at the bottom.
    """
    bar = np.full((height, width, 3), (235, 235, 235), dtype=np.uint8)
    n = len(modes)
    if n == 0:
        return bar

    btn_x = MODE_BTN_PAD
    y1, y2 = 4, height - 5

    for mode in modes:
        x1, x2 = btn_x, btn_x + MODE_BTN_W
        label = MODE_LABELS.get(mode, mode.value)

        if mode == active:
            cv2.rectangle(bar, (x1, y1), (x2, y2), (0, 140, 255), -1)
            text_color = (255, 255, 255)
        elif mode in available:
            cv2.rectangle(bar, (x1, y1), (x2, y2), (255, 255, 255), -1)
            cv2.rectangle(bar, (x1, y1), (x2, y2), (180, 180, 180), 1)
            text_color = (60, 60, 60)
        else:
            cv2.rectangle(bar, (x1, y1), (x2, y2), (210, 210, 210), -1)
            text_color = (160, 160, 160)

        ts = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        tx = x1 + (MODE_BTN_W - ts[0]) // 2
        ty = (height + ts[1]) // 2 - 2
        cv2.putText(bar, label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)

        btn_x = x2 + MODE_BTN_GAP

    # Bottom separator line
    cv2.line(bar, (0, height - 1), (width, height - 1), (200, 200, 200), 1)
    return bar


def mode_row_click(
    x: int, y: int,
    modes: List[OutputModeType],
    height: int = MODE_ROW_HEIGHT,
) -> Optional[OutputModeType]:
    """Hit-test a mode-button row. Return OutputModeType or None."""
    if y >= height or len(modes) == 0:
        return None
    local_x = x - MODE_BTN_PAD
    if local_x < 0:
        return None
    for i, mode in enumerate(modes):
        bx = i * (MODE_BTN_W + MODE_BTN_GAP)
        if bx <= local_x <= bx + MODE_BTN_W:
            return mode
    return None


# ---------------------------------------------------------------------------
# Arm control buttons (HOME / DRAW)
# ---------------------------------------------------------------------------

ARM_BTN_W = 56
ARM_BTN_GAP = 6
ARM_BTN_PAD = 10
ARM_BUTTONS = ["HOME", "DRAW", "PEN v", "PEN ^"]
_ARM_SEP_AFTER = {"DRAW"}  # draw vertical separator after these labels


def arm_buttons_width() -> int:
    """Total pixel width of the arm-button area (including separator gap)."""
    n = len(ARM_BUTTONS)
    # Extra space for vertical separators
    sep_extra = len(_ARM_SEP_AFTER) * (ARM_BTN_GAP + 2)
    return ARM_BTN_PAD * 2 + n * ARM_BTN_W + max(0, n - 1) * ARM_BTN_GAP + sep_extra


def render_arm_buttons(
    at_home: bool,
    width: int,
    height: int = TAB_BAR_HEIGHT,
    pen_down: bool = False,
) -> np.ndarray:
    """Render HOME / DRAW / PEN buttons for the right side of the tab bar.

    Args:
        at_home: True → HOME highlighted; False → DRAW highlighted.
        width: pixel width of the button area.
        height: pixel height.
        pen_down: current pen state (for PEN button highlighting).
    """
    bar = np.full((height, width, 3), (240, 240, 240), dtype=np.uint8)

    # Vertical separator on the left edge
    cv2.line(bar, (2, 6), (2, height - 6), (180, 180, 180), 1)

    btn_x = ARM_BTN_PAD
    y1, y2 = 6, height - 6

    for label in ARM_BUTTONS:
        x1, x2 = btn_x, btn_x + ARM_BTN_W

        if label == "HOME" and at_home:
            # At home → HOME green highlight
            cv2.rectangle(bar, (x1, y1), (x2, y2), (0, 180, 80), -1)
            text_color = (255, 255, 255)
        elif label == "DRAW" and not at_home:
            # Working → DRAW orange highlight
            cv2.rectangle(bar, (x1, y1), (x2, y2), (0, 140, 255), -1)
            text_color = (255, 255, 255)
        elif label == "PEN v" and at_home:
            # Disabled when at home
            cv2.rectangle(bar, (x1, y1), (x2, y2), (210, 210, 210), -1)
            text_color = (160, 160, 160)
        elif label == "PEN v" and pen_down:
            # Active: pen is down
            cv2.rectangle(bar, (x1, y1), (x2, y2), (0, 140, 255), -1)
            text_color = (255, 255, 255)
        elif label == "PEN ^" and at_home:
            # Disabled when at home
            cv2.rectangle(bar, (x1, y1), (x2, y2), (210, 210, 210), -1)
            text_color = (160, 160, 160)
        elif label == "PEN ^" and not pen_down:
            # Active: pen is up
            cv2.rectangle(bar, (x1, y1), (x2, y2), (0, 140, 255), -1)
            text_color = (255, 255, 255)
        else:
            # Inactive: white bg + gray border
            cv2.rectangle(bar, (x1, y1), (x2, y2), (255, 255, 255), -1)
            cv2.rectangle(bar, (x1, y1), (x2, y2), (180, 180, 180), 1)
            text_color = (60, 60, 60)

        ts = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        tx = x1 + (ARM_BTN_W - ts[0]) // 2
        ty = (height + ts[1]) // 2 - 2
        cv2.putText(bar, label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)

        btn_x = x2 + ARM_BTN_GAP

        # Draw vertical separator after designated buttons
        if label in _ARM_SEP_AFTER:
            sep_x = btn_x + 1
            cv2.line(bar, (sep_x, 6), (sep_x, height - 6), (180, 180, 180), 1)
            btn_x += ARM_BTN_GAP + 2

    return bar


def arm_button_from_click(
    x: int, y: int,
    frame_width: int,
    reserved_right: int,
) -> Optional[str]:
    """Return button label if click hits an arm button, else None.

    Args:
        reserved_right: total width of arm-button area (from render_arm_buttons).
    """
    if y >= TAB_BAR_HEIGHT or reserved_right <= 0:
        return None
    area_start = frame_width - reserved_right
    if x < area_start:
        return None
    local_x = x - area_start - ARM_BTN_PAD
    if local_x < 0:
        return None
    # Walk through buttons accounting for separator gaps
    bx = 0
    for i, label in enumerate(ARM_BUTTONS):
        if bx <= local_x <= bx + ARM_BTN_W:
            return label
        bx += ARM_BTN_W + ARM_BTN_GAP
        if label in _ARM_SEP_AFTER:
            bx += ARM_BTN_GAP + 2
    return None


def mode_button_from_click(
    x: int, y: int,
    modes: List[OutputModeType],
    frame_width: int,
) -> Optional[OutputModeType]:
    """Return the OutputModeType at click position, or None."""
    n = len(modes)
    if n == 0 or y >= TAB_BAR_HEIGHT:
        return None
    btn_area_w = mode_buttons_width(n)
    area_start = frame_width - btn_area_w
    if x < area_start:
        return None
    local_x = x - area_start - MODE_BTN_PAD
    if local_x < 0:
        return None
    for i, mode in enumerate(modes):
        bx = i * (MODE_BTN_W + MODE_BTN_GAP)
        if bx <= local_x <= bx + MODE_BTN_W:
            return mode
    return None
