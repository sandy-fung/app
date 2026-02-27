"""Physical DVS output — DVS gesture drives LinkerHand.

When active and tracking enabled, changes in dvs_stable are applied
to the hand via HandBridge (with game mode transformation).
"""

import numpy as np

from app.core.demo import OutputMode
from app.core.display import draw_paused_overlay
from app.core.inference.common import BATTLE_MAP

from .gui_output import _resolve_gesture, _draw_gesture_icon


class GesturePhysDVSOutput(OutputMode):
    """DVS gesture -> HandBridge -> LinkerHand."""

    def __init__(self, gesture_demo, hand_bridge, hand_thread):
        self._demo = gesture_demo
        self._bridge = hand_bridge
        self._hand = hand_thread
        self._result = None
        self._last_sent: str = ""
        self._was_moving: bool = False

    def activate(self) -> None:
        # Start paused for safety
        self._demo.tracking_enabled = False
        self._last_sent = ""
        self._was_moving = False
        print("[PHYS_DVS_GEST] Activated — PAUSED (press Space to start)")

    def deactivate(self) -> None:
        self._bridge.put_neutral()
        self._last_sent = ""
        self._was_moving = False
        print("[PHYS_DVS_GEST] Deactivated — hand returning to neutral")

    def on_tracking_changed(self, enabled: bool) -> None:
        if not enabled:
            self._bridge.put_neutral()
            self._last_sent = ""
        self._was_moving = False

    def process(self, result) -> None:
        self._result = result
        if not self._demo.tracking_enabled:
            return

        # Hand still moving — skip predictions
        if self._hand and self._hand.moving:
            self._was_moving = True
            return

        # Hand just arrived — flush stale voter data
        if self._was_moving:
            self._demo.reset_voters()
            self._was_moving = False
            return

        stable = result.dvs_stable
        if stable == self._last_sent or stable == "none":
            return

        # Apply game mode
        if result.game_mode == "battle":
            target = BATTLE_MAP.get(stable, stable)
        else:
            target = stable

        self._bridge.put_gesture(target)
        self._last_sent = stable

    def render(self) -> np.ndarray:
        result = self._result

        if result and result.dvs_display is not None:
            panel = result.dvs_display.copy()
        else:
            panel = np.zeros((480, 492, 3), dtype=np.uint8)

        if result:
            target = _resolve_gesture(result.dvs_stable, result.game_mode)
            _draw_gesture_icon(panel, target)

        if not self._demo.tracking_enabled:
            draw_paused_overlay(panel)

        return panel
