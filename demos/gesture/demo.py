"""Gesture demo — Rock-Paper-Scissors recognition tab.

DVS inference runs in a background thread (~200fps).
RGB inference runs in the main thread (~30fps).
Results are merged into GestureResult and forwarded to the active output.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.core.demo import Demo
from app.core.inference.common import MajorityVoter


@dataclass
class GestureResult:
    """Snapshot of one frame's gesture recognition results."""

    # DVS
    dvs_display: Optional[np.ndarray] = None
    dvs_gesture: str = "none"
    dvs_confidence: float = 0.0
    dvs_probs: Optional[np.ndarray] = None
    dvs_stable: str = "none"
    dvs_fps: float = 0.0
    dvs_elapsed_ms: float = 0.0

    # RGB
    rgb_frame: Optional[np.ndarray] = None
    rgb_gesture: str = "none"
    rgb_confidence: float = 0.0
    rgb_stable: str = "none"
    rgb_fps: float = 0.0
    rgb_elapsed_ms: float = 0.0

    # Mode
    game_mode: str = "battle"


class GestureDemo(Demo):
    """Gesture (RPS) recognition tab — DVS + RGB side-by-side.

    Always shown as a tab. Inference engines are lazy-loaded on first
    :meth:`activate` so that missing model files only affect this tab.
    """

    def __init__(self):
        super().__init__("Gesture")

        # Inference engines (lazy-loaded)
        self._dvs_inference = None
        self._rgb_inference = None

        # Voters
        self._dvs_voter: Optional[MajorityVoter] = None
        self._rgb_voter: Optional[MajorityVoter] = None

        # DVS background thread
        self._dvs_thread = None

        # State
        self._game_mode = "battle"
        self._tracking_enabled = False
        self._result = GestureResult()

        # Camera ref
        self._camera_mgr = None

        # Load flags (only try once)
        self._dvs_load_attempted = False
        self._rgb_load_attempted = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def activate(self, camera_mgr) -> None:
        self._camera_mgr = camera_mgr

        # Switch DVS to tracking mode (pure DVS ~200fps)
        camera_mgr.switch_dvs_to_tracking()

        from app.config import (
            DVS_GESTURE_MODEL, MEDIAPIPE_MODEL,
            GESTURE_CONF, DVS_HOLD_FRAMES, RGB_HOLD_FRAMES,
            GESTURE_VOTE_MODE,
        )
        import os

        # --- DVS inference (lazy, once) ---
        if not self._dvs_load_attempted:
            self._dvs_load_attempted = True
            if os.path.isfile(DVS_GESTURE_MODEL):
                try:
                    from app.core.inference.dvs_gesture import DVSGestureInference
                    self._dvs_inference = DVSGestureInference(
                        model_path=DVS_GESTURE_MODEL,
                        use_fp16=False,         # --fp32
                        use_tensorrt=True,      # --tensorrt
                    )
                except Exception as e:
                    print(f"[GESTURE] DVS model load failed: {e}")
                    self._dvs_inference = None
            else:
                print(f"[GESTURE] DVS model not found: {DVS_GESTURE_MODEL}")

        # --- RGB inference (lazy, once) ---
        if not self._rgb_load_attempted:
            self._rgb_load_attempted = True
            if os.path.isfile(MEDIAPIPE_MODEL):
                try:
                    from app.core.inference.rgb_gesture import MediaPipeGestureInference
                    self._rgb_inference = MediaPipeGestureInference(
                        model_path=MEDIAPIPE_MODEL,
                    )
                except Exception as e:
                    print(f"[GESTURE] RGB model load failed: {e}")
                    self._rgb_inference = None
            else:
                print(f"[GESTURE] RGB model not found: {MEDIAPIPE_MODEL}")

        # --- Voters ---
        self._dvs_voter = MajorityVoter(
            window_size=DVS_HOLD_FRAMES,
            conf_threshold=GESTURE_CONF,
            vote_mode=GESTURE_VOTE_MODE,
        )
        self._rgb_voter = MajorityVoter(
            window_size=RGB_HOLD_FRAMES,
            conf_threshold=GESTURE_CONF,
            vote_mode=GESTURE_VOTE_MODE,
        )

        # --- Start DVS background thread ---
        if self._dvs_inference is not None:
            from app.demos.gesture.dvs_thread import DVSGestureThread
            from app.config import DVS_SCALE
            self._dvs_thread = DVSGestureThread(
                camera_mgr.xe_cam,
                self._dvs_inference,
                self._dvs_voter,
                scale=DVS_SCALE,
                bit_depth=4,
            )
            self._dvs_thread.start()
            print("[GESTURE] DVS gesture thread started")

        # Activate current output mode
        if self.active_output:
            self.active_output.activate()

    def deactivate(self) -> None:
        if self.active_output:
            self.active_output.deactivate()
        if self._dvs_thread:
            self._dvs_thread.stop()
            self._dvs_thread = None
        # Reset voters
        if self._dvs_voter:
            self._dvs_voter.clear()
        if self._rgb_voter:
            self._rgb_voter.clear()

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------

    def process_frame(self, camera_mgr) -> None:
        import time as _time

        # --- DVS: snapshot from background thread ---
        dvs_display = None
        dvs_gesture = "none"
        dvs_conf = 0.0
        dvs_probs = None
        dvs_stable = "none"
        dvs_fps = 0.0
        dvs_elapsed_ms = 0.0

        if self._dvs_thread:
            (dvs_display, dvs_gesture, dvs_conf,
             dvs_probs, dvs_stable, dvs_fps, dvs_elapsed_ms) = (
                self._dvs_thread.get_latest()
            )

        # --- RGB: inference in main thread ---
        rgb_frame = camera_mgr.read_rgb_frame()
        rgb_gesture = "none"
        rgb_conf = 0.0
        rgb_stable = "none"
        rgb_fps = 0.0
        rgb_elapsed_ms = 0.0

        if rgb_frame is not None and self._rgb_inference is not None:
            gesture, conf, elapsed = self._rgb_inference.predict(rgb_frame)
            now = _time.perf_counter()
            self._rgb_voter.push(gesture, conf, now)
            rgb_gesture = gesture
            rgb_conf = conf
            rgb_stable = self._rgb_voter.majority()
            rgb_elapsed_ms = elapsed * 1000
            rgb_fps = min(1000.0 / rgb_elapsed_ms, 999.0) if rgb_elapsed_ms > 0 else 0.0

        # --- Build result snapshot ---
        self._result = GestureResult(
            dvs_display=dvs_display,
            dvs_gesture=dvs_gesture,
            dvs_confidence=dvs_conf,
            dvs_probs=dvs_probs,
            dvs_stable=dvs_stable,
            dvs_fps=dvs_fps,
            dvs_elapsed_ms=dvs_elapsed_ms,
            rgb_frame=rgb_frame,
            rgb_gesture=rgb_gesture,
            rgb_confidence=rgb_conf,
            rgb_stable=rgb_stable,
            rgb_fps=rgb_fps,
            rgb_elapsed_ms=rgb_elapsed_ms,
            game_mode=self._game_mode,
        )

        # Forward to active output
        if self.active_output:
            self.active_output.process(self._result)

    def render(self) -> np.ndarray:
        if self.active_output:
            return self.active_output.render()
        return np.zeros((480, 640, 3), dtype=np.uint8)

    # ------------------------------------------------------------------
    # Key handling
    # ------------------------------------------------------------------

    def handle_key(self, key: int) -> bool:
        if key == ord('m'):
            # Toggle game mode
            self._game_mode = "mirror" if self._game_mode == "battle" else "battle"
            print(f"[GESTURE] Game mode: {self._game_mode.upper()}")
            return True

        if key == ord(' '):
            self._tracking_enabled = not self._tracking_enabled
            if self.active_output:
                self.active_output.on_tracking_changed(self._tracking_enabled)
            state = "ON" if self._tracking_enabled else "OFF"
            print(f"[GESTURE] Tracking {state}")
            return True

        return super().handle_key(key)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def game_mode(self) -> str:
        return self._game_mode

    @property
    def tracking_enabled(self) -> bool:
        return self._tracking_enabled

    @tracking_enabled.setter
    def tracking_enabled(self, value: bool) -> None:
        self._tracking_enabled = value

    @property
    def result(self) -> GestureResult:
        return self._result

    @property
    def dvs_inference(self):
        return self._dvs_inference
