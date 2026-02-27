"""Background DVS gesture inference thread.

Reads DVS frames at ~200fps, runs inference, and shares the latest
result snapshot for the main thread to consume via get_latest().
"""

import threading
import time
import traceback
from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np

from app.config import DVS_WIDTH, DVS_HEIGHT, DVS_NORMALIZE_CENTER, DVS_NORMALIZE_STEEPNESS
from app.core.inference.common import MajorityVoter
from app.core.inference.dvs_gesture import DVSGestureInference


class DVSGestureThread:
    """Background thread: DVS camera -> inference -> vote -> snapshot.

    Args:
        xe_cam: XenReal camera module (uses ``g_cap.XeGetFrame`` for raw DVS data).
        inference: DVSGestureInference instance.
        voter: MajorityVoter for this thread's predictions.
        scale: Display scale factor for DVS preview.
        bit_depth: DVS bit depth (4 = pure DVS mode).
    """

    def __init__(
        self,
        xe_cam,
        inference: DVSGestureInference,
        voter: MajorityVoter,
        scale: int = 3,
        bit_depth: int = 4,
    ):
        self._xe_cam = xe_cam
        self._inference = inference
        self._voter = voter
        self._scale = scale
        self._bit_depth = bit_depth

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Latest snapshot (protected by _lock)
        self._display: Optional[np.ndarray] = None
        self._gesture: str = "none"
        self._confidence: float = 0.0
        self._probs: Optional[np.ndarray] = None
        self._stable: str = "none"
        self._fps: float = 0.0
        self._elapsed_ms: float = 0.0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

    def get_latest(self) -> Tuple[
        Optional[np.ndarray], str, float, Optional[np.ndarray], str, float, float
    ]:
        """Return latest snapshot: (display, gesture, conf, probs, stable, fps, elapsed_ms)."""
        with self._lock:
            return (
                self._display, self._gesture, self._confidence,
                self._probs, self._stable, self._fps, self._elapsed_ms,
            )

    def _run(self) -> None:
        """Thread main loop."""
        from cv2_like_xe_sdk import dvs_normalize_sigmoid

        fps_window = 30
        fps_timestamps: deque[float] = deque(maxlen=fps_window)

        while not self._stop_event.is_set():
            # Read raw DVS frame (bypass normalized API to get 0-15 data)
            dvs_raw, _ = self._xe_cam.g_cap.XeGetFrame(
                self._xe_cam.g_xereal_mode,
                self._xe_cam.g_xereal_bit_depth,
            )
            if dvs_raw is None:
                time.sleep(0.001)
                continue
            frame = dvs_raw

            try:
                dvs_img = frame.reshape((DVS_HEIGHT, DVS_WIDTH))
                dvs_img = dvs_normalize_sigmoid(
                    dvs_img, self._bit_depth,
                    center=DVS_NORMALIZE_CENTER,
                    steepness=DVS_NORMALIZE_STEEPNESS,
                )
                if dvs_img is None:
                    continue
            except Exception as e:
                print(f"[DVS_GEST] Frame error: {e}")
                continue

            # Inference
            try:
                gesture, conf, probs, elapsed = self._inference.predict(dvs_img)
            except Exception as e:
                print(f"[DVS_GEST] Inference error: {e}")
                traceback.print_exc()
                continue

            # Vote
            now = time.perf_counter()
            self._voter.push(gesture, conf, now)
            stable = self._voter.majority()

            # FPS
            fps_timestamps.append(now)
            if len(fps_timestamps) >= 2:
                span = fps_timestamps[-1] - fps_timestamps[0]
                fps = (len(fps_timestamps) - 1) / span if span > 0 else 0.0
            else:
                fps = 0.0

            # Build display image
            dvs_bgr = cv2.cvtColor(dvs_img, cv2.COLOR_GRAY2BGR)
            display = cv2.resize(
                dvs_bgr,
                (DVS_WIDTH * self._scale, DVS_HEIGHT * self._scale),
                interpolation=cv2.INTER_NEAREST,
            )

            # Update snapshot
            with self._lock:
                self._display = display
                self._gesture = gesture
                self._confidence = conf
                self._probs = probs
                self._stable = stable
                self._fps = fps
                self._elapsed_ms = elapsed * 1000
