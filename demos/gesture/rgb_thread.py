"""Background RGB gesture inference thread.

Reads RGB frames from the USB camera, runs MediaPipe inference, and shares
the latest result snapshot for the main thread to consume via get_latest().
"""

import threading
import time
import traceback
from collections import deque
from typing import Optional, Tuple

import numpy as np

from app.core.inference.common import MajorityVoter
from app.core.inference.rgb_gesture import MediaPipeGestureInference


class RGBGestureThread:
    """Background thread: RGB camera -> inference -> vote -> snapshot.

    Args:
        camera_mgr: CameraManager instance (uses ``read_rgb_frame()``).
        inference: MediaPipeGestureInference instance.
        voter: MajorityVoter for this thread's predictions.
    """

    def __init__(
        self,
        camera_mgr,
        inference: MediaPipeGestureInference,
        voter: MajorityVoter,
    ):
        self._camera_mgr = camera_mgr
        self._inference = inference
        self._voter = voter

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Latest snapshot (protected by _lock)
        self._frame: Optional[np.ndarray] = None
        self._gesture: str = "none"
        self._confidence: float = 0.0
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
        Optional[np.ndarray], str, float, str, float, float
    ]:
        """Return latest snapshot: (frame, gesture, conf, stable, fps, elapsed_ms)."""
        with self._lock:
            return (
                self._frame, self._gesture, self._confidence,
                self._stable, self._fps, self._elapsed_ms,
            )

    def _run(self) -> None:
        """Thread main loop."""
        fps_window = 30
        fps_timestamps: deque[float] = deque(maxlen=fps_window)

        while not self._stop_event.is_set():
            frame = self._camera_mgr.read_rgb_frame()
            if frame is None:
                time.sleep(0.001)
                continue

            # Inference
            try:
                gesture, conf, elapsed = self._inference.predict(frame)
            except Exception as e:
                print(f"[RGB_GEST] Inference error: {e}")
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

            # Update snapshot
            with self._lock:
                self._frame = frame
                self._gesture = gesture
                self._confidence = conf
                self._stable = stable
                self._fps = fps
                self._elapsed_ms = elapsed * 1000
