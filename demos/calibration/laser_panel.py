"""Laser Calibration panel — record video, analyze, save profile.

State machine:
    IDLE -> RECORDING -> ANALYZING -> IDLE (profile saved)

Integrates LaserTracker.calibrate_profile_from_video() into the
Calibration tab so users can calibrate without running a separate script.
"""

import os
import tempfile
import threading
import time
from enum import Enum, auto
from typing import Optional

import cv2
import numpy as np

from app.config import DEFAULT_LASER_PROFILE
from app.core.display import draw_hint_bar, resize_to_height

# Recording parameters
_RECORD_DURATION = 30.0  # seconds
_RECORD_FPS = 30
_RECORD_CODEC = "MJPG"

# Display
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_WHITE = (220, 220, 220)
_RED = (0, 0, 255)
_GREEN = (0, 200, 80)
_ORANGE = (0, 140, 255)
_GRAY = (130, 130, 130)


class _State(Enum):
    IDLE = auto()
    RECORDING = auto()
    ANALYZING = auto()


class LaserCalibrationPanel:
    """Self-contained panel for laser profile calibration."""

    def __init__(self, camera_mgr, args, cal_store=None):
        self._camera_mgr = camera_mgr
        self._args = args
        self._cal_store = cal_store

        self._state = _State.IDLE
        self._profile_loaded = os.path.isfile(DEFAULT_LASER_PROFILE)

        # Detect overlay (lazy)
        self._tracker = None
        self._init_tracker()

        # Recording state
        self._writer: Optional[cv2.VideoWriter] = None
        self._temp_path: Optional[str] = None
        self._rec_start: float = 0.0
        self._rec_frame_size: tuple = (0, 0)

        # Last rendered width (for analyzing canvas fallback)
        self._last_w: int = 0

        # Analyzing state
        self._analyze_thread: Optional[threading.Thread] = None
        self._analyze_progress: int = 0  # 0-100
        self._analyze_total: int = 1
        self._analyze_current: int = 0
        self._analyze_error: Optional[str] = None

    def _init_tracker(self):
        """Load tracker from profile if available."""
        if not os.path.isfile(DEFAULT_LASER_PROFILE):
            self._tracker = None
            self._profile_loaded = False
            return
        try:
            from laser_tracker import LaserProfile, LaserTracker
            profile = LaserProfile.load(DEFAULT_LASER_PROFILE)
            self._tracker = LaserTracker.from_profile(
                profile, roi=self._rgb_roi())
            self._profile_loaded = True
        except Exception as e:
            print(f"[LASER_CAL] Failed to load profile: {e}")
            self._tracker = None
            self._profile_loaded = False

    def _rgb_roi(self):
        """Return RGB quad ROI as (x1, y1, x2, y2) or None."""
        if self._cal_store is None:
            return None
        quad = self._cal_store.rgb_quad
        if quad is None:
            return None
        return quad.as_xyxy()

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self, height: int) -> np.ndarray:
        """Render the panel as a BGR image (width determined by aspect ratio)."""
        if self._state == _State.RECORDING:
            return self._render_recording(height)
        if self._state == _State.ANALYZING:
            return self._render_analyzing(height)
        return self._render_idle(height)

    def _render_idle(self, height: int) -> np.ndarray:
        """IDLE: RGB preview + optional detect overlay."""
        rgb = self._camera_mgr.read_rgb_frame()
        if rgb is not None:
            display = rgb.copy()
            # Detect overlay if profile loaded
            if self._tracker is not None:
                target = self._tracker.detect(display)
                if target:
                    cx, cy = int(target.cx), int(target.cy)
                    cv2.circle(display, (cx, cy), 12, _GREEN, 2)
                    cv2.drawMarker(display, (cx, cy), _GREEN,
                                   cv2.MARKER_CROSS, 20, 1)
            canvas = resize_to_height(display, height)
            self._last_w = canvas.shape[1]
        else:
            width = self._last_w or int(height * 4 / 3)
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(canvas, "RGB: no frame", (10, 30),
                        _FONT, 0.6, _RED, 1)

        # Hint bar
        profile_status = "loaded" if self._profile_loaded else "none"
        draw_hint_bar(canvas, [
            f"[R] Record  |  Profile: {profile_status}",
        ])
        return canvas

    def _render_recording(self, height: int) -> np.ndarray:
        """RECORDING: RGB preview + REC indicator + countdown."""
        rgb = self._camera_mgr.read_rgb_frame()
        if rgb is not None:
            # Write frame to video (unrotated raw is already rotated by camera_mgr)
            self._write_frame(rgb)
            canvas = resize_to_height(rgb.copy(), height)
            self._last_w = canvas.shape[1]
        else:
            width = self._last_w or int(height * 4 / 3)
            canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # Elapsed / remaining
        elapsed = time.monotonic() - self._rec_start
        remaining = max(0, _RECORD_DURATION - elapsed)

        # Auto-stop
        if remaining <= 0:
            self._stop_recording()
            self._start_analyzing()
            return canvas

        # REC indicator (blinking red dot)
        blink = int(elapsed * 3) % 2 == 0
        if blink:
            cv2.circle(canvas, (30, 30), 10, _RED, -1)
        cv2.putText(canvas, f"REC {remaining:.1f}s", (48, 38),
                    _FONT, 0.7, _RED, 2)

        draw_hint_bar(canvas, [
            "Recording... [Esc] cancel",
        ])
        return canvas

    def _render_analyzing(self, height: int) -> np.ndarray:
        """ANALYZING: progress bar + percentage."""
        width = self._last_w or int(height * 4 / 3)
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # Check if done
        if self._analyze_thread is not None and not self._analyze_thread.is_alive():
            self._analyze_thread = None
            if self._analyze_error:
                print(f"[LASER_CAL] Analysis failed: {self._analyze_error}")
            else:
                print("[LASER_CAL] Profile saved to "
                      f"{DEFAULT_LASER_PROFILE}")
                self._init_tracker()
            self._state = _State.IDLE
            return self._render_idle(height)

        # Progress bar
        pct = self._analyze_progress
        bar_w = min(width - 80, 400)
        bar_h = 24
        bx = (width - bar_w) // 2
        by = height // 2 - bar_h // 2

        cv2.rectangle(canvas, (bx, by), (bx + bar_w, by + bar_h),
                      (60, 60, 60), -1)
        fill = int(bar_w * pct / 100)
        if fill > 0:
            cv2.rectangle(canvas, (bx, by), (bx + fill, by + bar_h),
                          _ORANGE, -1)
        cv2.rectangle(canvas, (bx, by), (bx + bar_w, by + bar_h),
                      (100, 100, 100), 1)

        pct_text = f"{pct}%"
        ts = cv2.getTextSize(pct_text, _FONT, 0.7, 2)[0]
        tx = (width - ts[0]) // 2
        ty = by - 12
        cv2.putText(canvas, pct_text, (tx, ty), _FONT, 0.7, _WHITE, 2)

        cv2.putText(canvas, "Analyzing calibration video...",
                    (bx, by + bar_h + 30), _FONT, 0.5, _GRAY, 1)

        draw_hint_bar(canvas, ["Analyzing..."])
        return canvas

    # ------------------------------------------------------------------
    # Recording helpers
    # ------------------------------------------------------------------

    def _start_recording(self):
        """Begin recording to a temp .avi file."""
        rgb = self._camera_mgr.read_rgb_frame()
        if rgb is None:
            print("[LASER_CAL] Cannot start recording: no RGB frame")
            return

        h, w = rgb.shape[:2]
        self._rec_frame_size = (w, h)

        fd, path = tempfile.mkstemp(suffix=".avi", prefix="laser_cal_")
        os.close(fd)
        self._temp_path = path

        fourcc = cv2.VideoWriter_fourcc(*_RECORD_CODEC)
        self._writer = cv2.VideoWriter(path, fourcc, _RECORD_FPS,
                                       self._rec_frame_size)
        if not self._writer.isOpened():
            print(f"[LASER_CAL] Failed to open VideoWriter: {path}")
            self._writer = None
            return

        self._rec_start = time.monotonic()
        self._state = _State.RECORDING
        # Write first frame
        self._write_frame(rgb)
        print(f"[LASER_CAL] Recording started -> {path}")

    def _write_frame(self, frame: np.ndarray):
        """Write a frame to the active VideoWriter."""
        if self._writer is None:
            return
        # Ensure frame matches expected size
        fh, fw = frame.shape[:2]
        ew, eh = self._rec_frame_size
        if (fw, fh) != (ew, eh):
            frame = cv2.resize(frame, (ew, eh))
        self._writer.write(frame)

    def _stop_recording(self):
        """Stop recording and release VideoWriter."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        print("[LASER_CAL] Recording stopped")

    def _cancel_recording(self):
        """Cancel recording, delete temp file, return to IDLE."""
        self._stop_recording()
        self._cleanup_temp()
        self._state = _State.IDLE
        print("[LASER_CAL] Recording cancelled")

    def _cleanup_temp(self):
        """Remove temporary video file if it exists."""
        if self._temp_path and os.path.isfile(self._temp_path):
            try:
                os.remove(self._temp_path)
            except OSError as e:
                print(f"[LASER_CAL] Failed to delete temp: {e}")
        self._temp_path = None

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def _start_analyzing(self):
        """Launch background thread to analyze the recorded video."""
        if not self._temp_path or not os.path.isfile(self._temp_path):
            print("[LASER_CAL] No recording to analyze")
            self._state = _State.IDLE
            return

        self._state = _State.ANALYZING
        self._analyze_progress = 0
        self._analyze_error = None

        self._analyze_thread = threading.Thread(
            target=self._analyze_worker, daemon=True,
        )
        self._analyze_thread.start()

    def _analyze_worker(self):
        """Background worker: run calibrate_profile_from_video()."""
        try:
            from laser_tracker import LaserTracker

            def _progress(current, total):
                self._analyze_current = current
                self._analyze_total = max(1, total)
                self._analyze_progress = min(
                    99, int(100 * current / self._analyze_total))

            profile, stats = LaserTracker.calibrate_profile_from_video(
                self._temp_path,
                roi=self._rgb_roi(),
                auto_quad=False,
                rotate=0,  # frames already rotated by camera_mgr
                progress_callback=_progress,
            )

            # Save profile
            profile.save(DEFAULT_LASER_PROFILE)
            self._analyze_progress = 100

            det_rate = stats.get("detection_rate", 0)
            print(f"[LASER_CAL] Analysis done: detection_rate="
                  f"{det_rate:.0%}, hue_filter="
                  f"{profile.use_hue_filter}")

        except Exception as e:
            self._analyze_error = str(e)
        finally:
            self._cleanup_temp()

    # ------------------------------------------------------------------
    # Key handling
    # ------------------------------------------------------------------

    def handle_key(self, key: int) -> bool:
        """Handle keyboard input. Return True if consumed."""
        if self._state == _State.IDLE:
            if key in (ord('r'), ord('R')):
                self._start_recording()
                return True

        elif self._state == _State.RECORDING:
            if key == 27:  # Esc
                self._cancel_recording()
                return True

        return False

    # ------------------------------------------------------------------
    # Mouse handling
    # ------------------------------------------------------------------

    def mouse_callback(self, event: int, x: int, y: int,
                       flags: int, param) -> None:
        """Handle mouse events (currently unused)."""
        pass

