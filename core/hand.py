"""HandBridge + HandThread — LinkerHand control infrastructure.

Mirrors the CommandBridge + ArmThread pattern in arm.py.

Architecture:
    Producer (GesturePhysDVSOutput / GesturePhysRGBOutput)
      -> HandBridge.put_gesture("paper")
      -> HandThread._consume_loop() -> LinkerHandApi.finger_move()

    Tab / mode switch:
      -> hand_bridge.put_neutral()  -- drain queue + neutral pose

    Program exit:
      -> hand_thread.stop() -> _cleanup() -> neutral + release
"""

import queue
import sys
import threading
import time
from typing import Optional

from app.core.inference.common import POSES_O6

# Sentinel values
_CMD_NEUTRAL = "__NEUTRAL__"


class HandBridge:
    """Thread-safe gesture command queue for LinkerHand."""

    def __init__(self, maxsize: int = 100):
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)

    def put_gesture(self, gesture: str) -> None:
        """Producer: enqueue a gesture command."""
        if gesture not in POSES_O6:
            return
        try:
            self._queue.put_nowait(gesture)
        except queue.Full:
            pass

    def put_neutral(self) -> None:
        """Drain pending commands and enqueue neutral pose."""
        self.clear()
        try:
            self._queue.put_nowait(_CMD_NEUTRAL)
        except queue.Full:
            pass

    def clear(self) -> int:
        """Drain and discard all pending commands."""
        count = 0
        while True:
            try:
                self._queue.get_nowait()
                count += 1
            except queue.Empty:
                break
        return count

    def get(self, timeout: float = 0.1):
        """Consumer: dequeue next gesture (str or sentinel)."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def pending(self) -> int:
        return self._queue.qsize()


class HandThread:
    """Background consumer — sends gesture poses to LinkerHand via CAN.

    Args:
        bridge: HandBridge to consume from.
        can_name: CAN interface name (e.g. ``"can4"``).
        hand_type: ``"left"`` or ``"right"``.
        hand_joint: Hand model (default ``"O6"``).
        hand_sdk_path: Path to directory containing ``LinkerHand/`` package.
        min_cmd_interval: Minimum seconds between consecutive finger_move calls.
    """

    def __init__(
        self,
        bridge: HandBridge,
        can_name: str,
        hand_type: str = "right",
        hand_joint: str = "O6",
        hand_sdk_path: Optional[str] = None,
        min_cmd_interval: float = 0.1,
    ):
        self._bridge = bridge
        self._can_name = can_name
        self._hand_type = hand_type
        self._hand_joint = hand_joint
        self._hand_sdk_path = hand_sdk_path
        self._min_cmd_interval = min_cmd_interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Public state (readable from main thread)
        self.is_ready = threading.Event()
        self.is_running = False
        self.current_gesture: str = "none"
        self.move_count: int = 0
        self.error: Optional[str] = None

        # Internal
        self._hand = None
        self._last_cmd_time: float = 0.0

    def start(self) -> None:
        """Launch daemon thread."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal stop."""
        self._stop_event.set()

    def join(self, timeout: float = 10.0) -> None:
        """Wait for completion."""
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    # ------------------------------------------------------------------
    # Thread internals
    # ------------------------------------------------------------------

    def _run(self) -> None:
        try:
            self._init_hand()
            self.is_ready.set()
            self.is_running = True
            self._consume_loop()
        except Exception as e:
            self.error = str(e)
            print(f"[HAND ERROR] {e}")
        finally:
            self._cleanup()
            self.is_running = False

    def _init_hand(self) -> None:
        """Initialize LinkerHand SDK."""
        if self._hand_sdk_path:
            if self._hand_sdk_path not in sys.path:
                sys.path.insert(0, self._hand_sdk_path)

        from LinkerHand.linker_hand_api import LinkerHandApi

        print(f"[HAND] Initializing: joint={self._hand_joint}, "
              f"type={self._hand_type}, can={self._can_name}")
        self._hand = LinkerHandApi(
            hand_joint=self._hand_joint,
            hand_type=self._hand_type,
            can=self._can_name,
        )
        self._hand.set_speed(speed=[255, 255, 255, 255, 255, 255])
        print("[HAND] Ready!")

    def _consume_loop(self) -> None:
        while not self._stop_event.is_set():
            cmd = self._bridge.get(timeout=0.1)
            if cmd is None:
                continue

            if cmd is _CMD_NEUTRAL:
                self._move_to("none")
                continue

            # Regular gesture command
            self._move_to(cmd)

    def _move_to(self, gesture: str) -> None:
        """Send finger_move command with dedup + cooldown."""
        if gesture == self.current_gesture:
            return
        if gesture not in POSES_O6:
            return

        now = time.perf_counter()
        if (now - self._last_cmd_time) < self._min_cmd_interval:
            return

        pose = POSES_O6[gesture]
        try:
            self._hand.finger_move(pose=pose)
            self.current_gesture = gesture
            self.move_count += 1
            self._last_cmd_time = now
        except Exception as e:
            print(f"[HAND] finger_move failed: {e}")

    def _cleanup(self) -> None:
        """Move to neutral on shutdown and release C++ resources."""
        if self._hand is not None:
            try:
                self._hand.finger_move(pose=POSES_O6["none"])
            except Exception as e:
                print(f"[HAND] Cleanup error: {e}")
            # Release C++ object explicitly to avoid destructor issues
            # during interpreter shutdown.
            self._hand = None
