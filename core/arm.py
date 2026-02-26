"""CommandBridge + ArmThread — arm control infrastructure.

Moved from examples/ex17/main_dvs_drawing.py into the app package so that
arm lifecycle (safe-home on tab switch, cleanup on exit) lives next to the
GUI code that triggers it.

Architecture:
    Producer (DVSDrawingThread / PhysRGBOutput / ArmCalibrationPanel)
      -> CommandBridge.put(write, x, y)
      -> ArmThread._consume_loop() -> DrawingController.move()

    Tab / mode switch:
      -> bridge.put_safe_home()  -- drain queue + sentinel
      -> ArmThread._go_safe_home() -- pen_up, lift, home joints (motors stay on)

    Program exit:
      -> arm_thread.stop() -> _cleanup() -> safe_disable + motor off
"""

import queue
import threading
import time
from typing import Optional

# Sentinel values for special commands
_CMD_SAFE_HOME = "__SAFE_HOME__"
_CMD_PEN_UP = "__PEN_UP__"
_CMD_PEN_DOWN = "__PEN_DOWN__"


class CommandBridge:
    """Thread-safe FIFO command bridge. Every command is preserved."""

    def __init__(self, maxsize: int = 5000):
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)

    def put(self, write: bool, x: float, y: float) -> None:
        """Producer: enqueue one command per frame."""
        try:
            self._queue.put_nowait((write, x, y))
        except queue.Full:
            pass  # safety valve -- drop if queue impossibly full

    def put_safe_home(self) -> None:
        """Drain pending commands and enqueue a safe-home sentinel.

        Called by output modes / calibration demo on deactivate so that
        the arm returns to SAFE_HOME_JOINTS without processing stale
        draw commands.
        """
        self.clear()
        try:
            self._queue.put_nowait(_CMD_SAFE_HOME)
        except queue.Full:
            pass

    def put_pen_up(self) -> None:
        """Enqueue pen-up (lift at current XY)."""
        self._queue.put_nowait(_CMD_PEN_UP)

    def put_pen_down(self) -> None:
        """Enqueue pen-down (lower at current XY)."""
        self._queue.put_nowait(_CMD_PEN_DOWN)

    def clear(self) -> int:
        """Drain and discard all pending commands. Return count discarded."""
        count = 0
        while True:
            try:
                self._queue.get_nowait()
                count += 1
            except queue.Empty:
                break
        return count

    def get(self, timeout: float = 0.1):
        """Consumer: dequeue next command (tuple or sentinel string)."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def pending(self) -> int:
        return self._queue.qsize()


class ArmThread:
    """Background thread that consumes commands from CommandBridge."""

    def __init__(self, bridge: CommandBridge, can_name: str, speed: float):
        self._bridge = bridge
        self._can_name = can_name
        self._speed = speed
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Public state (readable from main thread for GUI)
        self.is_ready = threading.Event()
        self.is_running = False
        self.error: Optional[str] = None
        self.move_count = 0
        self.fail_count = 0

        # Internal references (set during _init_arm)
        self._conn = None
        self._drawer = None

        # Safe-home flag: True when arm is at SAFE_HOME_JOINTS
        self._at_home: bool = False

    def start(self) -> None:
        """Launch daemon thread."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal stop."""
        self._stop_event.set()

    def join(self, timeout: float = 20.0) -> None:
        """Wait for completion."""
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    @property
    def piper(self):
        """Access C_PiperInterface_V2 (available after is_ready)."""
        return self._conn.piper if self._conn is not None else None

    @property
    def at_home(self) -> bool:
        """Whether arm is at safe home position."""
        return self._at_home

    @property
    def pen_down(self) -> bool:
        """Whether pen is currently down (drawing)."""
        if self._drawer is None:
            return False
        return self._drawer.is_writing()

    # ------------------------------------------------------------------
    # Thread internals
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Thread entry point."""
        try:
            self._init_arm()
            self._at_home = False
            self.is_ready.set()
            self.is_running = True
            self._consume_loop()
        except Exception as e:
            self.error = str(e)
            print(f"[ARM ERROR] {e}")
        finally:
            self._cleanup()
            self.is_running = False

    def _init_arm(self) -> None:
        """Initialize arm hardware."""
        from piper_demo import PiperConnection, MotionController, JointReader
        from drawing import DrawingController, DrawingConfig

        print(f"[ARM] Connecting to {self._can_name}...")
        self._conn = PiperConnection(can_name=self._can_name)
        self._conn.connect()

        print("[ARM] Enabling arm...")
        self._conn.enable(go_home=False)
        time.sleep(1)

        motion = MotionController(self._conn.piper)
        reader = JointReader(self._conn.piper)

        config = DrawingConfig(
            draw_speed=self._speed,
            move_speed=self._speed,
        )
        self._drawer = DrawingController(motion, reader, config)

        # Move to center position (pen up)
        print("[ARM] Moving to center (0.5, 0.5)...")
        ok = self._drawer.move(False, 0.5, 0.5)
        if not ok:
            raise RuntimeError("Cannot reach center position (0.5, 0.5)")
        print("[ARM] Ready!")

    def _consume_loop(self) -> None:
        """Main consumer loop -- processes commands from bridge."""
        while not self._stop_event.is_set():
            cmd = self._bridge.get(timeout=0.1)
            if cmd is None:
                continue

            # Handle sentinel commands
            if cmd is _CMD_SAFE_HOME:
                self._go_safe_home()
                continue
            if cmd is _CMD_PEN_UP:
                self._drawer.pen_up()
                continue
            if cmd is _CMD_PEN_DOWN:
                self._drawer.pen_down()
                continue

            write, x, y = cmd
            self._at_home = False
            ok = self._drawer.move(write, x, y)
            self.move_count += 1
            if not ok:
                self.fail_count += 1

    def _go_safe_home(self) -> None:
        """Return arm to SAFE_HOME_JOINTS (pen_up -> lift -> home joints).

        Same sequence as DrawingController.safe_disable() but does NOT
        disable motors -- arm stays powered at home position until the
        next activate() or program exit.
        """
        if self._at_home or self._drawer is None:
            return

        from drawing import SAFE_HOME_JOINTS

        d = self._drawer
        # 1. Pen up
        if not d.pen_up():
            print("[ARM] Safe home aborted: pen_up failed")
            return
        # 2. Lift 10cm above safe_z
        if not d._move_to_xyz(
            d._x, d._y, d.config.safe_z + 0.10,
            speed=d.config.move_speed, wait=True,
        ):
            print("[ARM] Safe home aborted: lift failed")
            return
        # 3. Move to home joints (J1-J6 only; J7 gripper untouched)
        d.motion.move_joint(SAFE_HOME_JOINTS, speed_factor=d.config.move_speed)
        # 4. Wait until position reached
        d.reader.wait_for_position(
            SAFE_HOME_JOINTS, tolerance_rad=0.035, timeout_sec=10.0,
        )
        # 5. Reset drawing state (do NOT disable motors)
        d._current_joints = list(SAFE_HOME_JOINTS)
        self._at_home = True
        print("[ARM] Safe home reached (motors still enabled)")

    def _cleanup(self) -> None:
        """Safe shutdown of arm hardware and release C++ resources."""
        if self._drawer is not None:
            try:
                self._drawer.safe_disable()
            except Exception:
                pass
            self._drawer = None
        if self._conn is not None:
            # disconnect() calls safe_disable() internally then releases
            # the C++ piper object, avoiding destructor issues during
            # interpreter shutdown.
            try:
                self._conn.disconnect()
            except Exception:
                pass
            self._conn = None
