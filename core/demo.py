"""Abstract base classes for Demo and OutputMode."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Optional

import numpy as np


class OutputModeType(Enum):
    GUI = "gui"
    PHYS_DVS = "phys_dvs"
    PHYS_RGB = "phys_rgb"


class OutputMode(ABC):
    """Base class for demo output modes (GUI display, physical arm, etc.)."""

    @abstractmethod
    def activate(self) -> None:
        """Called when this output mode becomes active."""

    @abstractmethod
    def deactivate(self) -> None:
        """Called when switching away from this output mode."""

    @abstractmethod
    def process(self, result) -> None:
        """Process tracking/detection result each frame."""

    @abstractmethod
    def render(self) -> np.ndarray:
        """Render output as BGR image for display."""

    def on_tracking_changed(self, enabled: bool) -> None:
        """Called when tracking is toggled. Override to propagate state."""

    def mouse_callback(self, x: int, y: int) -> bool:
        """Handle mouse click at content-local coordinates. Return True if consumed."""
        return False

    def handle_key(self, key: int) -> bool:
        """Handle keypress. Return True if consumed."""
        return False


class Demo(ABC):
    """Base class for a demo tab (calibration, tracking, etc.)."""

    def __init__(self, name: str):
        self.name = name
        self._outputs: Dict[OutputModeType, OutputMode] = {}
        self._active_output_type: Optional[OutputModeType] = None

    def register_output(self, mode: OutputModeType, output: OutputMode) -> None:
        """Register an output mode."""
        self._outputs[mode] = output

    @property
    def active_output(self) -> Optional[OutputMode]:
        """Currently active output mode, or None."""
        if self._active_output_type is None:
            return None
        return self._outputs.get(self._active_output_type)

    def switch_output(self, mode: OutputModeType) -> None:
        """Switch to a different output mode."""
        if mode == self._active_output_type:
            return
        if mode not in self._outputs:
            return
        if self.active_output:
            self.active_output.deactivate()
        self._active_output_type = mode
        self.active_output.activate()

    @abstractmethod
    def activate(self, camera_mgr) -> None:
        """Called when this demo tab becomes active."""

    @abstractmethod
    def deactivate(self) -> None:
        """Called when switching away from this demo tab."""

    @abstractmethod
    def process_frame(self, camera_mgr) -> None:
        """Process one frame (read cameras, run detection, etc.)."""

    @abstractmethod
    def render(self) -> np.ndarray:
        """Render demo output as BGR image."""

    def handle_key(self, key: int) -> bool:
        """Handle keypress. Delegates to active output first."""
        if self.active_output and self.active_output.handle_key(key):
            return True
        return False
