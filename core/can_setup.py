"""Auto-detect and activate USB-CAN adapter + LinkerHand at startup.

Scans CAN interfaces via ``ip`` + ``ethtool``, classifies them as:
- Orin built-in (bus-info ending with ``.mttcan``) -- skipped
- USB-CAN adapter (has bus-info, not ``.mttcan``) -- arm
- No bus-info (ethtool error) -- LinkerHand

Interface names are **not** renamed; the detected name is used as-is
because CAN index numbers are not stable across reboots.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Dict


@dataclass
class CanSetupResult:
    """Result of :func:`setup_all_can`."""

    arm_can: str | None = None
    arm_error: str | None = None
    hand_can: str | None = None
    hand_error: str | None = None


# ---------------------------------------------------------------------------
# Shell helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a shell command without sudo."""
    return subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=check)


# ---------------------------------------------------------------------------
# Interface introspection
# ---------------------------------------------------------------------------

def _list_can_interfaces() -> list[str]:
    """Return names of all CAN interfaces currently known to the kernel."""
    result = _run(["ip", "-br", "link", "show", "type", "can"], check=False)
    interfaces: list[str] = []
    for line in result.stdout.strip().splitlines():
        parts = line.split()
        if parts:
            interfaces.append(parts[0])
    return interfaces


def _get_bus_info(iface: str) -> str | None:
    """Return ethtool bus-info for *iface*, or ``None`` on failure/empty."""
    result = _run(["ethtool", "-i", iface], check=False)
    for line in result.stdout.splitlines():
        if line.startswith("bus-info:"):
            val = line.split(":", 1)[1].strip()
            return val or None  # empty string → None
    return None


def _get_current_bitrate(iface: str) -> int | None:
    """Return configured bitrate for *iface*, or ``None`` if unavailable."""
    result = _run(["ip", "-details", "link", "show", iface], check=False)
    for line in result.stdout.splitlines():
        line = line.strip()
        if "bitrate" in line:
            parts = line.split()
            for i, tok in enumerate(parts):
                if tok == "bitrate" and i + 1 < len(parts):
                    try:
                        return int(parts[i + 1])
                    except ValueError:
                        pass
    return None


def _is_interface_up(iface: str) -> bool:
    """Check if *iface* is UP."""
    result = _run(["ip", "link", "show", iface], check=False)
    return "UP" in result.stdout and "LOWER_UP" in result.stdout


# ---------------------------------------------------------------------------
# Interface detection
# ---------------------------------------------------------------------------

def _find_usb_can(usb_port: str | None) -> tuple[str, str]:
    """Find the USB-CAN interface (arm).

    Returns:
        ``(interface_name, bus_info)``

    Raises:
        RuntimeError: when no or multiple USB-CAN adapters are found.
    """
    ifaces = _list_can_interfaces()
    usb_map: Dict[str, str] = {}

    for iface in ifaces:
        bus = _get_bus_info(iface)
        if bus is None:
            continue
        if bus.endswith(".mttcan"):
            continue
        usb_map[iface] = bus

    if usb_port:
        for iface, bus in usb_map.items():
            if bus == usb_port:
                return iface, bus
        raise RuntimeError(
            f"No CAN interface with USB port '{usb_port}'. "
            f"Available USB-CAN: {usb_map or '(none)'}"
        )

    if len(usb_map) == 0:
        raise RuntimeError("No USB-CAN adapter detected")
    if len(usb_map) > 1:
        desc = ", ".join(f"{k}={v}" for k, v in usb_map.items())
        raise RuntimeError(
            f"Multiple USB-CAN adapters found ({desc}). "
            "Use --usb-port to specify which one."
        )

    return next(iter(usb_map.items()))


def _find_linkerhand() -> str:
    """Find the LinkerHand CAN interface (no bus-info, not built-in).

    Returns:
        Interface name (e.g. ``"can4"``).

    Raises:
        RuntimeError: when zero or multiple candidates are found.
    """
    ifaces = _list_can_interfaces()
    candidates: list[str] = []

    for iface in ifaces:
        bus = _get_bus_info(iface)
        if bus is not None:
            # Has bus-info → either built-in (.mttcan) or USB-CAN adapter
            continue
        # No bus-info → LinkerHand candidate
        candidates.append(iface)

    if len(candidates) == 0:
        raise RuntimeError("No LinkerHand CAN interface detected (no ethtool-error interfaces)")
    if len(candidates) > 1:
        raise RuntimeError(
            f"Multiple LinkerHand candidates: {candidates}. "
            "Cannot auto-detect — manual configuration required."
        )

    return candidates[0]


# ---------------------------------------------------------------------------
# Interface activation
# ---------------------------------------------------------------------------

def _activate_interface(iface: str, bitrate: int, label: str, bus_info: str | None = None) -> str:
    """Bring up a CAN interface with the requested bitrate (idempotent).

    Args:
        iface: Kernel interface name (e.g. ``"can5"``).
        bitrate: CAN bus bitrate.
        label: Human-readable label for log messages (e.g. ``"arm"``).
        bus_info: Optional bus-info string for logging.

    Returns:
        The interface name (unchanged — no rename).
    """
    # Already up with correct bitrate → skip
    if _is_interface_up(iface) and _get_current_bitrate(iface) == bitrate:
        print(f"[CAN] {label}: {iface} already active (bitrate={bitrate})")
        return iface

    # Bring down → set bitrate → bring up
    _run(["ip", "link", "set", iface, "down"], check=False)
    _run(["ip", "link", "set", iface, "type", "can", "bitrate", str(bitrate)])
    _run(["ip", "link", "set", iface, "up"])

    info_str = f", bus-info={bus_info}" if bus_info else ""
    print(f"[CAN] {label}: activated {iface} (bitrate={bitrate}{info_str})")
    return iface


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def activate_can(bitrate: int = 1_000_000, usb_port: str | None = None) -> str:
    """Detect and activate the USB-CAN adapter (arm).

    Returns:
        Final interface name.
    """
    iface, bus_info = _find_usb_can(usb_port)
    return _activate_interface(iface, bitrate, "arm", bus_info)


def activate_hand_can(bitrate: int = 1_000_000) -> str:
    """Detect and activate the LinkerHand CAN interface.

    Returns:
        Final interface name.
    """
    iface = _find_linkerhand()
    return _activate_interface(iface, bitrate, "LinkerHand")


def setup_all_can(
    bitrate: int = 1_000_000,
    usb_port: str | None = None,
    skip_arm: bool = False,
    skip_hand: bool = False,
) -> CanSetupResult:
    """Detect and activate both arm and LinkerHand CAN interfaces.

    Each device is handled independently — one failure does not block the other.

    Returns:
        :class:`CanSetupResult` with detected names (or error messages).
    """
    result = CanSetupResult()

    # --- Arm USB-CAN ---
    if skip_arm:
        result.arm_error = "skipped (--no-arm)"
    else:
        try:
            result.arm_can = activate_can(bitrate, usb_port)
        except Exception as e:
            result.arm_error = str(e)

    # --- LinkerHand ---
    if skip_hand:
        result.hand_error = "skipped (--no-hand)"
    else:
        try:
            result.hand_can = activate_hand_can(bitrate)
        except Exception as e:
            result.hand_error = str(e)

    return result
