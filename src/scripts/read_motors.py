"""
Read Hardware Error Status (address 70) from a Dynamixel motor.

Usage:
    python scripts/read_hw_status.py --motor 18
    python scripts/read_hw_status.py --motor 18 --reboot
    python scripts/read_hw_status.py --all
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dynamixel_sdk import *

from reset_policy.config import config


ADDR_HARDWARE_ERROR_STATUS = 70
ADDR_TORQUE_ENABLE = 64
ADDR_PRESENT_POSITION = 132
ADDR_PRESENT_CURRENT = 126
ADDR_PRESENT_VOLTAGE = 144
ADDR_PRESENT_TEMPERATURE = 146
ADDR_OPERATING_MODE = 11

EXTENDED_POSITION_MODE = 4
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0

VALID_HW_ERROR_BITS = 0x01 | 0x04 | 0x10 | 0x20 | 0x08


def decode_hw_status(status):
    """Decode the Hardware Error Status byte."""
    if status is None:
        return ["No status"]
    errors = []
    if status & 0x01:
        errors.append("Input voltage error (0x01)")
    if status & 0x02:
        errors.append("Motor hall sensor error (0x02)")
    if status & 0x04:
        errors.append("Overheating (0x04)")
    if status & 0x08:
        errors.append("Motor encoder error (0x08)")
    if status & 0x10:
        errors.append("Electrical shock (0x10)")
    if status & 0x20:
        errors.append("Overload (0x20)")
    unknown = status & ~VALID_HW_ERROR_BITS
    if unknown:
        errors.append(f"Unknown bits 0x{unknown:02X}")
    return errors if errors else ["No error"]


def read_one(packet, port, motor_id, verbose=True):
    """Read hardware error status + diagnostic registers for one motor."""

    # Hardware Error Status (the key register)
    status, comm, error = packet.read1ByteTxRx(port, motor_id, ADDR_HARDWARE_ERROR_STATUS)

    print(f"\n{'='*60}")
    print(f"Motor {motor_id}")
    print(f"{'='*60}")

    if comm != COMM_SUCCESS:
        print(f"  [HW ERROR STATUS] COMM FAIL: {packet.getTxRxResult(comm)}")
        return None
    if error != 0:
        print(f"  [HW ERROR STATUS] PACKET ERROR: 0x{error:02X} ({packet.getRxPacketError(error)})")
        # Still print the status if we got it
        if status is not None:
            print(f"  [HW ERROR STATUS] Raw: 0x{status:02X} ({status:08b}b)")

    if status is not None:
        print(f"  [HW ERROR STATUS] Raw: 0x{status:02X} ({status:08b}b)")
        decoded = decode_hw_status(status)
        for d in decoded:
            print(f"    - {d}")

    # Additional registers for context
    if verbose:
        # Voltage
        volt_raw, c, e = packet.read2ByteTxRx(port, motor_id, ADDR_PRESENT_VOLTAGE)
        if c == COMM_SUCCESS and e == 0:
            print(f"  [Voltage]         {volt_raw * 0.1:.1f} V")

        # Temperature
        temp, c, e = packet.read1ByteTxRx(port, motor_id, ADDR_PRESENT_TEMPERATURE)
        if c == COMM_SUCCESS and e == 0:
            print(f"  [Temperature]     {temp} °C")

        # Current
        cur, c, e = packet.read2ByteTxRx(port, motor_id, ADDR_PRESENT_CURRENT)
        if c == COMM_SUCCESS and e == 0:
            if cur >= 0x8000:
                cur -= 0x10000
            print(f"  [Current]         {cur} mA")

        # Position
        pos, c, e = packet.read4ByteTxRx(port, motor_id, ADDR_PRESENT_POSITION)
        if c == COMM_SUCCESS and e == 0:
            if pos >= 0x80000000:
                pos -= 0x100000000
            print(f"  [Position]        {pos}")

        # Operating mode
        mode, c, e = packet.read1ByteTxRx(port, motor_id, ADDR_OPERATING_MODE)
        if c == COMM_SUCCESS and e == 0:
            print(f"  [Operating Mode]  {mode}")

        # Torque enable
        torque, c, e = packet.read1ByteTxRx(port, motor_id, ADDR_TORQUE_ENABLE)
        if c == COMM_SUCCESS and e == 0:
            print(f"  [Torque Enable]   {torque}")

    return status


def try_reboot(packet, port, motor_id):
    """Attempt a reboot and re-initialize."""
    print(f"\n[REBOOT] Attempting reboot of Motor {motor_id}...")
    comm = packet.reboot(port, motor_id)

    if comm != COMM_SUCCESS:
        print(f"[REBOOT] Communication Error: {packet.getTxRxResult(comm)}")
        return False

    print(f"[REBOOT] Successfully rebooted Motor {motor_id}.")
    time.sleep(1.0)

    # Re-initialize
    try:
        packet.write1ByteTxRx(port, motor_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
        packet.write1ByteTxRx(port, motor_id, ADDR_OPERATING_MODE, EXTENDED_POSITION_MODE)
        packet.write1ByteTxRx(port, motor_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
        print(f"[REBOOT] Motor {motor_id} re-initialized (torque enabled).")
    except Exception as e:
        print(f"[REBOOT] Re-init failed: {e}")
        return False

    # Re-read status
    print(f"[REBOOT] Re-reading hardware error status...")
    time.sleep(0.5)
    read_one(packet, port, motor_id, verbose=True)

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motor", type=int, default=None,
                        help="Motor ID to read (16, 17, 18, 19)")
    parser.add_argument("--all", action="store_true",
                        help="Read all motors")
    parser.add_argument("--reboot", action="store_true",
                        help="Attempt reboot if a hardware error is detected")
    args = parser.parse_args()

    if args.motor is None and not args.all:
        parser.error("Provide --motor ID or --all")

    # Open port
    port = PortHandler(config.dynamixel.port_name)
    packet = PacketHandler(config.dynamixel.protocol_version)

    if not port.openPort():
        raise RuntimeError("Cannot open Dynamixel port")
    if not port.setBaudRate(config.dynamixel.baudrate):
        raise RuntimeError("Cannot set baudrate")

    print(f"Port opened at {config.dynamixel.baudrate} baud")

    motors = config.motor.motor_ids if args.all else [args.motor]

    for m in motors:
        status = read_one(packet, port, m, verbose=True)

        if status is not None and status != 0 and args.reboot:
            try_reboot(packet, port, m)
        elif status is not None and status != 0:
            print(f"\n  Motor {m} has a hardware error. "
                  f"Re-run with --reboot to attempt recovery.")

    port.closePort()
    print("\nPort closed.")


if __name__ == "__main__":
    main()
