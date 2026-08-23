# scripts/control_motor.py

"""
Simple script to remotely control a single motor.
Uses the exact same communication method as the RL policy.
Usage: python control_motor.py --motor 16 --steps 100 --direction pull
"""

import sys
import time
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dynamixel_sdk import *


# ============================================================
# Constants (match your dynamixel_executor.py)
# ============================================================

PORT_NAME = (
    "/dev/serial/by-id/"
    "usb-FTDI_USB__-__Serial_Converter_FT89FK0C"
    "-if00-port0"
)

BAUDRATE = 1000000
PROTOCOL_VERSION = 2.0

# Register addresses (same as your executor)
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
ADDR_TORQUE_ENABLE = 64
ADDR_OPERATING_MODE = 11

# Constants
EXTENDED_POSITION_MODE = 4
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0
MAX_ENCODER_DELTA = 200  # Same as your RL action


def control_motor(motor_id: int, steps: int, direction: str = "pull"):
    """
    Control a single motor for a number of steps.
    Uses the exact same communication as the RL executor.
    
    Args:
        motor_id: Motor ID (16, 17, 18, 19)
        steps: Number of steps to move (each step = MAX_ENCODER_DELTA ticks)
        direction: "pull" or "release"
    """
    
    # Calculate direction
    if direction.lower() == "pull":
        delta_per_step = MAX_ENCODER_DELTA  # Positive = pull
        direction_name = "PULLING"
    else:
        delta_per_step = -MAX_ENCODER_DELTA  # Negative = release
        direction_name = "RELEASING"
    
    total_delta = steps * delta_per_step
    
    print(f"\n{'='*60}")
    print(f"CONTROLLING MOTOR {motor_id}")
    print(f"Direction: {direction_name}")
    print(f"Steps: {steps}")
    print(f"Total movement: {total_delta:+d} encoder ticks")
    print(f"{'='*60}\n")
    
    # Setup Dynamixel (same as your create_dynamixel_bus)
    port = PortHandler(PORT_NAME)
    packet = PacketHandler(PROTOCOL_VERSION)
    
    # Open port
    if not port.openPort():
        raise RuntimeError("Cannot open Dynamixel port")
    
    if not port.setBaudRate(BAUDRATE):
        raise RuntimeError("Cannot set baudrate")
    
    print(f"✓ Port opened at {BAUDRATE} baud")
    
    # Ping motor to verify connection
    model, comm, error = packet.ping(port, motor_id)
    if comm != COMM_SUCCESS:
        print(f"✗ Motor {motor_id} not found!")
        port.closePort()
        return
    
    print(f"✓ Motor {motor_id} found (model: {model})")
    
    # ============================================================
    # 1. Initialize motor (same as executor.initialize())
    # ============================================================
    
    # Disable torque
    comm, error = packet.write1ByteTxRx(port, motor_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
    if comm != COMM_SUCCESS or error != 0:
        print("✗ Could not disable torque")
        port.closePort()
        return
    
    # Set extended position mode
    comm, error = packet.write1ByteTxRx(port, motor_id, ADDR_OPERATING_MODE, EXTENDED_POSITION_MODE)
    if comm != COMM_SUCCESS or error != 0:
        print("✗ Could not set extended position mode")
        port.closePort()
        return
    
    # Enable torque
    comm, error = packet.write1ByteTxRx(port, motor_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
    if comm != COMM_SUCCESS or error != 0:
        print("✗ Could not enable torque")
        port.closePort()
        return
    
    print("✓ Motor initialized (extended position mode)")
    
    # ============================================================
    # 2. Read current position (same as read_position)
    # ============================================================
    
    pos, comm, error = packet.read4ByteTxRx(port, motor_id, ADDR_PRESENT_POSITION)
    if comm != COMM_SUCCESS or error != 0:
        print("✗ Could not read position")
        port.closePort()
        return
    
    # Convert unsigned to signed (same as your executor)
    if pos > 0x7FFFFFFF:
        pos -= 0x100000000
    
    print(f"✓ Current position: {pos}")
    
    # ============================================================
    # 3. Move step by step (using the same write method)
    # ============================================================
    
    print(f"\nMoving {direction_name} for {steps} steps...")
    print("-" * 40)
    
    # Track target position (same as self.targets in executor)
    target = pos
    
    for step in range(1, steps + 1):
        # Update target (same as executor: self.targets[motor] += encoder_delta)
        target += delta_per_step
        
        # ============================================================
        # Write goal position (same as _write_goal_position_direct)
        # ============================================================
        
        # Clamp to 32-bit signed range (same as executor)
        target_clamped = max(-2147483648, min(2147483647, target))
        
        # Split into 4 bytes (same as executor's param building)
        param = [
            DXL_LOBYTE(DXL_LOWORD(target_clamped)),
            DXL_HIBYTE(DXL_LOWORD(target_clamped)),
            DXL_LOBYTE(DXL_HIWORD(target_clamped)),
            DXL_HIBYTE(DXL_HIWORD(target_clamped)),
        ]
        
        # Write to goal position register (same as executor's write4ByteTxRx)
        comm, error = packet.write4ByteTxRx(port, motor_id, ADDR_GOAL_POSITION, target_clamped)
        
        if comm != COMM_SUCCESS or error != 0:
            print(f"✗ Step {step} failed: comm={comm}, error={error}")
            break
        
        # Small delay (same as executor's time.sleep(0.02))
        time.sleep(0.02)
        
        # Read actual position every 10 steps or at the end
        if step % 10 == 0 or step == steps:
            pos_read, comm, error = packet.read4ByteTxRx(port, motor_id, ADDR_PRESENT_POSITION)
            if comm == COMM_SUCCESS and error == 0:
                if pos_read > 0x7FFFFFFF:
                    pos_read -= 0x100000000
                print(f"  Step {step:3d}/{steps}: target={target:8d}, actual={pos_read:8d}")
            else:
                print(f"  Step {step:3d}/{steps}: target={target:8d}, actual=unknown")
    
    # ============================================================
    # 4. Final position read
    # ============================================================
    
    final_pos, comm, error = packet.read4ByteTxRx(port, motor_id, ADDR_PRESENT_POSITION)
    if comm == COMM_SUCCESS and error == 0:
        if final_pos > 0x7FFFFFFF:
            final_pos -= 0x100000000
        print(f"\n✓ Final position: {final_pos}")
        print(f"✓ Total movement: {final_pos - pos:+d} ticks")
    
    # ============================================================
    # 5. Disable torque (same as executor.shutdown)
    # ============================================================
    
    comm, error = packet.write1ByteTxRx(port, motor_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
    if comm == COMM_SUCCESS:
        print("✓ Torque disabled")
    else:
        print("⚠ Could not disable torque")
    
    # Close port
    port.closePort()
    print(f"\n{'='*60}")
    print("✓ Done!")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Control a single motor remotely")
    parser.add_argument(
        "--motor", "-m",
        type=int,
        required=True,
        choices=[16, 17, 18, 19],
        help="Motor ID to control (16, 17, 18, or 19)"
    )
    parser.add_argument(
        "--steps", "-s",
        type=int,
        required=True,
        help="Number of steps to move (each step = 200 encoder ticks)"
    )
    parser.add_argument(
        "--direction", "-d",
        type=str,
        default="pull",
        choices=["pull", "release"],
        help="Direction to move: 'pull' or 'release' (default: pull)"
    )
    
    args = parser.parse_args()
    
    control_motor(args.motor, args.steps, args.direction)


if __name__ == "__main__":
    main()