# scripts/control_motor.py

"""
Simple script to remotely control a single motor.
Usage: python control_motor.py --motor 16 --steps 100 --direction pull
"""

import sys
import time
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dynamixel_sdk import *
from reset_policy.control.dynamixel_executor import DynamixelExecutor


# ============================================================
# Constants (match your main.py)
# ============================================================

PORT_NAME = (
    "/dev/serial/by-id/"
    "usb-FTDI_USB__-__Serial_Converter_FT89FK0C"
    "-if00-port0"
)

BAUDRATE = 1000000
PROTOCOL_VERSION = 2.0
MOTOR_IDS = [16, 17, 18, 19]

MAX_ENCODER_DELTA = 200  # Same as your RL action scaling


# ============================================================
# Simple Motor Control
# ============================================================

def control_motor(motor_id: int, steps: int, direction: str = "pull"):
    """
    Control a single motor for a number of steps.
    
    Args:
        motor_id: Motor ID (16, 17, 18, 19)
        steps: Number of steps to move (each step = MAX_ENCODER_DELTA ticks)
        direction: "pull" or "release"
    """
    
    # Calculate direction
    if direction.lower() == "pull":
        delta = MAX_ENCODER_DELTA  # Positive = pull
        direction_name = "PULLING"
    else:
        delta = -MAX_ENCODER_DELTA  # Negative = release
        direction_name = "RELEASING"
    
    print(f"\n{'='*60}")
    print(f"CONTROLLING MOTOR {motor_id}")
    print(f"Direction: {direction_name}")
    print(f"Steps: {steps}")
    print(f"Total movement: {steps * delta} encoder ticks")
    print(f"{'='*60}\n")
    
    # Setup Dynamixel
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
    
    # Read current position
    pos, comm, error = packet.read4ByteTxRx(port, motor_id, 132)  # ADDR_PRESENT_POSITION
    if comm == COMM_SUCCESS and error == 0:
        if pos > 0x7FFFFFFF:
            pos -= 0x100000000
        print(f"✓ Current position: {pos}")
    else:
        print("✗ Could not read position")
        port.closePort()
        return
    
    # Enable torque
    comm, error = packet.write1ByteTxRx(port, motor_id, 64, 1)  # ADDR_TORQUE_ENABLE = 1
    if comm != COMM_SUCCESS or error != 0:
        print("✗ Could not enable torque")
        port.closePort()
        return
    print("✓ Torque enabled")
    
    # Move step by step
    print(f"\nMoving {direction_name} for {steps} steps...")
    print("-" * 40)
    
    for step in range(1, steps + 1):
        # Calculate target
        target = pos + (step * delta)
        
        # Write goal position
        comm, error = packet.write4ByteTxRx(port, motor_id, 116, target)  # ADDR_GOAL_POSITION
        if comm != COMM_SUCCESS or error != 0:
            print(f"✗ Step {step} failed!")
            break
        
        # Wait a bit for movement
        time.sleep(0.02)
        
        # Read actual position every 10 steps
        if step % 10 == 0 or step == steps:
            pos_read, comm, error = packet.read4ByteTxRx(port, motor_id, 132)
            if comm == COMM_SUCCESS and error == 0:
                if pos_read > 0x7FFFFFFF:
                    pos_read -= 0x100000000
                print(f"  Step {step}/{steps}: target={target}, actual={pos_read}")
            else:
                print(f"  Step {step}/{steps}: target={target}, actual=unknown")
    
    # Final position read
    final_pos, comm, error = packet.read4ByteTxRx(port, motor_id, 132)
    if comm == COMM_SUCCESS and error == 0:
        if final_pos > 0x7FFFFFFF:
            final_pos -= 0x100000000
        print(f"\n✓ Final position: {final_pos}")
        print(f"✓ Total movement: {final_pos - pos}")
    
    # Disable torque
    comm, error = packet.write1ByteTxRx(port, motor_id, 64, 0)  # ADDR_TORQUE_ENABLE = 0
    if comm == COMM_SUCCESS:
        print("✓ Torque disabled")
    
    # Close port
    port.closePort()
    print(f"\n{'='*60}")
    print("✓ Done!")
    print(f"{'='*60}")


# ============================================================
# Main
# ============================================================

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