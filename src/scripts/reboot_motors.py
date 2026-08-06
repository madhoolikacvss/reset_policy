# reboot_motors.py
"""
Quick script to reboot/reset Dynamixel motors when they flash red.
Run this when motors are in error state (flashing red LED).
"""

from dynamixel_sdk import *
import time

# Configuration - MATCH YOUR MAIN.PY SETTINGS
PORT_NAME = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT89FK0C-if00-port0"
BAUDRATE = 1000000
PROTOCOL_VERSION = 2.0
MOTOR_IDS = [16, 17, 18, 19]

# Dynamixel addresses
ADDR_TORQUE_ENABLE = 64
ADDR_OPERATING_MODE = 11
ADDR_LED = 65  # LED on/off
ADDR_STATUS_RETURN_LEVEL = 68  # Status return level
ADDR_GOAL_POSITION = 116

# Constants
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0
EXTENDED_POSITION_MODE = 4

def reboot_motors():
    """Reboot all motors and clear error states."""
    
    print("=" * 60)
    print("DYNAMIXEL REBOOT SCRIPT")
    print("=" * 60)
    
    # Initialize port
    port = PortHandler(PORT_NAME)
    packet = PacketHandler(PROTOCOL_VERSION)
    
    # Open port
    if not port.openPort():
        print("[ERROR] Failed to open port")
        return False
    
    print(f"[OK] Port {PORT_NAME} opened")
    
    # Set baudrate
    if not port.setBaudRate(BAUDRATE):
        print("[ERROR] Failed to set baudrate")
        return False
    
    print(f"[OK] Baudrate {BAUDRATE} set")
    
    # Reboot each motor
    for motor_id in MOTOR_IDS:
        print(f"\n--- Motor {motor_id} ---")
        
        # 1. Check if motor is alive
        model_num, comm, error = packet.ping(port, motor_id)
        if comm != COMM_SUCCESS:
            print(f"  [ERROR] Cannot ping motor {motor_id}: {packet.getTxRxResult(comm)}")
            continue
        print(f"  [OK] Motor {motor_id} found (Model: {model_num})")
        
        # 2. Read current status
        led, comm, error = packet.read1ByteTxRx(port, motor_id, ADDR_LED)
        if comm == COMM_SUCCESS:
            print(f"  LED status: {led}")
        
        # 3. Disable torque first (safe reset)
        print("  Disabling torque...")
        comm, error = packet.write1ByteTxRx(port, motor_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
        if comm != COMM_SUCCESS:
            print(f"  [WARNING] Failed to disable torque: {packet.getTxRxResult(comm)}")
        
        # 4. Set to extended position mode (re-apply mode)
        print("  Setting extended position mode...")
        comm, error = packet.write1ByteTxRx(port, motor_id, ADDR_OPERATING_MODE, EXTENDED_POSITION_MODE)
        if comm != COMM_SUCCESS:
            print(f"  [WARNING] Failed to set mode: {packet.getTxRxResult(comm)}")
        
        # 5. Enable torque (this clears some error states)
        print("  Enabling torque...")
        comm, error = packet.write1ByteTxRx(port, motor_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
        if comm != COMM_SUCCESS:
            print(f"  [ERROR] Failed to enable torque: {packet.getTxRxResult(comm)}")
            continue
        print("  [OK] Torque enabled")
        
        # 6. Turn LED green to indicate success (optional)
        # Some Dynamixel models: 0=off, 1=green, 2=red, 3=orange
        print("  Setting LED to green...")
        comm, error = packet.write1ByteTxRx(port, motor_id, ADDR_LED, 1)  # Green
        if comm != COMM_SUCCESS:
            print(f"  [WARNING] Failed to set LED: {packet.getTxRxResult(comm)}")
        
        # 7. Read current position to verify
        pos, comm, error = packet.read4ByteTxRx(port, motor_id, ADDR_GOAL_POSITION)
        if comm == COMM_SUCCESS:
            if pos > 0x7FFFFFFF:
                pos -= 0x100000000
            print(f"  Current position: {pos}")
        
        time.sleep(0.1)
    
    print("\n" + "=" * 60)
    print("REBOOT COMPLETE")
    print("=" * 60)
    
    # Close port
    port.closePort()
    return True

def hard_reset_motors():
    """
    Hard reset - use this if motors are completely unresponsive.
    This sends a factory reset command (use with caution!).
    """
    print("\n[WARNING] Performing HARD RESET (factory reset)")
    confirm = input("Are you sure? This will reset ALL motor settings! (yes/no): ")
    
    if confirm.lower() != 'yes':
        print("Hard reset cancelled")
        return False
    
    port = PortHandler(PORT_NAME)
    packet = PacketHandler(PROTOCOL_VERSION)
    
    if not port.openPort():
        print("[ERROR] Failed to open port")
        return False
    
    port.setBaudRate(BAUDRATE)
    
    for motor_id in MOTOR_IDS:
        print(f"Factory resetting motor {motor_id}...")
        # Factory reset (address 0x08, value 0x01 for reset all)
        comm, error = packet.write1ByteTxRx(port, motor_id, 0x08, 0x01)
        if comm != COMM_SUCCESS:
            print(f"  [ERROR] Reset failed: {packet.getTxRxResult(comm)}")
        else:
            print(f"  [OK] Reset command sent to motor {motor_id}")
        time.sleep(0.5)
    
    port.closePort()
    return True

def check_motor_status():
    """Quickly check motor status without modifying anything."""
    
    port = PortHandler(PORT_NAME)
    packet = PacketHandler(PROTOCOL_VERSION)
    
    if not port.openPort():
        print("[ERROR] Failed to open port")
        return
    
    port.setBaudRate(BAUDRATE)
    
    print("\n--- Motor Status ---")
    for motor_id in MOTOR_IDS:
        # Ping to check connection
        model_num, comm, error = packet.ping(port, motor_id)
        if comm != COMM_SUCCESS:
            print(f"Motor {motor_id}: [NOT RESPONDING]")
            continue
        
        # Read torque enable status
        torque, comm, error = packet.read1ByteTxRx(port, motor_id, ADDR_TORQUE_ENABLE)
        
        # Read LED status
        led, comm, error = packet.read1ByteTxRx(port, motor_id, ADDR_LED)
        
        # Read current position
        pos, comm, error = packet.read4ByteTxRx(port, motor_id, ADDR_GOAL_POSITION)
        if comm == COMM_SUCCESS:
            if pos > 0x7FFFFFFF:
                pos -= 0x100000000
        
        led_colors = {0: "OFF", 1: "GREEN", 2: "RED (ERROR)", 3: "ORANGE"}
        led_status = led_colors.get(led, f"UNKNOWN ({led})")
        
        print(f"Motor {motor_id}:")
        print(f"  Model: {model_num}")
        print(f"  Torque: {'ON' if torque else 'OFF'}")
        print(f"  LED: {led_status}")
        print(f"  Position: {pos}")
        print()
    
    port.closePort()

if __name__ == "__main__":
    print("\nChoose action:")
    print("1. Soft reboot (recommended for error recovery)")
    print("2. Hard reset (factory reset - use only if soft reboot fails)")
    print("3. Quick status check only")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == '1':
        reboot_motors()
    elif choice == '2':
        hard_reset_motors()
    elif choice == '3':
        # Just check status
        check_motor_status()
    else:
        print("Invalid choice")