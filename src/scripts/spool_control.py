#!/usr/bin/env python3
"""
Pull motor 10 steps - minimal version.
"""

from dynamixel_sdk import *
import time

# Settings
PORT_NAME = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT89FK0C-if00-port0"
BAUDRATE = 1000000
DXL_ID = 19
PROTOCOL_VERSION = 2.0

# Addresses
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132

# Constants
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0
STEP = 512
NUM_STEPS = 10

# Connect
port = PortHandler(PORT_NAME)
packet = PacketHandler(PROTOCOL_VERSION)

port.openPort()
port.setBaudRate(BAUDRATE)

# Enable torque
packet.write1ByteTxRx(port, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)

# Read current position
pos, _, _ = packet.read4ByteTxRx(port, DXL_ID, ADDR_PRESENT_POSITION)
if pos > 0x7FFFFFFF:
    pos -= 0x100000000

print(f"Start: {pos}")

# Pull 10 steps
for i in range(1, NUM_STEPS + 1):
    new_pos = pos + (STEP * i)
    packet.write4ByteTxRx(port, DXL_ID, ADDR_GOAL_POSITION, int(new_pos))
    print(f"Step {i}: {new_pos}")
    time.sleep(0.5)

# Disable torque
packet.write1ByteTxRx(port, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)

port.closePort()
print("Done!")