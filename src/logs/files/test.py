from dynamixel_sdk import *
import time

# =====================================================
# USER SETTINGS
# =====================================================

PORT_NAME = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT8ISG7I-if00-port0"
BAUDRATE = 1000000
PROTOCOL_VERSION = 2.0

ALL_IDS = [16, 17, 18, 19]

# =====================================================
# Addresses
# =====================================================

ADDR_PRESENT_POSITION = 132

# =====================================================
# SDK Setup
# =====================================================

portHandler = PortHandler(PORT_NAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)

if not portHandler.openPort():
    raise Exception("Failed to open port")

if not portHandler.setBaudRate(BAUDRATE):
    raise Exception("Failed to set baudrate")

print("Connected to motors.")

# =====================================================
# Read Positions
# =====================================================

def read_pos(dxl_id):
    pos, _, _ = packetHandler.read4ByteTxRx(
        portHandler,
        dxl_id,
        ADDR_PRESENT_POSITION
    )
    
    if pos > 0x7FFFFFFF:
        pos -= 0x100000000
    
    return pos

print("\n" + "="*50)
print("READING MOTOR POSITIONS")
print("="*50)
print("\nMake sure the strings are at the desired tension/position")
print("and press Enter to read the positions...")
input()

print("\nReading positions...\n")

positions = {}
for dxl_id in ALL_IDS:
    pos = read_pos(dxl_id)
    positions[dxl_id] = pos
    print(f"Motor {dxl_id}: {pos}")

print("\n" + "="*50)
print("HARDCODE THESE VALUES IN YOUR MAIN CODE")
print("="*50)

print("\nAdd these constants to your code:")
print(f'H_INITIAL_POS_16 = {positions[16]}')
print(f'H_INITIAL_POS_17 = {positions[17]}')
print(f'V_INITIAL_POS_18 = {positions[18]}')
print(f'V_INITIAL_POS_19 = {positions[19]}')

print("\nOr as a dictionary:")
print(f'INITIAL_POSITIONS = {{')
print(f'    16: {positions[16]},  # Horizontal Pull')
print(f'    17: {positions[17]},  # Horizontal Release')
print(f'    18: {positions[18]},  # Vertical Release')
print(f'    19: {positions[19]}   # Vertical Pull')
print(f'}}')

print("\n" + "="*50)

# =====================================================
# Cleanup
# =====================================================

portHandler.closePort()
print("\nPort closed.")
print("Finished.")


