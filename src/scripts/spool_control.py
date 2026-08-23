from dynamixel_sdk import *
import keyboard
import time
import csv
from datetime import datetime

# ==========================
# USER SETTINGS
# ==========================
PORT_NAME = (
    "/dev/serial/by-id/"
    "usb-FTDI_USB__-__Serial_Converter_FT89FK0C"
    "-if00-port0"
)

BAUDRATE = 1000000
DXL_ID = 19 # Motor you will control
PROTOCOL_VERSION = 2.0

# All motor IDs
ALL_MOTOR_IDS = [16, 17, 18, 19]

# ==========================
# Control Table Addresses
# ==========================
ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
ADDR_PRESENT_CURRENT = 126
ADDR_CURRENT_LIMIT = 38
ADDR_HARDWARE_ERROR_STATUS = 70  # Added hardware error status address

TORQUE_ENABLE = 1
TORQUE_DISABLE = 0

EXTENDED_POSITION_MODE = 4

# Encoder counts
COUNTS_PER_REV = 4096

# Amount moved per key press
STEP = 512          # 1/8 revolution
# STEP = 1024       # quarter revolution
# STEP = 2048       # half revolution

# ==========================
# Open Port
# ==========================
portHandler = PortHandler(PORT_NAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)

if not portHandler.openPort():
    raise Exception("Failed to open port")

if not portHandler.setBaudRate(BAUDRATE):
    raise Exception("Failed to set baudrate")

print("Connected.")

# ==========================
# Configure All Motors
# ==========================

for motor_id in ALL_MOTOR_IDS:
    # Disable torque before changing mode
    packetHandler.write1ByteTxRx(
        portHandler,
        motor_id,
        ADDR_TORQUE_ENABLE,
        TORQUE_DISABLE
    )

    # Set Extended Position Mode
    packetHandler.write1ByteTxRx(
        portHandler,
        motor_id,
        ADDR_OPERATING_MODE,
        EXTENDED_POSITION_MODE
    )

# Enable torque ONLY on the controlled motor (DXL_ID)
packetHandler.write1ByteTxRx(
    portHandler,
    DXL_ID,
    ADDR_TORQUE_ENABLE,
    TORQUE_ENABLE
)

# Disable torque on all other motors
for motor_id in ALL_MOTOR_IDS:
    if motor_id != DXL_ID:
        packetHandler.write1ByteTxRx(
            portHandler,
            motor_id,
            ADDR_TORQUE_ENABLE,
            TORQUE_DISABLE
        )

print(f"Motor {DXL_ID} enabled. All other motors disabled.")

# Read current position for the controlled motor
current_target, _, _ = packetHandler.read4ByteTxRx(
    portHandler,
    DXL_ID,
    ADDR_PRESENT_POSITION
)

# Convert unsigned -> signed
if current_target > 0x7FFFFFFF:
    current_target -= 0x100000000

print(f"Current position = {current_target}")

print()
print("Controls:")
print("  P : Pull (wind spool)")
print("  R : Release (unwind spool)")
print("  E : Check hardware error status")
print("  Q : Quit")
print()

# ==========================
# Current Reading Function
# ==========================

def read_current(motor_id):
    """Read current from motor in mA (for XL330-M288)."""
    current, dxl_comm_result, dxl_error = packetHandler.read2ByteTxRx(
        portHandler,
        motor_id,
        ADDR_PRESENT_CURRENT
    )
    
    if dxl_comm_result != COMM_SUCCESS:
        print(f"Motor {motor_id} - {packetHandler.getTxRxResult(dxl_comm_result)}")
        return None
    
    if dxl_error != 0:
        print(f"Motor {motor_id} - {packetHandler.getRxPacketError(dxl_error)}")
        return None
    
    # Convert unsigned 16-bit -> signed 16-bit
    if current >= 0x8000:
        current -= 0x10000
    
    # Convert to mA (XL330-M288: 1 unit = 2.69 mA)
    current_ma = current * 2.69
    
    return current_ma

def read_current_limit(motor_id):
    """Read current limit from motor in mA (for XL330-M288)."""
    current_limit, dxl_comm_result, dxl_error = packetHandler.read2ByteTxRx(
        portHandler,
        motor_id,
        ADDR_CURRENT_LIMIT
    )
    
    if dxl_comm_result != COMM_SUCCESS:
        print(f"Motor {motor_id} - {packetHandler.getTxRxResult(dxl_comm_result)}")
        return None
    
    if dxl_error != 0:
        print(f"Motor {motor_id} - {packetHandler.getRxPacketError(dxl_error)}")
        return None
    
    # Convert to mA (XL330-M288: 1 unit = 2.69 mA)
    current_limit_ma = current_limit * 2.69
    
    return current_limit_ma

def read_hardware_error_status(motor_id):
    """Read hardware error status to see what error occurred."""
    error_status, dxl_comm_result, dxl_error = packetHandler.read1ByteTxRx(
        portHandler,
        motor_id,
        ADDR_HARDWARE_ERROR_STATUS
    )
    
    if dxl_comm_result != COMM_SUCCESS:
        print(f"Motor {motor_id} - {packetHandler.getTxRxResult(dxl_comm_result)}")
        return None
    
    if dxl_error != 0:
        print(f"Motor {motor_id} - {packetHandler.getRxPacketError(dxl_error)}")
        return None
    
    return error_status

def decode_hardware_error(error_status):
    """Decode the hardware error status bits."""
    errors = []
    if error_status & 0x01:
        errors.append("Input Voltage Error")
    if error_status & 0x02:
        errors.append("Motor Encoder Error")
    if error_status & 0x04:
        errors.append("Motor Overheating")
    if error_status & 0x08:
        errors.append("Motor Overload")
    if error_status & 0x10:
        errors.append("Motor Driver Fault")
    if error_status & 0x20:
        errors.append("Motor Hall Sensor Error")
    return errors

def read_all_currents():
    """Read currents from all motors."""
    currents = {}
    for motor_id in ALL_MOTOR_IDS:
        currents[motor_id] = read_current(motor_id)
    return currents

def read_all_current_limits():
    """Read current limits from all motors."""
    limits = {}
    for motor_id in ALL_MOTOR_IDS:
        limits[motor_id] = read_current_limit(motor_id)
    return limits

def read_all_hardware_errors():
    """Read hardware error status from all motors."""
    errors = {}
    for motor_id in ALL_MOTOR_IDS:
        errors[motor_id] = read_hardware_error_status(motor_id)
    return errors

def move(position):
    packetHandler.write4ByteTxRx(
        portHandler,
        DXL_ID,
        ADDR_GOAL_POSITION,
        int(position)
    )

# ==========================
# Initialize CSV Logging
# ==========================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f"current_log_{timestamp}.csv"
csv_file = open(csv_filename, 'w', newline='')
csv_writer = csv.writer(csv_file)

# Write header with all motors and their limits
header = [
    "Timestamp",
    "Action",
    "Target_Position",
]
for motor_id in ALL_MOTOR_IDS:
    header.append(f"Motor_{motor_id}_Current_mA")
for motor_id in ALL_MOTOR_IDS:
    header.append(f"Motor_{motor_id}_CurrentLimit_mA")
for motor_id in ALL_MOTOR_IDS:
    header.append(f"Motor_{motor_id}_HardwareError")
csv_writer.writerow(header)
csv_file.flush()

print(f"Logging current to: {csv_filename}")
print()

# ==========================
# Main Loop
# ==========================

while True:

    event = keyboard.read_event()

    if event.event_type != keyboard.KEY_DOWN:
        continue

    key = event.name.lower()

    if key == 'p':
        current_target += STEP
        move(current_target)
        
        # Add a small delay to let the motor move
        time.sleep(0.1)
        
        # Read currents from all motors
        all_currents = read_all_currents()
        all_limits = read_all_current_limits()
        all_errors = read_all_hardware_errors()
        
        # Log to CSV
        row = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "PULL",
            current_target,
        ]
        for motor_id in ALL_MOTOR_IDS:
            current_ma = all_currents.get(motor_id)
            row.append(f"{current_ma:.1f}" if current_ma is not None else "ERROR")
        for motor_id in ALL_MOTOR_IDS:
            limit_ma = all_limits.get(motor_id)
            row.append(f"{limit_ma:.1f}" if limit_ma is not None else "ERROR")
        for motor_id in ALL_MOTOR_IDS:
            error = all_errors.get(motor_id)
            if error is not None:
                errors = decode_hardware_error(error)
                row.append("; ".join(errors) if errors else "NONE")
            else:
                row.append("ERROR")
        csv_writer.writerow(row)
        csv_file.flush()
        
        # Print to console
        print(f"\nPull -> Target = {current_target}")
        for motor_id in ALL_MOTOR_IDS:
            current_ma = all_currents.get(motor_id)
            limit_ma = all_limits.get(motor_id)
            error = all_errors.get(motor_id)
            
            if current_ma is not None and limit_ma is not None:
                print(f"  Motor {motor_id}: {current_ma:.1f} mA / Limit: {limit_ma:.1f} mA", end="")
                if error is not None:
                    errors = decode_hardware_error(error)
                    if errors:
                        print(f" ⚠️ ERROR: {', '.join(errors)}")
                    else:
                        print(" ✅ OK")
                else:
                    print(" ❌ READ ERROR")
            else:
                print(f"  Motor {motor_id}: ERROR")

    elif key == 'r':
        current_target -= STEP
        move(current_target)
        
        # Add a small delay to let the motor move
        time.sleep(0.1)
        
        # Read currents from all motors
        all_currents = read_all_currents()
        all_limits = read_all_current_limits()
        all_errors = read_all_hardware_errors()
        
        # Log to CSV
        row = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "RELEASE",
            current_target,
        ]
        for motor_id in ALL_MOTOR_IDS:
            current_ma = all_currents.get(motor_id)
            row.append(f"{current_ma:.1f}" if current_ma is not None else "ERROR")
        for motor_id in ALL_MOTOR_IDS:
            limit_ma = all_limits.get(motor_id)
            row.append(f"{limit_ma:.1f}" if limit_ma is not None else "ERROR")
        for motor_id in ALL_MOTOR_IDS:
            error = all_errors.get(motor_id)
            if error is not None:
                errors = decode_hardware_error(error)
                row.append("; ".join(errors) if errors else "NONE")
            else:
                row.append("ERROR")
        csv_writer.writerow(row)
        csv_file.flush()
        
        # Print to console
        print(f"\nRelease -> Target = {current_target}")
        for motor_id in ALL_MOTOR_IDS:
            current_ma = all_currents.get(motor_id)
            limit_ma = all_limits.get(motor_id)
            error = all_errors.get(motor_id)
            
            if current_ma is not None and limit_ma is not None:
                print(f"  Motor {motor_id}: {current_ma:.1f} mA / Limit: {limit_ma:.1f} mA", end="")
                if error is not None:
                    errors = decode_hardware_error(error)
                    if errors:
                        print(f" ⚠️ ERROR: {', '.join(errors)}")
                    else:
                        print(" ✅ OK")
                else:
                    print(" ❌ READ ERROR")
            else:
                print(f"  Motor {motor_id}: ERROR")

    elif key == 'e':
        # Manual hardware error check
        print("\n=== Hardware Error Status ===")
        for motor_id in ALL_MOTOR_IDS:
            error = read_hardware_error_status(motor_id)
            if error is not None:
                errors = decode_hardware_error(error)
                status = " ⚠️ " + ", ".join(errors) if errors else " ✅ OK"
                print(f"Motor {motor_id}: {status}")
            else:
                print(f"Motor {motor_id}: ❌ READ ERROR")

    elif key == 'q':
        break

# ==========================
# Cleanup
# ==========================

# Close CSV file
csv_file.close()
print(f"\nCurrent data saved to: {csv_filename}")

# Disable torque on all motors
for motor_id in ALL_MOTOR_IDS:
    packetHandler.write1ByteTxRx(
        portHandler,
        motor_id,
        ADDR_TORQUE_ENABLE,
        TORQUE_DISABLE
    )

portHandler.closePort()

print("Finished.")