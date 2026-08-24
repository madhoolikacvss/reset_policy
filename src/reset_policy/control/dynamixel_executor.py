"""
Low-level interface for executing RL actions on four Dynamixels.

Responsibilities:
- Initialize Dynamixels
- Convert RL actions into relative encoder movements
- Execute synchronized motor commands
- Read encoder positions and currents
- Read voltage, temperature, torque, PWM, velocity
- Decode hardware errors
- Log motor diagnostics
- Enforce relative position limits
- Recover motors after hardware shutdown
- Shutdown safely
"""

from __future__ import annotations

import csv
import time
import numpy as np

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

from dynamixel_sdk import *


# ============================================================
# Dynamixel addresses
# ============================================================

ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_HARDWARE_ERROR_STATUS = 70
ADDR_PRESENT_PWM = 124
ADDR_PRESENT_CURRENT = 126
ADDR_PRESENT_VELOCITY = 128
ADDR_PRESENT_POSITION = 132
ADDR_PRESENT_INPUT_VOLTAGE = 144
ADDR_PRESENT_TEMPERATURE = 146
ADDR_GOAL_POSITION = 116
ADDR_REBOOT = 0x08  # Protocol 2.0 reboot instruction


# ============================================================
# Constants
# ============================================================

EXTENDED_POSITION_MODE = 4
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0

# RL action [-1, 1] -> encoder delta
MAX_ENCODER_DELTA = 200
MAX_ENCODER_TRAVEL = 10000

# Diagnostic logging
DIAGNOSTIC_LOG_EVERY_N_ACTIONS = 1
DIAGNOSTIC_LOG_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "scripts"
    / "training_logs"
)
DIAGNOSTIC_LOG_FILE = DIAGNOSTIC_LOG_DIR / "motor_diagnostics.csv"
ACTION_LOG_FILE = DIAGNOSTIC_LOG_DIR / "action_log.csv"


# ============================================================
# Execution result
# ============================================================

@dataclass
class ExecutionResult:
    success: bool
    hardware_error: bool = False
    hardware_error_ids: list = None
    hardware_error_status: dict = None
    error_message: str = ""

    def __post_init__(self):
        if self.hardware_error_ids is None:
            self.hardware_error_ids = []
        if self.hardware_error_status is None:
            self.hardware_error_status = {}


# ============================================================
# Dynamixel Executor
# ============================================================

class DynamixelExecutor:

    def __init__(
        self,
        port_handler,
        packet_handler,
        motor_ids,
        group_sync_write,
    ):
        self.port = port_handler
        self.packet = packet_handler
        self.motor_ids = list(motor_ids)
        self.group_sync_write = group_sync_write

        # Position tracking
        self.targets = {}
        self.initial_positions = {}

        # Hardware error bookkeeping
        self.hardware_error = False
        self.hardware_error_ids = []
        self.hardware_error_status = {}
        self.hardware_error_message = None

        # Action counter
        self.action_count = 0

        # Diagnostic logging
        self.diagnostic_log_file = DIAGNOSTIC_LOG_FILE
        self.action_log_file = ACTION_LOG_FILE
        self._initialize_diagnostic_log()
        self._initialize_action_log()

        self.current_episode = 0
        self.current_step = 0

    # ========================================================
    # Diagnostic logging
    # ========================================================

    def _initialize_diagnostic_log(self):
        """Create the motor diagnostic CSV if it does not exist."""
        try:
            self.diagnostic_log_file.parent.mkdir(parents=True, exist_ok=True)
            if not self.diagnostic_log_file.exists():
                with open(self.diagnostic_log_file, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "timestamp", "action_count", "motor_id",
                        "target_position", "present_position",
                        "present_current_mA", "present_voltage_V", "temperature_C",
                        "torque_enabled", "present_pwm", "present_velocity",
                        "hardware_error_status", "hardware_error_hex",
                        "packet_error", "packet_error_hex",
                        "hardware_error_decoded",
                    ])
        except Exception as e:
            print(f"WARNING: Could not initialize diagnostic log: {e}")

    def _initialize_action_log(self):
        """Create the action-level CSV if it does not exist."""
        try:
            self.diagnostic_log_file.parent.mkdir(parents=True, exist_ok=True)
            if not self.action_log_file.exists():
                with open(self.action_log_file, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "timestamp", "action_count",
                        "action_m16", "action_m17", "action_m18", "action_m19",
                        "delta_m16", "delta_m17", "delta_m18", "delta_m19",
                        "target_before_m16", "target_before_m17", "target_before_m18", "target_before_m19",
                        "target_after_m16", "target_after_m17", "target_after_m18", "target_after_m19",
                        "position_m16", "position_m17", "position_m18", "position_m19",
                        "current_m16_mA", "current_m17_mA", "current_m18_mA", "current_m19_mA",
                        "voltage_m16_V", "voltage_m17_V", "voltage_m18_V", "voltage_m19_V",
                        "temperature_m16_C", "temperature_m17_C", "temperature_m18_C", "temperature_m19_C",
                        "pwm_m16", "pwm_m17", "pwm_m18", "pwm_m19",
                        "velocity_m16", "velocity_m17", "velocity_m18", "velocity_m19",
                        "hardware_status_m16", "hardware_status_m17", "hardware_status_m18", "hardware_status_m19",
                        "packet_error_m16", "packet_error_m17", "packet_error_m18", "packet_error_m19",
                        "success", "hardware_error", "hardware_error_ids", "error_message",
                    ])
        except Exception as e:
            print(f"WARNING: Could not initialize action log: {e}")

    def log_motor_diagnostics(self, motor_id, reason="periodic", packet_error=None, hardware_status=None):
        """Read and log important motor telemetry."""
        timestamp = datetime.now().isoformat()

        # Read all telemetry
        position = self._read4_raw(motor_id, ADDR_PRESENT_POSITION)
        
        current = self._read2_raw(motor_id, ADDR_PRESENT_CURRENT)
        if current is not None and current >= 0x8000:
            current -= 0x10000

        voltage_raw = self._read2_raw(motor_id, ADDR_PRESENT_INPUT_VOLTAGE)
        voltage = voltage_raw * 0.1 if voltage_raw is not None else None

        temperature = self._read1_raw(motor_id, ADDR_PRESENT_TEMPERATURE)
        torque = self._read1_raw(motor_id, ADDR_TORQUE_ENABLE)

        pwm = self._read2_raw(motor_id, ADDR_PRESENT_PWM)
        if pwm is not None and pwm >= 0x8000:
            pwm -= 0x10000

        velocity = self._read4_raw(motor_id, ADDR_PRESENT_VELOCITY)
        if velocity is not None and velocity >= 0x80000000:
            velocity -= 0x100000000

        if hardware_status is None:
            hardware_status = self._read1_raw(motor_id, ADDR_HARDWARE_ERROR_STATUS)

        decoded = self.decode_hardware_error(hardware_status)

        # Console output
        print(
            f"\n[MOTOR DIAGNOSTICS] reason={reason} | motor={motor_id}\n"
            f"  position : {position}\n"
            f"  target   : {self.targets.get(motor_id)}\n"
            f"  current  : {current} mA\n"
            f"  voltage  : {voltage} V\n"
            f"  temp     : {temperature} C\n"
            f"  torque   : {torque}\n"
            f"  PWM      : {pwm}\n"
            f"  velocity : {velocity}\n"
            f"  HW status: {hardware_status} ({self.format_hex(hardware_status)})\n"
            f"  decoded  : {', '.join(decoded)}\n"
            f"  packet error: {packet_error} ({self.format_hex(packet_error)})"
        )

        # CSV logging
        try:
            with open(self.diagnostic_log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp, self.action_count, motor_id,
                    self.targets.get(motor_id), position,
                    current, voltage, temperature,
                    torque, pwm, velocity,
                    hardware_status, self.format_hex(hardware_status),
                    packet_error, self.format_hex(packet_error),
                    ", ".join(decoded),
                ])
        except Exception as e:
            print(f"WARNING: failed to write diagnostic log: {e}")

    def log_all_motor_diagnostics(self, reason="periodic", packet_errors=None, hardware_statuses=None):
        """Capture diagnostics for every motor."""
        packet_errors = packet_errors or {}
        hardware_statuses = hardware_statuses or {}

        print("\n" + "=" * 48)
        print(f"MOTOR DIAGNOSTIC SNAPSHOT: {reason}")
        print("=" * 48)

        for motor in self.motor_ids:
            self.log_motor_diagnostics(
                motor,
                reason=reason,
                packet_error=packet_errors.get(motor),
                hardware_status=hardware_statuses.get(motor),
            )

    # ========================================================
    # Raw reads for diagnostics
    # ========================================================

    def _read1_raw(self, motor_id, address):
        value, comm, error = self.packet.read1ByteTxRx(self.port, motor_id, address)
        if comm != COMM_SUCCESS or error != 0:
            return None
        return value

    def _read2_raw(self, motor_id, address):
        value, comm, error = self.packet.read2ByteTxRx(self.port, motor_id, address)
        if comm != COMM_SUCCESS or error != 0:
            return None
        return value

    def _read4_raw(self, motor_id, address):
        value, comm, error = self.packet.read4ByteTxRx(self.port, motor_id, address)
        if comm != COMM_SUCCESS or error != 0:
            return None
        return value

    def _get_motor_telemetry(self, motor_id):
        """Read a complete low-level telemetry snapshot for one motor."""
        position = self._read4_raw(motor_id, ADDR_PRESENT_POSITION)
        if position is not None and position >= 0x80000000:
            position -= 0x100000000

        current = self._read2_raw(motor_id, ADDR_PRESENT_CURRENT)
        if current is not None and current >= 0x8000:
            current -= 0x10000

        voltage_raw = self._read2_raw(motor_id, ADDR_PRESENT_INPUT_VOLTAGE)
        voltage = voltage_raw * 0.1 if voltage_raw is not None else None

        temperature = self._read1_raw(motor_id, ADDR_PRESENT_TEMPERATURE)
        torque = self._read1_raw(motor_id, ADDR_TORQUE_ENABLE)

        pwm = self._read2_raw(motor_id, ADDR_PRESENT_PWM)
        if pwm is not None and pwm >= 0x8000:
            pwm -= 0x10000

        velocity = self._read4_raw(motor_id, ADDR_PRESENT_VELOCITY)
        if velocity is not None and velocity >= 0x80000000:
            velocity -= 0x100000000

        hardware_status = self._read1_raw(motor_id, ADDR_HARDWARE_ERROR_STATUS)

        return {
            "position": position,
            "current": current,
            "voltage": voltage,
            "temperature": temperature,
            "torque": torque,
            "pwm": pwm,
            "velocity": velocity,
            "hardware_status": hardware_status,
        }

    def _log_action(self, action, encoder_deltas, targets_before, targets_after, telemetry,
                    packet_errors=None, success=True, hardware_error=False,
                    hardware_error_ids=None, error_message=""):
        """Write ONE CSV row for ONE RL action."""
        packet_errors = packet_errors or {}
        hardware_error_ids = hardware_error_ids or []

        timestamp = datetime.now().isoformat()
        row = [timestamp, self.action_count]

        # RL action
        row.extend([float(action[0]), float(action[1]), float(action[2]), float(action[3])])

        # Encoder deltas
        for motor in self.motor_ids:
            row.append(encoder_deltas[motor])

        # Targets before and after
        for motor in self.motor_ids:
            row.append(targets_before[motor])
        for motor in self.motor_ids:
            row.append(targets_after[motor])

        # Telemetry
        for motor in self.motor_ids:
            row.append(telemetry[motor]["position"])
        for motor in self.motor_ids:
            row.append(telemetry[motor]["current"])
        for motor in self.motor_ids:
            row.append(telemetry[motor]["voltage"])
        for motor in self.motor_ids:
            row.append(telemetry[motor]["temperature"])
        for motor in self.motor_ids:
            row.append(telemetry[motor]["pwm"])
        for motor in self.motor_ids:
            row.append(telemetry[motor]["velocity"])
        for motor in self.motor_ids:
            row.append(telemetry[motor]["hardware_status"])
        for motor in self.motor_ids:
            row.append(packet_errors.get(motor))

        # Result
        row.extend([success, hardware_error, ",".join(str(x) for x in hardware_error_ids), error_message])

        try:
            with open(self.action_log_file, "a", newline="") as f:
                csv.writer(f).writerow(row)
        except Exception as e:
            print(f"WARNING: failed to write action log: {e}")

    # ========================================================
    # Initialization
    # ========================================================

    def initialize(self):
        print("Initializing Dynamixels...")

        for motor in self.motor_ids:
            self.write1(motor, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
            self.write1(motor, ADDR_OPERATING_MODE, EXTENDED_POSITION_MODE)
            self.write1(motor, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)

            pos = self.read_position(motor)
            if pos is None:
                raise RuntimeError(f"Could not read initial position of motor {motor}")

            self.initial_positions[motor] = pos
            self.targets[motor] = pos
            print(f"Motor {motor}: initial position = {pos}")

        self.log_all_motor_diagnostics(reason="initialization")
        print("Initialization complete.")

    # ========================================================
    # Write operations
    # ========================================================

    def write1(self, motor, address, value):
        comm, error = self.packet.write1ByteTxRx(self.port, motor, address, value)
        if comm != COMM_SUCCESS:
            raise RuntimeError(f"Motor {motor}: {self.packet.getTxRxResult(comm)}")
        if error != 0:
            raise RuntimeError(f"Motor {motor}: {self.packet.getRxPacketError(error)}")

    def _write_goal_position_direct(self, motor_id, target_position):
        """Write a goal position directly (with error handling)."""
        target_position = int(target_position)
        # Clamp to 32-bit signed range
        target_position = max(-2147483648, min(2147483647, target_position))

        comm, error = self.packet.write4ByteTxRx(self.port, motor_id, ADDR_GOAL_POSITION, target_position)
        if comm != COMM_SUCCESS:
            print(f"Motor {motor_id}: Write failed - {self.packet.getTxRxResult(comm)}")
            return False
        if error != 0:
            print(f"Motor {motor_id}: Write error - 0x{error:02X}")
            return False

        self.targets[motor_id] = target_position
        return True

    # ========================================================
    # Execute RL action
    # ========================================================

    def execute(self, action):
        if len(action) != len(self.motor_ids):
            raise ValueError("Action dimension does not match motors")

        self.action_count += 1
        action = np.asarray(action, dtype=np.float32)

        print(f"\n{'='*60}")
        print(f"RL ACTION #{self.action_count}")
        print(f"timestamp: {datetime.now().isoformat()}")
        print(f"raw action: {action}")
        print("=" * 60)

        # Save targets BEFORE action
        targets_before = {motor: int(self.targets[motor]) for motor in self.motor_ids}

        # Convert RL action -> encoder deltas
        encoder_deltas = {}
        targets_after = {}

        for motor, delta in zip(self.motor_ids, action):
            delta = np.clip(delta, -1.0, 1.0)
            encoder_delta = int(delta * MAX_ENCODER_DELTA)
            encoder_deltas[motor] = encoder_delta

            self.targets[motor] += encoder_delta
            # Relative safety limit
            lower_limit = self.initial_positions[motor] - MAX_ENCODER_TRAVEL
            upper_limit = self.initial_positions[motor] + MAX_ENCODER_TRAVEL
            self.targets[motor] = int(np.clip(self.targets[motor], lower_limit, upper_limit))
            targets_after[motor] = self.targets[motor]

            print(f"  Motor {motor}: action={float(delta):+.5f} delta={encoder_delta:+d} target={targets_before[motor]} -> {self.targets[motor]}")

        # Build and send synchronized packet
        self.group_sync_write.clearParam()
        for motor in self.motor_ids:
            target = self.targets[motor]
            param = [
                DXL_LOBYTE(DXL_LOWORD(target)),
                DXL_HIBYTE(DXL_LOWORD(target)),
                DXL_LOBYTE(DXL_HIWORD(target)),
                DXL_HIBYTE(DXL_HIWORD(target)),
            ]
            if not self.group_sync_write.addParam(motor, param):
                raise RuntimeError(f"Failed adding motor {motor}")

        print("  Sending synchronized motor command...")
        result = self.group_sync_write.txPacket()
        self.group_sync_write.clearParam()

        # Communication failure
        if result != COMM_SUCCESS:
            print("\n!!! COMMUNICATION FAILURE !!!")
            print(self.packet.getTxRxResult(result))

            telemetry = {motor: self._get_motor_telemetry(motor) for motor in self.motor_ids}
            self._log_action(
                action, encoder_deltas, targets_before, targets_after, telemetry,
                success=False, hardware_error=False,
                error_message=self.packet.getTxRxResult(result)
            )
            self.log_all_motor_diagnostics(reason="communication_failure")
            raise RuntimeError(f"Dynamixel communication failure: {self.packet.getTxRxResult(result)}")

        print("  Command transmitted successfully.")
        time.sleep(0.02)

        # Check Hardware Error Status
        hardware_error_ids = []
        hardware_error_status = {}
        packet_errors = {}

        for motor in self.motor_ids:
            status, packet_error = self.read_hardware_error_with_packet_error(motor)
            hardware_error_status[motor] = status
            packet_errors[motor] = packet_error

            if status is not None and status != 0:
                valid_hw_bits = 0x01 | 0x04 | 0x10 | 0x20
                if status & valid_hw_bits:
                    hardware_error_ids.append(motor)
                else:
                    print(f"Motor {motor}: Ignoring unknown hardware status 0x{status:02X}")

        # Read complete telemetry
        telemetry = {motor: self._get_motor_telemetry(motor) for motor in self.motor_ids}

        # Hardware error occurred
        if len(hardware_error_ids) > 0:
            message = f"Dynamixel hardware error detected. Motors: {hardware_error_ids}. Hardware status: {hardware_error_status}. Packet errors: {packet_errors}"
            print("\n" + "!" * 40)
            print(f"HARDWARE ERROR DURING ACTION #{self.action_count}")
            print(message)
            print("!" * 40)

            self._log_action(
                action, encoder_deltas, targets_before, targets_after, telemetry,
                packet_errors=packet_errors, success=False, hardware_error=True,
                hardware_error_ids=hardware_error_ids, error_message=message
            )

            for motor in self.motor_ids:
                self.log_motor_diagnostics(
                    motor,
                    reason=f"HARDWARE_ERROR_ACTION_{self.action_count}",
                    packet_error=packet_errors.get(motor),
                    hardware_status=hardware_error_status.get(motor),
                )

            return ExecutionResult(
                success=False,
                hardware_error=True,
                hardware_error_ids=hardware_error_ids,
                hardware_error_status=hardware_error_status,
                error_message=message,
            )

        # Successful action
        self._log_action(
            action, encoder_deltas, targets_before, targets_after, telemetry,
            packet_errors=packet_errors, success=True, hardware_error=False,
            hardware_error_ids=[], error_message=""
        )

        # Print telemetry
        print(f"\nACTION #{self.action_count} TELEMETRY")
        for motor in self.motor_ids:
            t = telemetry[motor]
            print(f"  Motor {motor}: pos={t['position']} target={self.targets[motor]} current={t['current']}mA voltage={t['voltage']}V temp={t['temperature']}C PWM={t['pwm']} velocity={t['velocity']} HW={t['hardware_status']}")

        print(f"  ACTION #{self.action_count} SUCCESS")
        print("=" * 60)

        if self.action_count % DIAGNOSTIC_LOG_EVERY_N_ACTIONS == 0:
            self.log_all_motor_diagnostics(reason=f"ACTION_{self.action_count}")

        return ExecutionResult(
            success=True,
            hardware_error=False,
            hardware_error_ids=[],
            hardware_error_status=hardware_error_status,
            error_message="",
        )

    # ========================================================
    # Read operations
    # ========================================================

    def read_position(self, motor_id):
        pos, dxl_comm_result, dxl_error = self.packet.read4ByteTxRx(self.port, motor_id, ADDR_PRESENT_POSITION)

        if dxl_comm_result != COMM_SUCCESS:
            print(f"Motor {motor_id}: Communication error reading position: {self.packet.getTxRxResult(dxl_comm_result)}")
            return None

        if dxl_error != 0:
            print(f"Motor {motor_id}: Packet error reading position: 0x{dxl_error:02X} ({self.packet.getRxPacketError(dxl_error)})")
            time.sleep(0.01)
            pos, dxl_comm_result, dxl_error = self.packet.read4ByteTxRx(self.port, motor_id, ADDR_PRESENT_POSITION)
            if dxl_comm_result == COMM_SUCCESS and dxl_error == 0:
                if pos > 0x7FFFFFFF:
                    pos -= 0x100000000
                return pos
            return None

        if pos > 0x7FFFFFFF:
            pos -= 0x100000000

        # Check for position corruption
        if motor_id in self.initial_positions:
            initial = self.initial_positions[motor_id]
            if pos is not None and abs(pos - initial) > 15000:
                print(f"WARNING: Motor {motor_id} position is corrupted!")
                print(f"  Position: {pos}, Initial: {initial}")

                # Try to fix by reading again
                time.sleep(0.01)
                pos2, comm2, error2 = self.packet.read4ByteTxRx(self.port, motor_id, ADDR_PRESENT_POSITION)
                if comm2 == COMM_SUCCESS and error2 == 0:
                    if pos2 > 0x7FFFFFFF:
                        pos2 -= 0x100000000
                    if abs(pos2 - initial) <= 15000:
                        print(f"  Second read: {pos2} (reasonable)")
                        return pos2
                    print(f"  Second read: {pos2} (still corrupted)")

                # Try to fix by writing initial position
                if pos is not None:
                    print(f"  Attempting to fix Motor {motor_id} position...")
                    self._write_goal_position_direct(motor_id, initial)
                    time.sleep(0.1)
                    pos3, comm3, error3 = self.packet.read4ByteTxRx(self.port, motor_id, ADDR_PRESENT_POSITION)
                    if comm3 == COMM_SUCCESS and error3 == 0:
                        if pos3 > 0x7FFFFFFF:
                            pos3 -= 0x100000000
                        if abs(pos3 - initial) <= 15000:
                            print(f"  Fixed! New position: {pos3}")
                            return pos3

        return pos

    def read_positions(self):
        positions = []
        for motor_id in self.motor_ids:
            position = self.read_position(motor_id)
            if position is None:
                return None
            positions.append(position)
        return positions

    def read_current(self, motor_id):
        current, dxl_comm_result, dxl_error = self.packet.read2ByteTxRx(self.port, motor_id, ADDR_PRESENT_CURRENT)

        if dxl_comm_result != COMM_SUCCESS:
            print(f"Communication error reading current from motor {motor_id}: {self.packet.getTxRxResult(dxl_comm_result)}")
            return None

        if dxl_error != 0:
            self._record_hardware_error(motor_id, dxl_error)
            return None

        if current >= 0x8000:
            current -= 0x10000
        return current

    def read_currents(self):
        currents = []
        for motor_id in self.motor_ids:
            current = self.read_current(motor_id)
            if current is None:
                time.sleep(0.05)
                current = self.read_current(motor_id)
                if current is None:
                    return None
            currents.append(current)
        return currents

    def read_voltage(self, motor_id):
        voltage_raw, comm, error = self.packet.read2ByteTxRx(self.port, motor_id, ADDR_PRESENT_INPUT_VOLTAGE)
        if comm != COMM_SUCCESS or error != 0:
            return None
        return voltage_raw * 0.1

    def read_voltages(self):
        voltages = []
        for motor_id in self.motor_ids:
            voltage = self.read_voltage(motor_id)
            if voltage is None:
                return None
            voltages.append(voltage)
        return np.array(voltages, dtype=np.float32)

    # ========================================================
    # Hardware error handling
    # ========================================================

    def read_hardware_error_with_packet_error(self, motor_id):
        status, dxl_comm_result, dxl_error = self.packet.read1ByteTxRx(self.port, motor_id, ADDR_HARDWARE_ERROR_STATUS)

        if dxl_comm_result != COMM_SUCCESS:
            print(f"Motor {motor_id}: COMMUNICATION failure while reading status: {self.packet.getTxRxResult(dxl_comm_result)}")
            return None, dxl_comm_result

        if dxl_error != 0:
            print(f"\nMotor {motor_id}: Protocol packet error while reading status: 0x{dxl_error:02X} ({self.packet.getRxPacketError(dxl_error)})")
            if status is not None:
                valid_hw_bits = 0x01 | 0x04 | 0x10 | 0x20
                if status & valid_hw_bits:
                    self._record_hardware_error(motor_id, status)
                    return status, dxl_error
                print(f"  Ignoring: Status=0x{status:02X} (not a valid hardware error)")
                return 0, dxl_error
            print(f"  No status received - communication issue")
            return 0, dxl_error

        if status is not None and status != 0:
            valid_hw_bits = 0x01 | 0x04 | 0x10 | 0x20
            if status & valid_hw_bits:
                self._record_hardware_error(motor_id, status)
                return status, None
            print(f"Motor {motor_id}: Ignoring unknown hardware status 0x{status:02X}")
            return 0, None

        return 0, None

    def read_hardware_error(self, motor_id):
        status, _ = self.read_hardware_error_with_packet_error(motor_id)
        return status

    def _record_hardware_error(self, motor_id, error_code, message=None):
        self.hardware_error = True
        if motor_id not in self.hardware_error_ids:
            self.hardware_error_ids.append(motor_id)

        self.hardware_error_status[motor_id] = error_code
        decoded_errors = self.decode_hardware_error(error_code)

        if message is None:
            message = f"Motor {motor_id}: hardware error status {error_code} (0x{error_code:02X}) - {', '.join(decoded_errors)}"

        self.hardware_error_message = message

        print("\n!!! DYNAMIXEL HARDWARE ERROR !!!")
        print(f"Motor ID: {motor_id}")
        print(f"Hardware Error Status: {error_code} (0x{error_code:02X})")
        print(f"Decoded: {', '.join(decoded_errors)}")

    def get_hardware_error_state(self):
        return (
            self.hardware_error,
            self.hardware_error_ids.copy(),
            self.hardware_error_status.copy(),
            self.hardware_error_message,
        )

    def clear_hardware_errors(self):
        self.hardware_error = False
        self.hardware_error_ids.clear()
        self.hardware_error_status.clear()
        self.hardware_error_message = None

    @staticmethod
    def format_hex(value):
        if value is None:
            return "None"
        return f"0x{int(value):02X}"

    @staticmethod
    def decode_hardware_error(status):
        """Decode XL330 Hardware Error Status (address 70)."""
        if status is None:
            return ["No hardware status available"]

        status = int(status)
        errors = []

        if status & 0x01:
            errors.append("Input voltage error")
        if status & 0x04:
            errors.append("Overheating")
        if status & 0x10:
            errors.append("Electrical shock")
        if status & 0x20:
            errors.append("Overload")

        known_mask = 0x01 | 0x04 | 0x10 | 0x20
        unknown_bits = status & ~known_mask
        if unknown_bits:
            errors.append(f"Unknown hardware bits 0x{unknown_bits:02X}")

        return errors if errors else ["No hardware error"]

    def move_to_positions_gradual(self, target_positions: dict, step_size: int = 200):
        """
        Move motors to target positions in small steps.
        """
        # Read current positions
        current = self.read_positions()
        if current is None:
            print("  Failed to read current positions")
            return
        
        current_dict = {motor: pos for motor, pos in zip(self.motor_ids, current)}
        
        # Calculate max steps needed
        max_dist = max(abs(target_positions.get(m, current_dict[m]) - current_dict[m]) 
                    for m in self.motor_ids)
        num_steps = max(1, int(max_dist / step_size) + 1)
        
        print(f"  Moving gradually in {num_steps} steps")
        
        for step in range(num_steps):
            progress = (step + 1) / num_steps
            for motor in self.motor_ids:
                target = target_positions.get(motor, current_dict[motor])
                intermediate = int(current_dict[motor] + (target - current_dict[motor]) * progress)
                self._write_goal_position_direct(motor, intermediate)
            time.sleep(0.05)
        
        print("  Gradual move complete")


    # In dynamixel_executor.py

    def recovery(self):
        """
        Simple recovery: attempt to re-establish communication with all motors.
        """
        print("\n================ RECOVERY ================")
        
        # 1. Wait for motors to settle
        print("Waiting for motors to settle...")
        time.sleep(3.0)
        
        # 2. Try to read positions from all motors
        print("Checking motor communication...")
        all_ok = True
        for motor in self.motor_ids:
            pos = self.read_position(motor)
            if pos is None:
                print(f"  Motor {motor}: No response")
                all_ok = False
                # Try waiting a little more
                time.sleep(2.0)
                pos = self.read_position(motor)
                if pos is not None:
                    print(f"  Motor {motor}: Recovered!")
                else:
                    print(f"  Motor {motor}: Still unresponsive")
                    return False
            else:
                print(f"  Motor {motor}: OK (pos={pos})")
        
        # 3. Clear hardware error flags
        self.clear_hardware_errors()
        
        print("Recovery complete.")
        print("==========================================\n")
        
        return all_ok

    # ========================================================
    # Shutdown
    # ========================================================

    def shutdown(self):
        print("\n========== EXECUTOR SHUTDOWN ==========")

        print("Disabling motors...")
        for motor in self.motor_ids:
            try:
                self.write1(motor, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
                print(f"  Motor {motor}: torque disabled")
            except Exception as e:
                print(f"  Motor {motor}: failed to disable torque: {e}")

        if hasattr(self, 'port') and self.port is not None:
            try:
                if self.port.is_open:
                    self.port.closePort()
                    print("Port closed")
            except Exception as e:
                print(f"Port close error: {e}")

        print("========== SHUTDOWN COMPLETE ==========\n")