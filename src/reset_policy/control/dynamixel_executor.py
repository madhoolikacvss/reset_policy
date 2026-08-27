"""
DynamixelExecutor: Low-level interface for controlling four Dynamixel XL330 motors.

Responsibilities:
- Initialize Dynamixels
- Convert RL actions into relative encoder movements
- Execute synchronized motor commands
- Read encoder positions and currents
- Read voltage, temperature, torque, PWM, velocity
- Decode hardware errors
- Log motor diagnostics
- Enforce relative position limits
- Shutdown safely

Note: Hardware errors are FATAL - no recovery is attempted.
"""

from __future__ import annotations

import csv
import time
import numpy as np

from dataclasses import dataclass
from datetime import datetime

from dynamixel_sdk import *

from reset_policy.config import config


# Dynamixel addresses (from config)
ADDR_OPERATING_MODE = config.dynamixel.addr_operating_mode
ADDR_TORQUE_ENABLE = config.dynamixel.addr_torque_enable
ADDR_HARDWARE_ERROR_STATUS = config.dynamixel.addr_hardware_error_status
ADDR_PRESENT_PWM = config.dynamixel.addr_present_pwm
ADDR_PRESENT_CURRENT = config.dynamixel.addr_present_current
ADDR_PRESENT_VELOCITY = config.dynamixel.addr_present_velocity
ADDR_PRESENT_POSITION = config.dynamixel.addr_present_position
ADDR_PRESENT_INPUT_VOLTAGE = config.dynamixel.addr_present_input_voltage
ADDR_PRESENT_TEMPERATURE = config.dynamixel.addr_present_temperature
ADDR_GOAL_POSITION = config.dynamixel.addr_goal_position

# Constants (from config)
EXTENDED_POSITION_MODE = config.dynamixel.extended_position_mode
TORQUE_ENABLE = config.dynamixel.torque_enable
TORQUE_DISABLE = config.dynamixel.torque_disable
MAX_ENCODER_DELTA = config.dynamixel.max_encoder_delta
MAX_ENCODER_TRAVEL = config.dynamixel.max_encoder_travel

# Valid hardware error bits (Voltage, Temp, Shock, Overload)
VALID_HW_ERROR_BITS = 0x01 | 0x04 | 0x10 | 0x20

# Position corruption threshold
POSITION_CORRUPTION_THRESHOLD = 15000


# Execution result
@dataclass
class ExecutionResult:
    """Result of executing an RL action."""
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


# Dynamixel Executor
class DynamixelExecutor:
    """Low-level Dynamixel motor controller."""
    
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

        # Logging
        self.diagnostic_log_file = config.training_log_dir / "motor_diagnostics.csv"
        self.action_log_file = config.training_log_dir / "action_log.csv"
        self._initialize_logs()

    # Logging initialization    
    def _initialize_logs(self):
        """Initialize CSV log files."""
        try:
            config.training_log_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"WARNING: Could not create log directory: {e}")
        
        self._initialize_diagnostic_log()
        self._initialize_action_log()
    
    def _initialize_diagnostic_log(self):
        """Create motor diagnostic CSV if it doesn't exist."""
        try:
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
        """Create action-level CSV if it doesn't exist."""
        try:
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

    # Raw read operations    
    def _read_raw(self, motor_id, address, size, signed=False):
        """Generic read with retry."""
        for attempt in range(2):  # Try twice
            if size == 1:
                value, comm, error = self.packet.read1ByteTxRx(self.port, motor_id, address)
                if signed and value is not None and value >= 0x80:
                    value -= 0x100
            elif size == 2:
                value, comm, error = self.packet.read2ByteTxRx(self.port, motor_id, address)
                if signed and value is not None and value >= 0x8000:
                    value -= 0x10000
            elif size == 4:
                value, comm, error = self.packet.read4ByteTxRx(self.port, motor_id, address)
                if signed and value is not None and value >= 0x80000000:
                    value -= 0x100000000
            else:
                return None
            
            if comm == COMM_SUCCESS and error == 0:
                return value
            
            if attempt == 0:
                time.sleep(0.02)  # Small delay before retry
        
        return None
    
    def _read1_raw(self, motor_id, address):
        """Read 1 byte."""
        return self._read_raw(motor_id, address, 1)
    
    def _read2_raw(self, motor_id, address):
        """Read 2 bytes."""
        return self._read_raw(motor_id, address, 2)
    
    def _read4_raw(self, motor_id, address):
        """Read 4 bytes (signed)."""
        return self._read_raw(motor_id, address, 4, signed=True)

    # Telemetry    
    def _get_motor_telemetry(self, motor_id):
        """Read complete telemetry snapshot for one motor."""
        return {
            "position": self._read4_raw(motor_id, ADDR_PRESENT_POSITION),
            "current": self._read_raw(motor_id, ADDR_PRESENT_CURRENT, 2, signed=True),
            "voltage": self._read_raw(motor_id, ADDR_PRESENT_INPUT_VOLTAGE, 2) * 0.1 
                      if self._read_raw(motor_id, ADDR_PRESENT_INPUT_VOLTAGE, 2) is not None else None,
            "temperature": self._read1_raw(motor_id, ADDR_PRESENT_TEMPERATURE),
            "torque": self._read1_raw(motor_id, ADDR_TORQUE_ENABLE),
            "pwm": self._read_raw(motor_id, ADDR_PRESENT_PWM, 2, signed=True),
            "velocity": self._read_raw(motor_id, ADDR_PRESENT_VELOCITY, 4, signed=True),
            "hardware_status": self._read1_raw(motor_id, ADDR_HARDWARE_ERROR_STATUS),
        }

    # High-level read operations    
    def read_position(self, motor_id):
        """Read position for one motor with corruption check."""
        pos = self._read4_raw(motor_id, ADDR_PRESENT_POSITION)
        
        if pos is None:
            print(f"Motor {motor_id}: Failed to read position")
            return None
        
        # Check for position corruption
        if motor_id in self.initial_positions:
            initial = self.initial_positions[motor_id]
            if abs(pos - initial) > POSITION_CORRUPTION_THRESHOLD:
                print(f"WARNING: Motor {motor_id} position corrupted!")
                print(f"  Position: {pos}, Initial: {initial}")
                
                # Try to recover by re-reading
                time.sleep(0.01)
                pos = self._read4_raw(motor_id, ADDR_PRESENT_POSITION)
                if pos is not None and abs(pos - initial) <= POSITION_CORRUPTION_THRESHOLD:
                    print(f"  Recovered: {pos}")
                    return pos
                
                # Try to fix by writing initial position
                print(f"  Attempting position fix...")
                self._write_goal_position_direct(motor_id, initial)
                time.sleep(0.1)
                pos = self._read4_raw(motor_id, ADDR_PRESENT_POSITION)
                if pos is not None and abs(pos - initial) <= POSITION_CORRUPTION_THRESHOLD:
                    print(f"  Fixed: {pos}")
                    return pos
        
        return pos
    
    def read_positions(self):
        """Read positions for all motors."""
        positions = []
        for motor_id in self.motor_ids:
            pos = self.read_position(motor_id)
            if pos is None:
                return None
            positions.append(pos)
        return positions
    
    def read_current(self, motor_id):
        """Read current for one motor (mA, signed)."""
        return self._read_raw(motor_id, ADDR_PRESENT_CURRENT, 2, signed=True)
    
    def read_currents(self):
        """Read currents for all motors."""
        currents = []
        for motor_id in self.motor_ids:
            current = self.read_current(motor_id)
            if current is None:
                return None
            currents.append(current)
        return currents
    
    def read_voltage(self, motor_id):
        """Read voltage for one motor (V)."""
        voltage_raw = self._read_raw(motor_id, ADDR_PRESENT_INPUT_VOLTAGE, 2)
        return voltage_raw * 0.1 if voltage_raw is not None else None
    
    def read_voltages(self):
        """Read voltages for all motors."""
        voltages = []
        for motor_id in self.motor_ids:
            voltage = self.read_voltage(motor_id)
            if voltage is None:
                return None
            voltages.append(voltage)
        return np.array(voltages, dtype=np.float32)
    
    def read_temperature(self, motor_id):
        """Read temperature for one motor (°C)."""
        temp = self._read1_raw(motor_id, ADDR_PRESENT_TEMPERATURE)
        return float(temp) if temp is not None else None
    
    def read_temperatures(self):
        """Read temperatures for all motors."""
        temperatures = []
        for motor_id in self.motor_ids:
            temp = self.read_temperature(motor_id)
            if temp is None:
                return None
            temperatures.append(temp)
        return np.array(temperatures, dtype=np.float32)

    # Write operations    
    def write1(self, motor_id, address, value):
        """Write 1 byte with error checking."""
        comm, error = self.packet.write1ByteTxRx(self.port, motor_id, address, value)
        if comm != COMM_SUCCESS:
            raise RuntimeError(f"Motor {motor_id}: {self.packet.getTxRxResult(comm)}")
        if error != 0:
            raise RuntimeError(f"Motor {motor_id}: {self.packet.getRxPacketError(error)}")
    
    def _write_goal_position_direct(self, motor_id, target_position):
        """Write goal position with error handling."""
        target_position = int(np.clip(target_position, -2147483648, 2147483647))
        
        comm, error = self.packet.write4ByteTxRx(
            self.port, motor_id, ADDR_GOAL_POSITION, target_position
        )
        
        if comm != COMM_SUCCESS or error != 0:
            print(f"Motor {motor_id}: Write failed")
            return False
        
        self.targets[motor_id] = target_position
        return True
    
    def move_motor_by_delta(self, motor_id, delta, max_delta=100):
        """Move a single motor by encoder delta."""
        delta = max(-max_delta, min(max_delta, delta))
        
        if motor_id in self.targets:
            self.targets[motor_id] += delta
            
            # Clamp to position limits
            lower_limit = self.initial_positions[motor_id] - MAX_ENCODER_TRAVEL
            upper_limit = self.initial_positions[motor_id] + MAX_ENCODER_TRAVEL
            self.targets[motor_id] = int(np.clip(
                self.targets[motor_id], lower_limit, upper_limit
            ))
        
        # Send single motor command
        return self._send_sync_write({motor_id: self.targets[motor_id]})
    
    def _send_sync_write(self, targets_dict):
        """Send synchronized write to multiple motors."""
        self.group_sync_write.clearParam()
        
        for motor_id, target in targets_dict.items():
            param = [
                DXL_LOBYTE(DXL_LOWORD(target)),
                DXL_HIBYTE(DXL_LOWORD(target)),
                DXL_LOBYTE(DXL_HIWORD(target)),
                DXL_HIBYTE(DXL_HIWORD(target)),
            ]
            if not self.group_sync_write.addParam(motor_id, param):
                print(f"ERROR: Failed to add motor {motor_id}")
                return False
        
        result = self.group_sync_write.txPacket()
        self.group_sync_write.clearParam()
        
        if result != COMM_SUCCESS:
            print(f"Sync write failed: {self.packet.getTxRxResult(result)}")
            return False
        
        return True

    # Initialization    
    def initialize(self):
        """Initialize all motors."""
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

    # Execute RL action    
    def execute(self, action):
        """Execute RL action."""
        if len(action) != len(self.motor_ids):
            raise ValueError("Action dimension does not match motors")
        
        self.action_count += 1
        action = np.asarray(action, dtype=np.float32)
        
        print(f"\n{'='*60}")
        print(f"RL ACTION #{self.action_count}")
        print(f"raw action: {action}")
        print("=" * 60)
        
        # Save targets before
        targets_before = {m: int(self.targets[m]) for m in self.motor_ids}
        
        # Convert action to encoder deltas
        encoder_deltas = {}
        targets_after = {}
        
        for motor, action_val in zip(self.motor_ids, action):
            action_val = np.clip(action_val, -1.0, 1.0)
            encoder_delta = int(action_val * MAX_ENCODER_DELTA)
            encoder_deltas[motor] = encoder_delta
            
            self.targets[motor] += encoder_delta
            lower_limit = self.initial_positions[motor] - MAX_ENCODER_TRAVEL
            upper_limit = self.initial_positions[motor] + MAX_ENCODER_TRAVEL
            self.targets[motor] = int(np.clip(self.targets[motor], lower_limit, upper_limit))
            targets_after[motor] = self.targets[motor]
            
            print(f"  Motor {motor}: action={action_val:+.5f} "
                  f"delta={encoder_delta:+d} "
                  f"target={targets_before[motor]} -> {self.targets[motor]}")
        
        # Send synchronized command
        if not self._send_sync_write(targets_after):
            print("\n!!! COMMUNICATION FAILURE !!!")
            telemetry = {m: self._get_motor_telemetry(m) for m in self.motor_ids}
            self._log_action(
                action, encoder_deltas, targets_before, targets_after, telemetry,
                success=False, error_message="Communication failure"
            )
            raise RuntimeError("Dynamixel communication failure")
        
        time.sleep(0.02)
        
        # Check hardware errors
        hardware_error_ids = []
        hardware_error_status = {}
        packet_errors = {}
        
        for motor in self.motor_ids:
            status = self._read1_raw(motor, ADDR_HARDWARE_ERROR_STATUS)
            hardware_error_status[motor] = status
            
            if status is not None and status & VALID_HW_ERROR_BITS:
                hardware_error_ids.append(motor)
        
        # Read telemetry
        telemetry = {m: self._get_motor_telemetry(m) for m in self.motor_ids}
        
        # Handle hardware errors
        if hardware_error_ids:
            message = (f"Dynamixel hardware error. Motors: {hardware_error_ids}. "
                      f"Status: {hardware_error_status}")
            print("\n" + "!" * 40)
            print(f"HARDWARE ERROR DURING ACTION #{self.action_count}")
            print(message)
            print("!" * 40)
            
            self._log_action(
                action, encoder_deltas, targets_before, targets_after, telemetry,
                packet_errors=packet_errors, success=False, hardware_error=True,
                hardware_error_ids=hardware_error_ids, error_message=message
            )
            
            return ExecutionResult(
                success=False,
                hardware_error=True,
                hardware_error_ids=hardware_error_ids,
                hardware_error_status=hardware_error_status,
                error_message=message,
            )
        
        # Success
        self._log_action(
            action, encoder_deltas, targets_before, targets_after, telemetry,
            packet_errors=packet_errors, success=True
        )
        
        # Print telemetry
        print(f"\nACTION #{self.action_count} TELEMETRY")
        for motor in self.motor_ids:
            t = telemetry[motor]
            print(f"  Motor {motor}: pos={t['position']} target={self.targets[motor]} "
                  f"current={t['current']}mA voltage={t['voltage']}V "
                  f"temp={t['temperature']}°C PWM={t['pwm']} "
                  f"velocity={t['velocity']} HW={t['hardware_status']}")
        
        print(f"  ACTION #{self.action_count} SUCCESS")
        
        return ExecutionResult(success=True)
    
    # Hardware error handling    
    def _record_hardware_error(self, motor_id, error_code):
        """Record hardware error for a motor."""
        self.hardware_error = True
        if motor_id not in self.hardware_error_ids:
            self.hardware_error_ids.append(motor_id)
        
        self.hardware_error_status[motor_id] = error_code
        decoded = self.decode_hardware_error(error_code)
        self.hardware_error_message = (
            f"Motor {motor_id}: status {error_code} (0x{error_code:02X}) - {', '.join(decoded)}"
        )
        
        print("\n!!! DYNAMIXEL HARDWARE ERROR !!!")
        print(f"Motor ID: {motor_id}")
        print(f"Status: {error_code} (0x{error_code:02X})")
        print(f"Decoded: {', '.join(decoded)}")
    
    def get_hardware_error_state(self):
        """Get hardware error state."""
        return (
            self.hardware_error,
            self.hardware_error_ids.copy(),
            self.hardware_error_status.copy(),
            self.hardware_error_message,
        )
    
    def clear_hardware_errors(self):
        """Clear hardware error state."""
        self.hardware_error = False
        self.hardware_error_ids.clear()
        self.hardware_error_status.clear()
        self.hardware_error_message = None
    
    @staticmethod
    def format_hex(value):
        """Format value as hex string."""
        return "None" if value is None else f"0x{int(value):02X}"
    
    @staticmethod
    def decode_hardware_error(status):
        """Decode XL330 hardware error status."""
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
        
        unknown_bits = status & ~VALID_HW_ERROR_BITS
        if unknown_bits:
            errors.append(f"Unknown bits 0x{unknown_bits:02X}")
        
        return errors if errors else ["No hardware error"]
    
    # Logging    
    def log_motor_diagnostics(self, motor_id, reason="periodic", packet_error=None, hardware_status=None):
        """Log diagnostic data for one motor."""
        timestamp = datetime.now().isoformat()
        telemetry = self._get_motor_telemetry(motor_id)
        
        if hardware_status is None:
            hardware_status = telemetry["hardware_status"]
        
        decoded = self.decode_hardware_error(hardware_status)
        
        try:
            with open(self.diagnostic_log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp, self.action_count, motor_id,
                    self.targets.get(motor_id), telemetry["position"],
                    telemetry["current"], telemetry["voltage"], telemetry["temperature"],
                    telemetry["torque"], telemetry["pwm"], telemetry["velocity"],
                    hardware_status, self.format_hex(hardware_status),
                    packet_error, self.format_hex(packet_error),
                    ", ".join(decoded),
                ])
        except Exception as e:
            print(f"WARNING: failed to write diagnostic log: {e}")
    
    def log_all_motor_diagnostics(self, reason="periodic", packet_errors=None, hardware_statuses=None):
        """Log diagnostics for all motors."""
        packet_errors = packet_errors or {}
        hardware_statuses = hardware_statuses or {}
        
        for motor in self.motor_ids:
            self.log_motor_diagnostics(
                motor,
                reason=reason,
                packet_error=packet_errors.get(motor),
                hardware_status=hardware_statuses.get(motor),
            )
    
    def _log_action(self, action, encoder_deltas, targets_before, targets_after, telemetry,
                    packet_errors=None, success=True, hardware_error=False,
                    hardware_error_ids=None, error_message=""):
        """Write one CSV row for one RL action."""
        packet_errors = packet_errors or {}
        hardware_error_ids = hardware_error_ids or []
        
        timestamp = datetime.now().isoformat()
        row = [timestamp, self.action_count]
        
        # RL action
        row.extend([float(a) for a in action])
        
        # Encoder deltas
        row.extend([encoder_deltas[m] for m in self.motor_ids])
        
        # Targets
        row.extend([targets_before[m] for m in self.motor_ids])
        row.extend([targets_after[m] for m in self.motor_ids])
        
        # Telemetry
        for key in ["position", "current", "voltage", "temperature", "pwm", "velocity", "hardware_status"]:
            row.extend([telemetry[m][key] for m in self.motor_ids])
        
        # Packet errors
        row.extend([packet_errors.get(m) for m in self.motor_ids])
        
        # Result
        row.extend([success, hardware_error, ",".join(map(str, hardware_error_ids)), error_message])
        
        try:
            with open(self.action_log_file, "a", newline="") as f:
                csv.writer(f).writerow(row)
        except Exception as e:
            print(f"WARNING: failed to write action log: {e}")
    
    # Shutdown    
    def shutdown(self):
        """Shutdown all motors safely using sync write."""
        print("\n========== EXECUTOR SHUTDOWN ==========")
        
        # Wait for any in-progress command to complete
        print("Waiting for in-progress commands to complete...")
        time.sleep(2.0)
        
        # Check port availability
        try:
            port_available = self.port.is_open if self.port else False
        except:
            port_available = False
        
        if not port_available:
            print("Port not available - cannot disable torque")
            return
        
        print("Disabling all motors via sync write...")
        
        # Use sync write to disable all motors at once
        try:
            # Create a temporary sync write for torque enable (address 64, 1 byte)
            # Note: Your current GroupSyncWrite is for goal position (address 116, 4 bytes)
            # We need a different approach for 1-byte writes
            
            # Method 1: Individual writes with retry
            for motor in self.motor_ids:
                for attempt in range(3):  # Try 3 times
                    try:
                        comm, error = self.packet.write1ByteTxRx(
                            self.port,
                            motor,
                            ADDR_TORQUE_ENABLE,
                            TORQUE_DISABLE
                        )
                        
                        if comm == COMM_SUCCESS and error == 0:
                            print(f"  Motor {motor}: torque disabled")
                            break
                        else:
                            if attempt < 2:
                                print(f"  Motor {motor}: retry {attempt+1}...")
                                time.sleep(0.5)
                            else:
                                print(f"  Motor {motor}: failed after 3 attempts")
                    except Exception as e:
                        if attempt < 2:
                            print(f"  Motor {motor}: exception, retrying...")
                            time.sleep(0.5)
                        else:
                            print(f"  Motor {motor}: failed - {e}")
        except Exception as e:
            print(f"Torque disable error: {e}")
        
        # Close port
        try:
            if self.port and self.port.is_open:
                self.port.closePort()
                print("Port closed")
        except Exception as e:
            print(f"Port close error: {e}")
        
        print("========== SHUTDOWN COMPLETE ==========\n")