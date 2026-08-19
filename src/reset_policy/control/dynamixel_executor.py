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

# Protocol 2.0 reboot instruction
ADDR_REBOOT = 0x08


# ============================================================
# Constants
# ============================================================

EXTENDED_POSITION_MODE = 4

TORQUE_ENABLE = 1
TORQUE_DISABLE = 0


# ============================================================
# RL action scaling
# ============================================================

# RL action [-1, 1] -> encoder delta
MAX_ENCODER_DELTA = 400


# ============================================================
# Safety
# ============================================================

MAX_ENCODER_TRAVEL = 10000


# ============================================================
# Diagnostic logging
# ============================================================

# ============================================================
# Diagnostic logging
# ============================================================

DIAGNOSTIC_LOG_EVERY_N_ACTIONS = 1

DIAGNOSTIC_LOG_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "scripts"
    / "training_logs"
)

DIAGNOSTIC_LOG_FILE = (
    DIAGNOSTIC_LOG_DIR
    / "motor_diagnostics.csv"
)

ACTION_LOG_FILE = (
    DIAGNOSTIC_LOG_DIR
    / "action_log.csv"
)


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

        # ----------------------------------------------------
        # Position tracking
        # ----------------------------------------------------

        self.targets = {}

        self.initial_positions = {}

        # ----------------------------------------------------
        # Hardware error bookkeeping
        # ----------------------------------------------------

        self.hardware_error = False

        self.hardware_error_ids = []

        self.hardware_error_status = {}

        self.hardware_error_message = None

        # ----------------------------------------------------
        # Action counter
        # ----------------------------------------------------

        self.action_count = 0

        # ----------------------------------------------------
        # Diagnostic logging
        # ----------------------------------------------------

        self.diagnostic_log_file = DIAGNOSTIC_LOG_FILE

        self.action_log_file = ACTION_LOG_FILE

        self._initialize_diagnostic_log()

        self._initialize_action_log()

        from reset_policy.control.motor_logger import MotorLogger
        self.logger = MotorLogger(
            port_handler=port_handler,
            packet_handler=packet_handler,
            motor_ids=motor_ids,
            log_frequency_hz=100.0,  # Log at 100 Hz
            buffer_size=50000,
        )
        self.logger.start()

        self.current_episode = 0
        self.current_step = 0
        


    # ========================================================
    # Diagnostic logging
    # ========================================================

    def update_context(self, episode: int, step: int, action: np.ndarray = None,
                       cube_x: float = None, cube_y: float = None):
        """Update context from environment."""
        self.current_episode = episode
        self.current_step = step
        if hasattr(self, 'logger'):
            self.logger.update_context(
                episode=episode,
                step=step,
                action=action,
                cube_x=cube_x,
                cube_y=cube_y,
            )

    def _initialize_diagnostic_log(self):

        """
        Create the motor diagnostic CSV if it does not exist.
        """

        try:

            self.diagnostic_log_file.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            if not self.diagnostic_log_file.exists():

                with open(
                    self.diagnostic_log_file,
                    "w",
                    newline="",
                ) as f:

                    writer = csv.writer(f)

                    writer.writerow([
                        "timestamp",
                        "action_count",
                        "motor_id",

                        "target_position",
                        "present_position",

                        "present_current_mA",
                        "present_voltage_V",
                        "temperature_C",

                        "torque_enabled",

                        "present_pwm",
                        "present_velocity",

                        "hardware_error_status",
                        "hardware_error_hex",

                        "packet_error",
                        "packet_error_hex",

                        "hardware_error_decoded",

                    ])

        except Exception as e:

            print(
                f"WARNING: Could not initialize "
                f"diagnostic log: {e}"
            )

    def _initialize_action_log(self):

        """
        Create the action-level CSV if it does not exist.

        One row corresponds to ONE RL action.

        This allows us to reconstruct exactly what the policy
        commanded immediately before a hardware failure.
        """

        try:

            self.diagnostic_log_file.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            if not self.action_log_file.exists():

                with open(
                    self.action_log_file,
                    "w",
                    newline="",
                ) as f:

                    writer = csv.writer(f)

                    writer.writerow([
                        "timestamp",
                        "action_count",

                        # RL action
                        "action_m16",
                        "action_m17",
                        "action_m18",
                        "action_m19",

                        # Encoder deltas
                        "delta_m16",
                        "delta_m17",
                        "delta_m18",
                        "delta_m19",

                        # Targets before action
                        "target_before_m16",
                        "target_before_m17",
                        "target_before_m18",
                        "target_before_m19",

                        # Targets after action
                        "target_after_m16",
                        "target_after_m17",
                        "target_after_m18",
                        "target_after_m19",

                        # Actual motor positions
                        "position_m16",
                        "position_m17",
                        "position_m18",
                        "position_m19",

                        # Current
                        "current_m16_mA",
                        "current_m17_mA",
                        "current_m18_mA",
                        "current_m19_mA",

                        # Voltage
                        "voltage_m16_V",
                        "voltage_m17_V",
                        "voltage_m18_V",
                        "voltage_m19_V",

                        # Temperature
                        "temperature_m16_C",
                        "temperature_m17_C",
                        "temperature_m18_C",
                        "temperature_m19_C",

                        # PWM
                        "pwm_m16",
                        "pwm_m17",
                        "pwm_m18",
                        "pwm_m19",

                        # Velocity
                        "velocity_m16",
                        "velocity_m17",
                        "velocity_m18",
                        "velocity_m19",

                        # Hardware status
                        "hardware_status_m16",
                        "hardware_status_m17",
                        "hardware_status_m18",
                        "hardware_status_m19",

                        # Packet errors
                        "packet_error_m16",
                        "packet_error_m17",
                        "packet_error_m18",
                        "packet_error_m19",

                        # Execution result
                        "success",
                        "hardware_error",
                        "hardware_error_ids",
                        "error_message",
                    ])

        except Exception as e:

            print(
                f"WARNING: Could not initialize "
                f"action log: {e}"
            )


    def log_motor_diagnostics(
        self,
        motor_id,
        reason="periodic",
        packet_error=None,
        hardware_status=None,
    ):

        """
        Read and log important motor telemetry.

        This is intentionally independent of observation.py.
        It is a low-level hardware diagnostic snapshot.

        Logged information:
            - Position
            - Current
            - Input voltage
            - Temperature
            - Torque enable
            - PWM
            - Velocity
            - Hardware error status
            - Protocol packet error
        """

        timestamp = datetime.now().isoformat()

        # ----------------------------------------------------
        # Read position
        # ----------------------------------------------------

        position = self._read4_raw(
            motor_id,
            ADDR_PRESENT_POSITION,
        )

        # ----------------------------------------------------
        # Read current
        # ----------------------------------------------------

        current = self._read2_raw(
            motor_id,
            ADDR_PRESENT_CURRENT,
        )

        if current is not None and current >= 0x8000:
            current -= 0x10000

        # ----------------------------------------------------
        # Read voltage
        # ----------------------------------------------------

        voltage_raw = self._read2_raw(
            motor_id,
            ADDR_PRESENT_INPUT_VOLTAGE,
        )

        voltage = None

        if voltage_raw is not None:
            voltage = voltage_raw * 0.1

        # ----------------------------------------------------
        # Read temperature
        # ----------------------------------------------------

        temperature = self._read1_raw(
            motor_id,
            ADDR_PRESENT_TEMPERATURE,
        )

        # ----------------------------------------------------
        # Read torque enable
        # ----------------------------------------------------

        torque = self._read1_raw(
            motor_id,
            ADDR_TORQUE_ENABLE,
        )

        # ----------------------------------------------------
        # Read PWM
        # ----------------------------------------------------

        pwm = self._read2_raw(
            motor_id,
            ADDR_PRESENT_PWM,
        )

        if pwm is not None and pwm >= 0x8000:
            pwm -= 0x10000

        # ----------------------------------------------------
        # Read velocity
        # ----------------------------------------------------

        velocity = self._read4_raw(
            motor_id,
            ADDR_PRESENT_VELOCITY,
        )

        if velocity is not None and velocity >= 0x80000000:
            velocity -= 0x100000000

        # ----------------------------------------------------
        # Hardware status
        # ----------------------------------------------------

        if hardware_status is None:

            hardware_status = (
                self._read1_raw(
                    motor_id,
                    ADDR_HARDWARE_ERROR_STATUS,
                )
            )

        decoded = self.decode_hardware_error(
            hardware_status
        )

        # ----------------------------------------------------
        # Console output
        # ----------------------------------------------------

        print(
            "\n"
            f"[MOTOR DIAGNOSTICS] "
            f"reason={reason} | "
            f"motor={motor_id}\n"
            f"  position : {position}\n"
            f"  target   : {self.targets.get(motor_id)}\n"
            f"  current  : {current} mA\n"
            f"  voltage  : {voltage} V\n"
            f"  temp     : {temperature} C\n"
            f"  torque   : {torque}\n"
            f"  PWM      : {pwm}\n"
            f"  velocity : {velocity}\n"
            f"  HW status: "
            f"{hardware_status} "
            f"({self.format_hex(hardware_status)})\n"
            f"  decoded  : "
            f"{', '.join(decoded)}\n"
            f"  packet error: "
            f"{packet_error} "
            f"({self.format_hex(packet_error)})"
        )

        # ----------------------------------------------------
        # CSV logging
        # ----------------------------------------------------

        try:

            with open(
                self.diagnostic_log_file,
                "a",
                newline="",
            ) as f:

                writer = csv.writer(f)

                writer.writerow([
                    timestamp,
                    self.action_count,
                    motor_id,

                    self.targets.get(motor_id),
                    position,

                    current,
                    voltage,
                    temperature,

                    torque,

                    pwm,
                    velocity,

                    hardware_status,
                    self.format_hex(hardware_status),

                    packet_error,
                    self.format_hex(packet_error),

                    ", ".join(decoded),
                ])

        except Exception as e:

            print(
                f"WARNING: failed to write "
                f"diagnostic log: {e}"
            )


    def log_all_motor_diagnostics(
        self,
        reason="periodic",
        packet_errors=None,
        hardware_statuses=None,
    ):

        """
        Capture diagnostics for every motor.

        Used periodically and whenever a hardware error occurs.
        """

        if packet_errors is None:
            packet_errors = {}

        if hardware_statuses is None:
            hardware_statuses = {}

        print(
            "\n"
            "================================================"
        )
        print(
            f"MOTOR DIAGNOSTIC SNAPSHOT: {reason}"
        )
        print(
            "================================================"
        )

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

    def _read1_raw(
        self,
        motor_id,
        address,
    ):

        value, comm, error = (
            self.packet.read1ByteTxRx(
                self.port,
                motor_id,
                address,
            )
        )

        if comm != COMM_SUCCESS:
            return None

        if error != 0:
            return None

        return value


    def _read2_raw(
        self,
        motor_id,
        address,
    ):

        value, comm, error = (
            self.packet.read2ByteTxRx(
                self.port,
                motor_id,
                address,
            )
        )

        if comm != COMM_SUCCESS:
            return None

        if error != 0:
            return None

        return value


    def _read4_raw(
        self,
        motor_id,
        address,
    ):

        value, comm, error = (
            self.packet.read4ByteTxRx(
                self.port,
                motor_id,
                address,
            )
        )

        if comm != COMM_SUCCESS:
            return None

        if error != 0:
            return None

        return value

    def _get_motor_telemetry(self, motor_id):

        """
        Read a complete low-level telemetry snapshot for one motor.

        Returns a dictionary.

        This is used for action-level logging.
        """

        position = self._read4_raw(
            motor_id,
            ADDR_PRESENT_POSITION,
        )

        if position is not None and position >= 0x80000000:
            position -= 0x100000000

        current = self._read2_raw(
            motor_id,
            ADDR_PRESENT_CURRENT,
        )

        if current is not None and current >= 0x8000:
            current -= 0x10000

        voltage_raw = self._read2_raw(
            motor_id,
            ADDR_PRESENT_INPUT_VOLTAGE,
        )

        voltage = None

        if voltage_raw is not None:
            voltage = voltage_raw * 0.1

        temperature = self._read1_raw(
            motor_id,
            ADDR_PRESENT_TEMPERATURE,
        )

        torque = self._read1_raw(
            motor_id,
            ADDR_TORQUE_ENABLE,
        )

        pwm = self._read2_raw(
            motor_id,
            ADDR_PRESENT_PWM,
        )

        if pwm is not None and pwm >= 0x8000:
            pwm -= 0x10000

        velocity = self._read4_raw(
            motor_id,
            ADDR_PRESENT_VELOCITY,
        )

        if velocity is not None and velocity >= 0x80000000:
            velocity -= 0x100000000

        hardware_status = self._read1_raw(
            motor_id,
            ADDR_HARDWARE_ERROR_STATUS,
        )

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

    def _log_action(
        self,
        action,
        encoder_deltas,
        targets_before,
        targets_after,
        telemetry,
        packet_errors=None,
        success=True,
        hardware_error=False,
        hardware_error_ids=None,
        error_message="",
    ):

        """
        Write ONE CSV row for ONE RL action.

        This is the primary experiment-level action log.
        """

        if packet_errors is None:
            packet_errors = {}

        if hardware_error_ids is None:
            hardware_error_ids = []

        timestamp = datetime.now().isoformat()

        row = [
            timestamp,
            self.action_count,
        ]

        # ----------------------------------------------------
        # RL action
        # ----------------------------------------------------

        row.extend([
            float(action[0]),
            float(action[1]),
            float(action[2]),
            float(action[3]),
        ])

        # ----------------------------------------------------
        # Encoder deltas
        # ----------------------------------------------------

        for motor in self.motor_ids:
            row.append(
                encoder_deltas[motor]
            )

        # ----------------------------------------------------
        # Targets before
        # ----------------------------------------------------

        for motor in self.motor_ids:
            row.append(
                targets_before[motor]
            )

        # ----------------------------------------------------
        # Targets after
        # ----------------------------------------------------

        for motor in self.motor_ids:
            row.append(
                targets_after[motor]
            )

        # ----------------------------------------------------
        # Telemetry
        # ----------------------------------------------------

        for motor in self.motor_ids:
            row.append(
                telemetry[motor]["position"]
            )

        for motor in self.motor_ids:
            row.append(
                telemetry[motor]["current"]
            )

        for motor in self.motor_ids:
            row.append(
                telemetry[motor]["voltage"]
            )

        for motor in self.motor_ids:
            row.append(
                telemetry[motor]["temperature"]
            )

        for motor in self.motor_ids:
            row.append(
                telemetry[motor]["pwm"]
            )

        for motor in self.motor_ids:
            row.append(
                telemetry[motor]["velocity"]
            )

        # ----------------------------------------------------
        # Hardware errors
        # ----------------------------------------------------

        for motor in self.motor_ids:

            row.append(
                telemetry[motor]["hardware_status"]
            )

        # ----------------------------------------------------
        # Packet errors
        # ----------------------------------------------------

        for motor in self.motor_ids:

            row.append(
                packet_errors.get(motor)
            )

        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        row.extend([
            success,
            hardware_error,
            ",".join(
                str(x)
                for x in hardware_error_ids
            ),
            error_message,
        ])

        # ----------------------------------------------------
        # Write
        # ----------------------------------------------------

        try:

            with open(
                self.action_log_file,
                "a",
                newline="",
            ) as f:

                writer = csv.writer(f)

                writer.writerow(row)

        except Exception as e:

            print(
                f"WARNING: failed to write "
                f"action log: {e}"
            )

    # ========================================================
    # Initialization
    # ========================================================

    def initialize(self):

        print(
            "Initializing Dynamixels..."
        )

        for motor in self.motor_ids:

            # ------------------------------------------------
            # Disable torque
            # ------------------------------------------------

            self.write1(
                motor,
                ADDR_TORQUE_ENABLE,
                TORQUE_DISABLE,
            )

            # ------------------------------------------------
            # Extended position mode
            # ------------------------------------------------

            self.write1(
                motor,
                ADDR_OPERATING_MODE,
                EXTENDED_POSITION_MODE,
            )

            # ------------------------------------------------
            # Enable torque
            # ------------------------------------------------

            self.write1(
                motor,
                ADDR_TORQUE_ENABLE,
                TORQUE_ENABLE,
            )

            # ------------------------------------------------
            # Read initial position
            # ------------------------------------------------

            pos = self.read_position(motor)

            if pos is None:

                raise RuntimeError(
                    f"Could not read initial "
                    f"position of motor {motor}"
                )

            self.initial_positions[motor] = pos

            self.targets[motor] = pos

            print(
                f"Motor {motor}: "
                f"initial position = {pos}"
            )

        # ----------------------------------------------------
        # Initial diagnostic snapshot
        # ----------------------------------------------------

        self.log_all_motor_diagnostics(
            reason="initialization"
        )

        print(
            "Initialization complete."
        )


    # ========================================================
    # Execute RL action
    # ========================================================
    # ========================================================
    # Execute RL action
    # ========================================================

    def execute(
        self,
        action,
    ):

        if len(action) != len(self.motor_ids):

            raise ValueError(
                "Action dimension does not match motors"
            )

        self.action_count += 1

        action = np.asarray(
            action,
            dtype=np.float32,
        )

        # ====================================================
        # ACTION HEADER
        # ====================================================

        print(
            "\n"
            "============================================================"
        )

        print(
            f"RL ACTION #{self.action_count}"
        )

        print(
            f"timestamp: "
            f"{datetime.now().isoformat()}"
        )

        print(
            f"raw action: {action}"
        )

        print(
            "============================================================"
        )

        # ====================================================
        # Save targets BEFORE action
        # ====================================================

        targets_before = {
            motor: int(self.targets[motor])
            for motor in self.motor_ids
        }

        # ====================================================
        # Convert RL action -> encoder deltas
        # ====================================================

        encoder_deltas = {}

        targets_after = {}

        for motor, delta in zip(
            self.motor_ids,
            action,
        ):

            delta = np.clip(
                delta,
                -1.0,
                1.0,
            )

            encoder_delta = int(
                delta * MAX_ENCODER_DELTA
            )

            encoder_deltas[motor] = (
                encoder_delta
            )

            # ------------------------------------------------
            # Update target
            # ------------------------------------------------

            self.targets[motor] += (
                encoder_delta
            )

            # ------------------------------------------------
            # Relative safety limit
            # ------------------------------------------------

            lower_limit = (
                self.initial_positions[motor]
                -
                MAX_ENCODER_TRAVEL
            )

            upper_limit = (
                self.initial_positions[motor]
                +
                MAX_ENCODER_TRAVEL
            )

            self.targets[motor] = int(
                np.clip(
                    self.targets[motor],
                    lower_limit,
                    upper_limit,
                )
            )

            targets_after[motor] = (
                self.targets[motor]
            )

            print(
                f"  Motor {motor}: "
                f"action={float(delta):+.5f} "
                f"delta={encoder_delta:+d} "
                f"target="
                f"{targets_before[motor]}"
                f" -> "
                f"{self.targets[motor]}"
            )

        # ====================================================
        # Build synchronized packet
        # ====================================================

        self.group_sync_write.clearParam()

        for motor in self.motor_ids:

            target = self.targets[motor]

            param = [

                DXL_LOBYTE(
                    DXL_LOWORD(target)
                ),

                DXL_HIBYTE(
                    DXL_LOWORD(target)
                ),

                DXL_LOBYTE(
                    DXL_HIWORD(target)
                ),

                DXL_HIBYTE(
                    DXL_HIWORD(target)
                ),
            ]

            if not self.group_sync_write.addParam(
                motor,
                param,
            ):

                raise RuntimeError(
                    f"Failed adding motor {motor}"
                )

        # ====================================================
        # Send synchronized command
        # ====================================================

        print(
            "  Sending synchronized motor command..."
        )

        result = (
            self.group_sync_write.txPacket()
        )

        self.group_sync_write.clearParam()

        # ====================================================
        # Communication failure
        # ====================================================

        if result != COMM_SUCCESS:

            print(
                "\n"
                "!!! COMMUNICATION FAILURE !!!"
            )

            print(
                self.packet.getTxRxResult(result)
            )

            telemetry = {}

            for motor in self.motor_ids:

                telemetry[motor] = (
                    self._get_motor_telemetry(
                        motor
                    )
                )

            self._log_action(
                action=action,
                encoder_deltas=encoder_deltas,
                targets_before=targets_before,
                targets_after=targets_after,
                telemetry=telemetry,
                success=False,
                hardware_error=False,
                error_message=(
                    self.packet.getTxRxResult(
                        result
                    )
                ),
            )

            self.log_all_motor_diagnostics(
                reason="communication_failure"
            )

            raise RuntimeError(
                "Dynamixel communication failure: "
                f"{self.packet.getTxRxResult(result)}"
            )

        print(
            "  Command transmitted successfully."
        )

        # ====================================================
        # Give motor a short amount of time to respond
        # ====================================================

        # This is intentionally small. The environment/camera
        # may need more time elsewhere.
        time.sleep(0.02)

        # ====================================================
        # Check Hardware Error Status
        # ====================================================

        hardware_error_ids = []

        hardware_error_status = {}

        packet_errors = {}

        for motor in self.motor_ids:

            status, packet_error = (
                self.read_hardware_error_with_packet_error(
                    motor
                )
            )

            hardware_error_status[motor] = (
                status
            )

            packet_errors[motor] = (
                packet_error
            )

            if (
                status is None
                or status != 0
                or packet_error is not None
            ):

                hardware_error_ids.append(
                    motor
                )

        # ====================================================
        # Read COMPLETE telemetry for THIS ACTION
        # ====================================================

        telemetry = {}

        for motor in self.motor_ids:

            telemetry[motor] = (
                self._get_motor_telemetry(
                    motor
                )
            )

        # ====================================================
        # Hardware error occurred
        # ====================================================

        if len(hardware_error_ids) > 0:

            message = (
                "Dynamixel hardware error detected. "
                f"Motors: {hardware_error_ids}. "
                f"Hardware status: "
                f"{hardware_error_status}. "
                f"Packet errors: "
                f"{packet_errors}"
            )

            print(
                "\n"
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            )

            print(
                f"HARDWARE ERROR DURING ACTION "
                f"#{self.action_count}"
            )

            print(message)

            print(
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            )

            # ------------------------------------------------
            # Action-level log
            # ------------------------------------------------

            self._log_action(
                action=action,
                encoder_deltas=encoder_deltas,
                targets_before=targets_before,
                targets_after=targets_after,
                telemetry=telemetry,
                packet_errors=packet_errors,
                success=False,
                hardware_error=True,
                hardware_error_ids=(
                    hardware_error_ids
                ),
                error_message=message,
            )

            # ------------------------------------------------
            # Detailed diagnostic snapshot
            # ------------------------------------------------

            for motor in self.motor_ids:

                self.log_motor_diagnostics(
                    motor,
                    reason=(
                        f"HARDWARE_ERROR_ACTION_"
                        f"{self.action_count}"
                    ),
                    packet_error=(
                        packet_errors.get(motor)
                    ),
                    hardware_status=(
                        hardware_error_status.get(
                            motor
                        )
                    ),
                )

            return ExecutionResult(
                success=False,
                hardware_error=True,
                hardware_error_ids=(
                    hardware_error_ids
                ),
                hardware_error_status=(
                    hardware_error_status
                ),
                error_message=message,
            )

        # ====================================================
        # Successful action
        # ====================================================

        self.logger.update_context(
            episode=self.current_episode,
            step=self.current_step,
            action=action,
        )

        self._log_action(
            action=action,
            encoder_deltas=encoder_deltas,
            targets_before=targets_before,
            targets_after=targets_after,
            telemetry=telemetry,
            packet_errors=packet_errors,
            success=True,
            hardware_error=False,
            hardware_error_ids=[],
            error_message="",
        )

        # ====================================================
        # Print action telemetry
        # ====================================================

        print(
            "\n"
            f"ACTION #{self.action_count} TELEMETRY"
        )

        for motor in self.motor_ids:

            t = telemetry[motor]

            print(
                f"  Motor {motor}: "
                f"pos={t['position']} "
                f"target={self.targets[motor]} "
                f"current={t['current']}mA "
                f"voltage={t['voltage']}V "
                f"temp={t['temperature']}C "
                f"PWM={t['pwm']} "
                f"velocity={t['velocity']} "
                f"HW={t['hardware_status']}"
            )

        print(
            f"  ACTION #{self.action_count} SUCCESS"
        )

        print(
            "============================================================"
        )

        # ====================================================
        # Periodic detailed diagnostic snapshot
        # ====================================================

        if (
            self.action_count
            % DIAGNOSTIC_LOG_EVERY_N_ACTIONS
            == 0
        ):

            self.log_all_motor_diagnostics(
                reason=(
                    f"ACTION_{self.action_count}"
                )
            )

        # ====================================================
        # Return
        # ====================================================

        return ExecutionResult(
            success=True,
            hardware_error=False,
            hardware_error_ids=[],
            hardware_error_status=(
                hardware_error_status
            ),
            error_message="",
        )

    # ========================================================
    # Read motor position
    # ========================================================

    def read_position(
        self,
        motor_id,
    ):

        pos, dxl_comm_result, dxl_error = (
            self.packet.read4ByteTxRx(
                self.port,
                motor_id,
                ADDR_PRESENT_POSITION,
            )
        )

        if dxl_comm_result != COMM_SUCCESS:

            message = (
                f"Communication error reading "
                f"motor {motor_id}: "
                f"{self.packet.getTxRxResult(dxl_comm_result)}"
            )

            print(message)

            self.hardware_error = True

            if motor_id not in self.hardware_error_ids:
                self.hardware_error_ids.append(motor_id)

            self.hardware_error_status[motor_id] = None

            self.hardware_error_message = message

            return None

        if dxl_error != 0:

            # IMPORTANT:
            #
            # dxl_error is a Protocol 2.0 packet error.
            # It is NOT Hardware Error Status(70).

            print(
                f"Motor {motor_id}: "
                f"packet error while reading position: "
                f"0x{dxl_error:02X} "
                f"({self.packet.getRxPacketError(dxl_error)})"
            )

            self._record_hardware_error(
                motor_id,
                dxl_error,
                message=(
                    f"Motor {motor_id}: "
                    f"packet error while reading position: "
                    f"0x{dxl_error:02X}"
                ),
            )

            return None

        # Convert unsigned -> signed
        if pos > 0x7FFFFFFF:
            pos -= 0x100000000

        return pos


    def read_positions(self):

        positions = []

        for motor_id in self.motor_ids:

            position = self.read_position(
                motor_id
            )

            if position is None:
                return None

            positions.append(position)

        return positions


    # ========================================================
    # Read motor current
    # ========================================================

    def read_current(
        self,
        motor_id,
    ):

        current, dxl_comm_result, dxl_error = (
            self.packet.read2ByteTxRx(
                self.port,
                motor_id,
                ADDR_PRESENT_CURRENT,
            )
        )

        if dxl_comm_result != COMM_SUCCESS:

            print(
                f"Communication error reading "
                f"current from motor {motor_id}: "
                f"{self.packet.getTxRxResult(dxl_comm_result)}"
            )

            return None

        if dxl_error != 0:

            self._record_hardware_error(
                motor_id,
                dxl_error,
            )

            return None

        # Signed 16-bit
        if current >= 0x8000:
            current -= 0x10000

        return current


    def read_currents(self):

        currents = []

        for motor_id in self.motor_ids:

            current = self.read_current(
                motor_id
            )

            if current is None:
                return None

            currents.append(current)

        return currents


    # ========================================================
    # Read hardware error + packet error separately
    # ========================================================

    def read_hardware_error_with_packet_error(
        self,
        motor_id,
    ):

        """
        Return:

            hardware_status,
            packet_error

        These are deliberately kept separate.

        hardware_status:
            Register 70.

        packet_error:
            Protocol 2.0 Status Packet error byte.
        """

        status, dxl_comm_result, dxl_error = (
            self.packet.read1ByteTxRx(
                self.port,
                motor_id,
                ADDR_HARDWARE_ERROR_STATUS,
            )
        )

        # ----------------------------------------------------
        # Communication failure
        # ----------------------------------------------------

        if dxl_comm_result != COMM_SUCCESS:

            print(
                f"Motor {motor_id}: "
                f"communication failure while reading "
                f"Hardware Error Status: "
                f"{self.packet.getTxRxResult(dxl_comm_result)}"
            )

            self.hardware_error = True

            if motor_id not in self.hardware_error_ids:
                self.hardware_error_ids.append(motor_id)

            self.hardware_error_status[motor_id] = None

            self.hardware_error_message = (
                f"Communication failure reading "
                f"Hardware Error Status for motor "
                f"{motor_id}"
            )

            return None, None

        # ----------------------------------------------------
        # Protocol packet error
        # ----------------------------------------------------

        if dxl_error != 0:

            print(
                "\n"
                f"Motor {motor_id}: "
                f"Protocol packet error while reading "
                f"Hardware Error Status"
            )

            print(
                f"  Packet error: "
                f"0x{dxl_error:02X}"
            )

            print(
                f"  Description: "
                f"{self.packet.getRxPacketError(dxl_error)}"
            )

            # IMPORTANT:
            #
            # The returned `status` is still the data byte
            # from address 70 if the SDK provides it.
            #
            # Keep it separate from dxl_error.

            if status is not None and status != 0:

                self._record_hardware_error(
                    motor_id,
                    status,
                )

            else:

                self.hardware_error = True

                if motor_id not in self.hardware_error_ids:
                    self.hardware_error_ids.append(
                        motor_id
                    )

                self.hardware_error_status[
                    motor_id
                ] = status

                self.hardware_error_message = (
                    f"Motor {motor_id}: "
                    f"Protocol packet error "
                    f"0x{dxl_error:02X}"
                )

            return status, dxl_error

        # ----------------------------------------------------
        # Actual Hardware Error Status register
        # ----------------------------------------------------

        if status != 0:

            self._record_hardware_error(
                motor_id,
                status,
            )

        return status, None


    def read_hardware_error(
        self,
        motor_id,
    ):

        status, _ = (
            self.read_hardware_error_with_packet_error(
                motor_id
            )
        )

        return status


    # ========================================================
    # Record hardware error
    # ========================================================

    def _record_hardware_error(
        self,
        motor_id,
        error_code,
        message=None,
    ):

        self.hardware_error = True

        if motor_id not in self.hardware_error_ids:

            self.hardware_error_ids.append(
                motor_id
            )

        self.hardware_error_status[
            motor_id
        ] = error_code

        decoded_errors = (
            self.decode_hardware_error(
                error_code
            )
        )

        if message is None:

            message = (
                f"Motor {motor_id}: "
                f"hardware error status "
                f"{error_code} "
                f"(0x{error_code:02X}) - "
                f"{', '.join(decoded_errors)}"
            )

        self.hardware_error_message = message

        print(
            "\n!!! DYNAMIXEL HARDWARE ERROR !!!"
        )

        print(
            f"Motor ID: {motor_id}"
        )

        print(
            f"Hardware Error Status: "
            f"{error_code} "
            f"(0x{error_code:02X})"
        )

        print(
            f"Decoded: "
            f"{', '.join(decoded_errors)}"
        )


    # ========================================================
    # Error state
    # ========================================================

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


    # ========================================================
    # Helpers
    # ========================================================

    @staticmethod
    def format_hex(value):

        if value is None:
            return "None"

        return f"0x{int(value):02X}"


    # ========================================================
    # Write one byte
    # ========================================================

    def write1(
        self,
        motor,
        address,
        value,
    ):

        comm, error = (
            self.packet.write1ByteTxRx(
                self.port,
                motor,
                address,
                value,
            )
        )

        if comm != COMM_SUCCESS:

            raise RuntimeError(
                f"Motor {motor}: "
                f"{self.packet.getTxRxResult(comm)}"
            )

        if error != 0:

            raise RuntimeError(
                f"Motor {motor}: "
                f"{self.packet.getRxPacketError(error)}"
            )


    # ========================================================
    # Recovery
    # ========================================================

    def recovery(self):

        """
        Recover Dynamixels and return them to their
        initial encoder positions.

        Motors reporting a hardware shutdown are rebooted
        before torque/mode configuration is attempted.
        """

        print(
            "\n================ RECOVERY ================"
        )

        # ====================================================
        # 1. Capture diagnostic state BEFORE recovery
        # ====================================================

        print(
            "Capturing pre-recovery motor diagnostics..."
        )

        self.log_all_motor_diagnostics(
            reason="PRE_RECOVERY"
        )

        # ====================================================
        # 2. Detect motors requiring reboot
        # ====================================================

        motors_to_reboot = []

        hardware_statuses = {}

        for motor in self.motor_ids:

            status, packet_error = (
                self.read_hardware_error_with_packet_error(
                    motor
                )
            )

            hardware_statuses[motor] = status

            print(
                f"Motor {motor}: "
                f"hardware status={status}, "
                f"packet error={packet_error}"
            )

            if (
                status is None
                or status != 0
                or packet_error is not None
            ):

                motors_to_reboot.append(
                    motor
                )

        # ====================================================
        # 3. Reboot affected motors
        # ====================================================

        if motors_to_reboot:

            print(
                "\nMotors requiring reboot: "
                f"{motors_to_reboot}"
            )

            for motor in motors_to_reboot:

                print(
                    f"\nRebooting motor {motor}..."
                )

                # Diagnostic snapshot immediately before reboot
                self.log_motor_diagnostics(
                    motor,
                    reason="BEFORE_REBOOT",
                    hardware_status=(
                        hardware_statuses.get(motor)
                    ),
                )

                success = self.reboot_motor(
                    motor
                )

                if not success:

                    # Try to capture diagnostics one last time.
                    self.log_motor_diagnostics(
                        motor,
                        reason="REBOOT_FAILED",
                    )

                    raise RuntimeError(
                        f"Motor {motor} reboot failed."
                    )

            time.sleep(0.5)

        else:

            print(
                "No motors require reboot."
            )

        # ====================================================
        # 4. Verify communication
        # ====================================================

        print(
            "\nVerifying motor communication..."
        )

        for motor in self.motor_ids:

            pos = self.read_position(
                motor
            )

            if pos is None:

                self.log_motor_diagnostics(
                    motor,
                    reason="POST_REBOOT_COMM_FAILURE",
                )

                raise RuntimeError(
                    f"Motor {motor} did not recover "
                    "after reboot."
                )

            print(
                f"Motor {motor}: "
                f"communication restored, "
                f"position={pos}"
            )

        # ====================================================
        # 5. Disable torque
        # ====================================================

        print(
            "\nDisabling torque..."
        )

        for motor in self.motor_ids:

            self.write1(
                motor,
                ADDR_TORQUE_ENABLE,
                TORQUE_DISABLE,
            )

        # ====================================================
        # 6. Restore extended position mode
        # ====================================================

        print(
            "Restoring extended position mode..."
        )

        for motor in self.motor_ids:

            self.write1(
                motor,
                ADDR_OPERATING_MODE,
                EXTENDED_POSITION_MODE,
            )

        # ====================================================
        # 7. Re-enable torque
        # ====================================================

        print(
            "Re-enabling torque..."
        )

        for motor in self.motor_ids:

            self.write1(
                motor,
                ADDR_TORQUE_ENABLE,
                TORQUE_ENABLE,
            )

        # ====================================================
        # 8. Return to initial positions
        # ====================================================

        print(
            "Commanding return to center..."
        )

        self.group_sync_write.clearParam()

        for motor in self.motor_ids:

            target = self.initial_positions[
                motor
            ]

            self.targets[motor] = target

            param = [

                DXL_LOBYTE(
                    DXL_LOWORD(target)
                ),

                DXL_HIBYTE(
                    DXL_LOWORD(target)
                ),

                DXL_LOBYTE(
                    DXL_HIWORD(target)
                ),

                DXL_HIBYTE(
                    DXL_HIWORD(target)
                ),
            ]

            if not self.group_sync_write.addParam(
                motor,
                param,
            ):

                raise RuntimeError(
                    f"Failed adding motor {motor} "
                    f"during recovery"
                )

        # ====================================================
        # 9. Send synchronized command
        # ====================================================

        result = (
            self.group_sync_write.txPacket()
        )

        self.group_sync_write.clearParam()

        if result != COMM_SUCCESS:

            self.log_all_motor_diagnostics(
                reason="RECOVERY_COMMAND_FAILURE"
            )

            raise RuntimeError(
                "Recovery failed: "
                f"{self.packet.getTxRxResult(result)}"
            )

        print(
            "Recovery command sent."
        )

        # ====================================================
        # 10. Wait for physical movement
        # ====================================================

        time.sleep(1.0)

        # ====================================================
        # 11. Post-recovery diagnostics
        # ====================================================

        print(
            "\nPost-recovery motor diagnostics..."
        )

        self.log_all_motor_diagnostics(
            reason="POST_RECOVERY"
        )

        # ====================================================
        # 12. Verify positions
        # ====================================================

        for motor in self.motor_ids:

            pos = self.read_position(
                motor
            )

            if pos is None:

                raise RuntimeError(
                    f"Motor {motor} failed "
                    "post-recovery verification."
                )

            print(
                f"Motor {motor}: "
                f"post-recovery position={pos}"
            )

        # ====================================================
        # 13. Clear software error state
        # ====================================================

        self.clear_hardware_errors()

        print(
            "Recovery complete."
        )

        print(
            "==========================================\n"
        )


    # ========================================================
    # Reboot
    # ========================================================

    def reboot_motor(
        self,
        motor_id,
    ):

        print(
            f"\n========== REBOOT MOTOR "
            f"{motor_id} =========="
        )

        comm, error = (
            self.packet.reboot(
                self.port,
                motor_id,
            )
        )

        print(
            f"Reboot communication result: "
            f"{comm}"
        )

        print(
            f"Reboot packet error: "
            f"{error}"
        )

        # ----------------------------------------------------
        # Communication error
        # ----------------------------------------------------

        if comm != COMM_SUCCESS:

            print(
                f"Reboot communication error: "
                f"{self.packet.getTxRxResult(comm)}"
            )

            return False

        # ----------------------------------------------------
        # Packet-level error
        # ----------------------------------------------------

        if error != 0:

            print(
                f"Motor returned packet error: "
                f"0x{error:02X}"
            )

            print(
                f"Description: "
                f"{self.packet.getRxPacketError(error)}"
            )

            return False

        # ----------------------------------------------------
        # Wait for reboot
        # ----------------------------------------------------

        time.sleep(0.5)

        print(
            f"Motor {motor_id} reboot completed."
        )

        return True


    # ========================================================
    # Decode Hardware Error Status
    # ========================================================

    @staticmethod
    def decode_hardware_error(
        status
    ):

        """
        Decode XL330 Hardware Error Status (address 70).

        IMPORTANT:
        0x80 is NOT a Hardware Error Status bit on the XL330.

        Hardware Error Status bits:

            0x01 -> Input Voltage Error
            0x04 -> Overheating
            0x10 -> Electrical Shock
            0x20 -> Overload

        The Protocol 2.0 Alert bit is a separate packet-level
        error field.
        """

        if status is None:

            return [
                "No hardware status available"
            ]

        status = int(status)

        errors = []

        # ----------------------------------------------------
        # Input voltage
        # ----------------------------------------------------

        if status & 0x01:

            errors.append(
                "Input voltage error"
            )

        # ----------------------------------------------------
        # Overheating
        # ----------------------------------------------------

        if status & 0x04:

            errors.append(
                "Overheating"
            )

        # ----------------------------------------------------
        # Electrical shock
        # ----------------------------------------------------

        if status & 0x10:

            errors.append(
                "Electrical shock"
            )

        # ----------------------------------------------------
        # Overload
        # ----------------------------------------------------

        if status & 0x20:

            errors.append(
                "Overload"
            )

        # ----------------------------------------------------
        # Unknown bits
        # ----------------------------------------------------

        known_mask = (
            0x01
            |
            0x04
            |
            0x10
            |
            0x20
        )

        unknown_bits = (
            status & ~known_mask
        )

        if unknown_bits:

            errors.append(
                f"Unknown hardware bits "
                f"0x{unknown_bits:02X}"
            )

        if not errors:

            errors.append(
                "No hardware error"
            )

        return errors


    # ========================================================
    # Shutdown
    # ========================================================

    def shutdown(self):

        print(
            "Disabling motors..."
        )

        for motor in self.motor_ids:

            try:

                self.write1(
                    motor,
                    ADDR_TORQUE_ENABLE,
                    TORQUE_DISABLE,
                )

            except Exception as e:

                print(
                    f"Motor {motor}: "
                    f"failed to disable torque: "
                    f"{e}"
                )

        print(
            "Shutdown complete."
        )

