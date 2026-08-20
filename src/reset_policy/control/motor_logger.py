# reset_policy/control/motor_logger.py

"""
High-frequency motor telemetry logger for diagnostic purposes.

This runs in a separate thread to log motor data at the control frequency
(100-200 Hz) without blocking the main training loop.
"""

from __future__ import annotations

import csv
import threading
import time
import numpy as np
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque

from dynamixel_sdk import *


# ============================================================
# Register Addresses
# ============================================================

ADDR_PRESENT_PWM = 124
ADDR_PRESENT_CURRENT = 126
ADDR_PRESENT_VELOCITY = 128
ADDR_PRESENT_POSITION = 132
ADDR_VELOCITY_TRAJECTORY = 136
ADDR_POSITION_TRAJECTORY = 140
ADDR_PRESENT_INPUT_VOLTAGE = 144
ADDR_PRESENT_TEMPERATURE = 146
ADDR_HARDWARE_ERROR_STATUS = 70
ADDR_TORQUE_ENABLE = 64
ADDR_MOVING = 122
ADDR_GOAL_POSITION = 116


@dataclass
class MotorSnapshot:
    """Single time snapshot of all motor data."""
    
    timestamp: float  # time.time()
    
    # Per motor data
    positions: List[Optional[int]] = field(default_factory=list)
    currents: List[Optional[float]] = field(default_factory=list)  # mA
    velocities: List[Optional[float]] = field(default_factory=list)
    pwm: List[Optional[float]] = field(default_factory=list)
    voltage: List[Optional[float]] = field(default_factory=list)  # V
    temperature: List[Optional[float]] = field(default_factory=list)  # °C
    hardware_error: List[Optional[int]] = field(default_factory=list)
    torque_enabled: List[Optional[int]] = field(default_factory=list)
    moving: List[Optional[int]] = field(default_factory=list)
    goal_positions: List[Optional[int]] = field(default_factory=list)
    trajectory_positions: List[Optional[int]] = field(default_factory=list)
    
    # Episode info
    episode: int = -1
    step: int = -1
    action: Optional[np.ndarray] = None
    cube_x: Optional[float] = None
    cube_y: Optional[float] = None
    action_in_episode: int = -1  # Action number within current episode


class MotorLogger:
    """
    High-frequency motor data logger running in a separate thread.
    """
    
    def __init__(
        self,
        port_handler,
        packet_handler,
        motor_ids: List[int],
        log_dir: Path = None,
        log_frequency_hz: float = 100.0,
        buffer_size: int = 10000,
        trigger_buffer_size: int = 500,
    ):
        self.port = port_handler
        self.packet = packet_handler
        self.motor_ids = motor_ids
        self.num_motors = len(motor_ids)
        
        # Log directory
        if log_dir is None:
            log_dir = Path(__file__).resolve().parent.parent.parent / "logs" / "motor_data"
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Logging parameters
        self.log_frequency_hz = log_frequency_hz
        self.log_interval = 1.0 / log_frequency_hz
        self.buffer_size = buffer_size
        self.trigger_buffer_size = trigger_buffer_size
        
        # Data buffers
        self.buffer: deque = deque(maxlen=buffer_size)
        self.trigger_buffer: deque = deque(maxlen=trigger_buffer_size)
        
        # Threading
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        
        # Log file management
        self.csv_file = None
        self.csv_writer = None
        self.csv_headers_written = False
        
        # Current episode/step tracking
        self._episode = 0
        self._step = 0
        self._action = None
        self._cube_x = None
        self._cube_y = None
        
        # Track action counts per episode for logging
        self.action_count_in_episode = 0
        
        # Statistics
        self.sample_count = 0
        self.error_count = 0
        self.last_log_time = 0.0
        
        # Hardware error detection
        self.error_triggered = False
        self.error_trigger_time = 0.0
        self.error_log_file = None
        
        print(f"MotorLogger initialized: {log_frequency_hz} Hz, buffer_size={buffer_size}")
        
    def start(self):
        """Start the logging thread."""
        if self.thread is not None and self.thread.is_alive():
            print("Logger already running")
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._log_loop, daemon=True)
        self.thread.start()
        print(f"MotorLogger thread started (PID: {threading.get_ident()})")
        
    def stop(self, timeout: float = 2.0):
        """Stop the logging thread."""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=timeout)
            if self.thread.is_alive():
                print("Warning: Logger thread did not stop gracefully")
            else:
                print("MotorLogger thread stopped")
        
        # Close CSV file
        self._close_csv()
        
    def _log_loop(self):
        """Main logging loop."""
        while self.running:
            loop_start = time.perf_counter()
            
            try:
                # Read snapshot with current context
                snapshot = self._read_snapshot()
                
                # Store in buffers
                with self.lock:
                    self.buffer.append(snapshot)
                    self.trigger_buffer.append(snapshot)
                    self.sample_count += 1
                
                # Write to CSV if enabled
                if self.csv_writer is not None:
                    self._write_snapshot_csv(snapshot)
                
                # Check for errors
                self._check_errors(snapshot)
                
            except Exception as e:
                self.error_count += 1
                if self.error_count % 10 == 1:
                    print(f"MotorLogger error: {e}")
            
            # Maintain frequency
            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0, self.log_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
                    
    def _read_snapshot(self) -> MotorSnapshot:
        """Read snapshot with current context."""
        snapshot = MotorSnapshot(
            timestamp=time.time(),
            episode=self._episode,
            step=self._step,
            action=self._action.copy() if self._action is not None else None,
            cube_x=self._cube_x,
            cube_y=self._cube_y,
            action_in_episode=self.action_count_in_episode,
        )
        
        for motor_id in self.motor_ids:
            # Read position (signed 32-bit)
            pos = self._read_4byte_signed(motor_id, ADDR_PRESENT_POSITION)
            snapshot.positions.append(pos)
            
            # Read current (signed 16-bit, mA)
            curr = self._read_2byte_signed(motor_id, ADDR_PRESENT_CURRENT)
            snapshot.currents.append(curr)
            
            # Read velocity (signed 32-bit)
            vel = self._read_4byte_signed(motor_id, ADDR_PRESENT_VELOCITY)
            snapshot.velocities.append(vel)
            
            # Read PWM (signed 16-bit)
            pwm = self._read_2byte_signed(motor_id, ADDR_PRESENT_PWM)
            snapshot.pwm.append(pwm)
            
            # Read voltage (unsigned 16-bit, *0.1 V)
            volt_raw = self._read_2byte_unsigned(motor_id, ADDR_PRESENT_INPUT_VOLTAGE)
            if volt_raw is not None:
                volt = volt_raw * 0.1
            else:
                volt = None
            snapshot.voltage.append(volt)
            
            # Read temperature (unsigned 8-bit, °C)
            temp = self._read_1byte_unsigned(motor_id, ADDR_PRESENT_TEMPERATURE)
            snapshot.temperature.append(temp)
            
            # Read hardware error status (unsigned 8-bit)
            hw_err = self._read_1byte_unsigned(motor_id, ADDR_HARDWARE_ERROR_STATUS)
            snapshot.hardware_error.append(hw_err)
            
            # Read torque enable (unsigned 8-bit)
            torque = self._read_1byte_unsigned(motor_id, ADDR_TORQUE_ENABLE)
            snapshot.torque_enabled.append(torque)
            
            # Read moving status (unsigned 8-bit)
            moving = self._read_1byte_unsigned(motor_id, ADDR_MOVING)
            snapshot.moving.append(moving)
            
            # Read goal position (signed 32-bit)
            goal = self._read_4byte_signed(motor_id, ADDR_GOAL_POSITION)
            snapshot.goal_positions.append(goal)
            
            # Read trajectory position (signed 32-bit)
            traj = self._read_4byte_signed(motor_id, ADDR_POSITION_TRAJECTORY)
            snapshot.trajectory_positions.append(traj)
            
        return snapshot
        
    # ============================================================
    # Helper read functions with error handling
    # ============================================================
    
    def _read_1byte_unsigned(self, motor_id: int, address: int) -> Optional[int]:
        """Read unsigned 8-bit value."""
        try:
            value, comm, error = self.packet.read1ByteTxRx(self.port, motor_id, address)
            if comm != COMM_SUCCESS or error != 0:
                return None
            return value
        except Exception:
            return None
            
    def _read_2byte_unsigned(self, motor_id: int, address: int) -> Optional[int]:
        """Read unsigned 16-bit value."""
        try:
            value, comm, error = self.packet.read2ByteTxRx(self.port, motor_id, address)
            if comm != COMM_SUCCESS or error != 0:
                return None
            return value
        except Exception:
            return None
            
    def _read_2byte_signed(self, motor_id: int, address: int) -> Optional[float]:
        """Read signed 16-bit value."""
        try:
            value, comm, error = self.packet.read2ByteTxRx(self.port, motor_id, address)
            if comm != COMM_SUCCESS or error != 0:
                return None
            if value >= 0x8000:
                value -= 0x10000
            return float(value)
        except Exception:
            return None
            
    def _read_4byte_signed(self, motor_id: int, address: int) -> Optional[int]:
        """Read signed 32-bit value."""
        try:
            value, comm, error = self.packet.read4ByteTxRx(self.port, motor_id, address)
            if comm != COMM_SUCCESS or error != 0:
                return None
            if value >= 0x80000000:
                value -= 0x100000000
            return value
        except Exception:
            return None
            
    # ============================================================
    # CSV Logging
    # ============================================================
    
    def start_csv_logging(self, episode: int = -1):
        """Start writing to a CSV file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"motor_log_ep{episode}_{timestamp}.csv"
        self.csv_file = self.log_dir / filename
        
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self._get_csv_headers())
        
        self.csv_writer = open(self.csv_file, 'a', newline='')
        self.csv_headers_written = True
        print(f"Motor CSV logging started: {self.csv_file}")
        
    def _get_csv_headers(self) -> List[str]:
        """Get CSV header columns."""
        headers = [
            "timestamp",
            "episode",
            "step",
            "action_in_episode",
            "cube_x",
            "cube_y",
            "action_m16", "action_m17", "action_m18", "action_m19",
        ]
        
        for motor in self.motor_ids:
            prefix = f"m{motor}"
            headers.extend([
                f"{prefix}_pos",
                f"{prefix}_current_mA",
                f"{prefix}_velocity",
                f"{prefix}_pwm",
                f"{prefix}_voltage_V",
                f"{prefix}_temp_C",
                f"{prefix}_hw_error",
                f"{prefix}_torque",
                f"{prefix}_moving",
                f"{prefix}_goal_pos",
                f"{prefix}_traj_pos",
            ])
        
        # Error decoded
        headers.extend([f"hw_error_decoded_m{motor}" for motor in self.motor_ids])
        
        return headers
        
    def _write_snapshot_csv(self, snapshot: MotorSnapshot):
        """Write a snapshot to CSV."""
        try:
            row = [
                snapshot.timestamp,
                snapshot.episode,
                snapshot.step,
                snapshot.action_in_episode,
                snapshot.cube_x if snapshot.cube_x is not None else "",
                snapshot.cube_y if snapshot.cube_y is not None else "",
            ]
            
            # Actions
            if snapshot.action is not None:
                row.extend(snapshot.action.tolist())
            else:
                row.extend([""] * self.num_motors)
            
            # Motor data
            for i in range(self.num_motors):
                row.extend([
                    snapshot.positions[i] if snapshot.positions[i] is not None else "",
                    snapshot.currents[i] if snapshot.currents[i] is not None else "",
                    snapshot.velocities[i] if snapshot.velocities[i] is not None else "",
                    snapshot.pwm[i] if snapshot.pwm[i] is not None else "",
                    snapshot.voltage[i] if snapshot.voltage[i] is not None else "",
                    snapshot.temperature[i] if snapshot.temperature[i] is not None else "",
                    snapshot.hardware_error[i] if snapshot.hardware_error[i] is not None else "",
                    snapshot.torque_enabled[i] if snapshot.torque_enabled[i] is not None else "",
                    snapshot.moving[i] if snapshot.moving[i] is not None else "",
                    snapshot.goal_positions[i] if snapshot.goal_positions[i] is not None else "",
                    snapshot.trajectory_positions[i] if snapshot.trajectory_positions[i] is not None else "",
                ])
            
            # Decoded hardware errors
            for i, hw_err in enumerate(snapshot.hardware_error):
                if hw_err is not None and hw_err != 0:
                    decoded = self._decode_hardware_error(hw_err)
                    row.append("|".join(decoded))
                else:
                    row.append("")
            
            self.csv_writer.writerow(row)
            self.csv_writer.flush()
            
        except Exception as e:
            print(f"CSV write error: {e}")
            
    def _close_csv(self):
        """Close the CSV file."""
        if self.csv_writer is not None:
            self.csv_writer.close()
            self.csv_writer = None
            self.csv_headers_written = False
            
    def _check_errors(self, snapshot: MotorSnapshot):
        """Check for hardware errors and trigger error logging."""
        for i, motor_id in enumerate(self.motor_ids):
            hw_err = snapshot.hardware_error[i]
            if hw_err is not None and hw_err != 0:
                # Hardware error detected!
                if not self.error_triggered:
                    self.error_triggered = True
                    self.error_trigger_time = snapshot.timestamp
                    
                    # Save error log
                    self._save_error_log(snapshot, motor_id, hw_err)
                    
                    print(f"\n!!! HARDWARE ERROR DETECTED !!!")
                    print(f"Motor {motor_id}: 0x{hw_err:02X}")
                    print(f"Decoded: {self._decode_hardware_error(hw_err)}")
                    print(f"At episode {snapshot.episode}, step {snapshot.step}")
                    
                # Keep triggering for 1 second after error
                elif snapshot.timestamp - self.error_trigger_time > 1.0:
                    self.error_triggered = False
                    
    def _save_error_log(self, snapshot: MotorSnapshot, motor_id: int, error_code: int):
        """Save a detailed error log with pre/post trigger data."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        error_file = self.log_dir / f"error_{timestamp}_m{motor_id}_0x{error_code:02X}.csv"
        
        # Get trigger buffer data
        trigger_data = list(self.trigger_buffer)
        
        with open(error_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write error info
            writer.writerow(["=== HARDWARE ERROR REPORT ==="])
            writer.writerow([f"Timestamp: {datetime.now().isoformat()}"])
            writer.writerow([f"Motor: {motor_id}"])
            writer.writerow([f"Error Code: 0x{error_code:02X}"])
            writer.writerow([f"Decoded: {', '.join(self._decode_hardware_error(error_code))}"])
            writer.writerow([f"Episode: {snapshot.episode}"])
            writer.writerow([f"Step: {snapshot.step}"])
            writer.writerow([f"Action in episode: {snapshot.action_in_episode}"])
            writer.writerow([f"Trigger Buffer Size: {len(trigger_data)}"])
            writer.writerow([])
            
            # Write data headers
            headers = self._get_csv_headers()
            writer.writerow(headers)
            
            # Write all trigger data
            for data in trigger_data:
                row = self._snapshot_to_row(data)
                writer.writerow(row)
            
        print(f"Error log saved: {error_file}")
        
    def _snapshot_to_row(self, snapshot: MotorSnapshot) -> List:
        """Convert snapshot to CSV row (helper for error logs)."""
        row = [
            snapshot.timestamp,
            snapshot.episode,
            snapshot.step,
            snapshot.action_in_episode,
            snapshot.cube_x if snapshot.cube_x is not None else "",
            snapshot.cube_y if snapshot.cube_y is not None else "",
        ]
        
        if snapshot.action is not None:
            row.extend(snapshot.action.tolist())
        else:
            row.extend([""] * self.num_motors)
            
        for i in range(self.num_motors):
            row.extend([
                snapshot.positions[i] if snapshot.positions[i] is not None else "",
                snapshot.currents[i] if snapshot.currents[i] is not None else "",
                snapshot.velocities[i] if snapshot.velocities[i] is not None else "",
                snapshot.pwm[i] if snapshot.pwm[i] is not None else "",
                snapshot.voltage[i] if snapshot.voltage[i] is not None else "",
                snapshot.temperature[i] if snapshot.temperature[i] is not None else "",
                snapshot.hardware_error[i] if snapshot.hardware_error[i] is not None else "",
                snapshot.torque_enabled[i] if snapshot.torque_enabled[i] is not None else "",
                snapshot.moving[i] if snapshot.moving[i] is not None else "",
                snapshot.goal_positions[i] if snapshot.goal_positions[i] is not None else "",
                snapshot.trajectory_positions[i] if snapshot.trajectory_positions[i] is not None else "",
            ])
            
        return row
        
    # ============================================================
    # Error Decoding
    # ============================================================
    
    @staticmethod
    def _decode_hardware_error(status: int) -> List[str]:
        """Decode XL330 Hardware Error Status (address 70)."""
        if status is None:
            return ["No status"]
            
        status = int(status)
        errors = []
        
        # XL330 specific error bits
        if status & 0x01:
            errors.append("Input voltage error")
        if status & 0x04:
            errors.append("Overheating")
        if status & 0x10:
            errors.append("Electrical shock")
        if status & 0x20:
            errors.append("Overload")
        
        # Unknown bits
        known_mask = 0x01 | 0x04 | 0x10 | 0x20
        unknown_bits = status & ~known_mask
        if unknown_bits:
            errors.append(f"Unknown bits 0x{unknown_bits:02X}")
            
        if not errors:
            errors.append("No hardware error")
            
        return errors
        
    # ============================================================
    # Public API
    # ============================================================
    
    def update_context(self, episode: int, step: int, action: np.ndarray = None,
                       cube_x: float = None, cube_y: float = None):
        """Update the current context."""
        # Detect episode change
        if episode != self._episode:
            # New episode started
            self._episode = episode
            self.action_count_in_episode = 0
            print(f"MotorLogger: Episode {episode} started")
            
            # Start new CSV file for this episode
            self._start_episode_csv(episode)
            
        self._episode = episode
        self._step = step
        if action is not None:
            self._action = action.copy()
            self.action_count_in_episode += 1
        self._cube_x = cube_x
        self._cube_y = cube_y

    def _start_episode_csv(self, episode: int):
        """Start a new CSV file for an episode."""
        # Close previous CSV
        self._close_csv()
        
        # Start new one
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"motor_log_ep{episode:04d}_{timestamp}.csv"
        self.csv_file = self.log_dir / filename
        
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self._get_csv_headers())
        
        self.csv_writer = open(self.csv_file, 'a', newline='')
        self.csv_headers_written = True
        print(f"Episode {episode}: CSV logging started: {self.csv_file}")
        
    def get_recent_data(self, n: int = 100) -> List[MotorSnapshot]:
        """Get the n most recent samples."""
        with self.lock:
            return list(self.buffer)[-n:]
            
    def get_stats(self) -> Dict:
        """Get logger statistics."""
        with self.lock:
            return {
                "sample_count": self.sample_count,
                "error_count": self.error_count,
                "buffer_size": len(self.buffer),
                "trigger_buffer_size": len(self.trigger_buffer),
                "running": self.running,
                "is_alive": self.thread is not None and self.thread.is_alive(),
            }
            
    def plot_currents(self, n: int = 1000):
        """Plot recent motor currents."""
        try:
            import matplotlib.pyplot as plt
            
            data = self.get_recent_data(n)
            if not data:
                print("No data to plot")
                return
                
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(f"Motor Currents (last {len(data)} samples)")
            
            motor_colors = ['blue', 'red', 'green', 'orange']
            motor_labels = [f"Motor {mid}" for mid in self.motor_ids]
            
            for i, ax in enumerate(axes.flat):
                if i < self.num_motors:
                    currents = [d.currents[i] for d in data if d.currents[i] is not None]
                    times = [d.timestamp for d in data if d.currents[i] is not None]
                    
                    if currents:
                        times = np.array(times) - times[0]
                        ax.plot(times, currents, color=motor_colors[i], label=motor_labels[i])
                        ax.axhline(y=500, color='green', linestyle='--', alpha=0.5, label='Safe (<500mA)')
                        ax.axhline(y=1800, color='orange', linestyle='--', alpha=0.5, label='Moderate (<1800mA)')
                        ax.axhline(y=2500, color='red', linestyle='--', alpha=0.5, label='High (>2500mA)')
                        ax.set_xlabel('Time (s)')
                        ax.set_ylabel('Current (mA)')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("matplotlib not installed")
            
    def plot_hardware_errors(self):
        """Plot hardware error occurrences over time."""
        try:
            import matplotlib.pyplot as plt
            
            data = list(self.buffer)
            if not data:
                print("No data to plot")
                return
                
            fig, axes = plt.subplots(self.num_motors, 1, figsize=(12, 3*self.num_motors))
            if self.num_motors == 1:
                axes = [axes]
                
            fig.suptitle("Hardware Error Status Over Time")
            
            for i, ax in enumerate(axes):
                errors = [d.hardware_error[i] for d in data]
                times = [d.timestamp for d in data]
                
                times = np.array(times) - times[0]
                ax.step(times, errors, where='post', label=f"Motor {self.motor_ids[i]}")
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Error Code (0=OK)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Highlight non-zero errors
                error_indices = [j for j, e in enumerate(errors) if e is not None and e != 0]
                if error_indices:
                    error_times = times[error_indices]
                    error_vals = [errors[j] for j in error_indices]
                    ax.scatter(error_times, error_vals, color='red', s=50, zorder=5)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("matplotlib not installed")