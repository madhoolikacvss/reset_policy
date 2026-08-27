# reset_policy/config.py
"""
Central configuration for the reset policy system.
All constants and thresholds defined in one place.
"""

from dataclasses import dataclass
from typing import List, Dict, Any
from pathlib import Path

@dataclass
class MotorConfig:
    """Motor IDs and their physical arrangement."""
    motor_ids: List[int] = None
    
    # Motor pairs based on coordinate system:
    # Y-axis = horizontal, X-axis = vertical
    horizontal_pair: List[int] = None  # Motors controlling Y-axis
    vertical_pair: List[int] = None    # Motors controlling X-axis
    
    def __post_init__(self):
        if self.motor_ids is None:
            self.motor_ids = [16, 17, 18, 19]
        if self.horizontal_pair is None:
            self.horizontal_pair = [16, 17]  # Y-axis (left/right)
        if self.vertical_pair is None:
            self.vertical_pair = [18, 19]    # X-axis (up/down)

@dataclass
class DynamixelConfig:
    """Dynamixel communication and control settings."""
    # Communication
    port_name: str = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT89FK0C-if00-port0"
    baudrate: int = 1000000
    protocol_version: float = 2.0
    
    # Control
    max_encoder_delta: int = 150       # Max ticks per action
    max_encoder_travel: int = 10000    # Max total travel from initial
    action_duration: float = 0.8       # Seconds to wait after action
    
    # Addresses
    addr_operating_mode: int = 11
    addr_torque_enable: int = 64
    addr_hardware_error_status: int = 70
    addr_present_pwm: int = 124
    addr_present_current: int = 126
    addr_present_velocity: int = 128
    addr_present_position: int = 132
    addr_present_input_voltage: int = 144
    addr_present_temperature: int = 146
    addr_goal_position: int = 116
    
    # Modes
    extended_position_mode: int = 4
    torque_enable: int = 1
    torque_disable: int = 0

@dataclass
class SafetyConfig:
    """Safety filter thresholds and settings."""
    # Current thresholds (mA)
    max_pair_current: float = 800.0
    max_single_current: float = 600.0
    
    # Current-aware scaling
    heavy_load_threshold: float = 400.0
    critical_load_threshold: float = 700.0
    heavy_load_scale: float = 0.3
    moderate_load_scale: float = 0.6
    
    # Position limits (encoder ticks relative to initial)
    max_position: float = 8000.0
    position_scale_factor: float = 0.5
    
    # Voltage thresholds (V)
    min_voltage: float = 4.0
    critical_voltage: float = 3.5
    voltage_scale_factor: float = 0.3
    
    # Temperature thresholds (°C)
    temp_threshold: float = 45.0
    temp_critical: float = 55.0
    temp_scale_factor: float = 0.3
    
    # Tension constraint (mA)
    min_tension_threshold: float = 30.0
    
    # Single motor tension (mA)
    single_motor_tension_threshold: float = 300.0
    single_motor_tension_high: float = 500.0
    single_motor_tension_critical: float = 700.0
    
    # Reactive current scaling
    current_scale_factor: float = 0.25
    
    # Feature flags
    enable_current_safety: bool = True
    enable_position_safety: bool = True
    enable_tension_safety: bool = True
    enable_voltage_safety: bool = True
    enable_current_aware_safety: bool = True
    enable_tension_constraint: bool = True
    enable_temperature_safety: bool = True
    enable_single_motor_tension: bool = False  # Disabled initially - too aggressive
    # enable_stuck_motor_safety: bool = False     # Not implemented yet
    
    # Logging
    log_interventions: bool = True

@dataclass
class EnvironmentConfig:
    """Environment and training settings."""
    # Episode settings
    max_steps: int = 200
    target_coverage: float = 0.95
    
    # Current thresholds for termination
    safe_current_threshold: float = 500.0
    moderate_current_threshold: float = 1800.0
    high_current_threshold: float = 2500.0
    
    # Safety penalty
    safety_penalty_weight: float = 1.0
    
    # Board bounds (meters)
    x_min: float = -0.76
    x_max: float = -0.47
    y_min: float = -0.94
    y_max: float = 0.10
    
    # Grid
    cell_size_m: float = 0.01

@dataclass
class TrainingConfig:
    """PPO training settings."""
    episodes: int = 1000
    save_every: int = 5
    render_every: int = 5
    min_steps_before_update: int = 50
    max_buffer_size: int = 500
    device: str = "cuda" if __import__('torch').cuda.is_available() else "cpu"

class Config:
    """Central configuration manager."""
    
    def __init__(self):
        self.motor = MotorConfig()
        self.dynamixel = DynamixelConfig()
        self.safety = SafetyConfig()
        self.environment = EnvironmentConfig()
        self.training = TrainingConfig()
        
        # Derived paths
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.config_dir = self.project_root / "configs"
        self.checkpoint_dir = self.project_root / "checkpoints"
        self.training_log_dir = self.project_root / "training_logs"
        self.occupancy_dir = self.training_log_dir / "occupancy"
        
        # Create directories
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.training_log_dir.mkdir(exist_ok=True)
        self.occupancy_dir.mkdir(exist_ok=True)
    
    def get_motor_indices(self, motor_ids: List[int]) -> Dict[str, List[int]]:
        """Get indices for horizontal and vertical pairs."""
        return {
            'horizontal': [motor_ids.index(m) for m in self.motor.horizontal_pair],
            'vertical': [motor_ids.index(m) for m in self.motor.vertical_pair],
        }

# Global config instance
config = Config()