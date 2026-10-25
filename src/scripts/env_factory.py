"""
Environment factory for the cube-string reset policy.

Extracted from main.py so training scripts (PPO, SAC, eval) can share it.
"""

from __future__ import annotations

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from reset_policy.config import config

# Perception
from reset_policy.perception.cameras import CameraPair
from reset_policy.perception.april_tag.track_apriltag import AprilTagTracker
from reset_policy.perception.cube_tracker import CubeTracker

# Control
from reset_policy.control.dynamixel_executor import DynamixelExecutor
from reset_policy.control.safety_filter import SafetyFilter

from dynamixel_sdk import (
    PortHandler, PacketHandler, GroupSyncWrite, GroupSyncRead,
)

# Environment
from reset_policy.environment.occupancy_grid import OccupancyGrid
from reset_policy.environment.reward import RewardFunction
from reset_policy.environment.observation import ObservationBuilder
from reset_policy.environment.environment import ResetPolicyEnv


CAMERA_INTRINSICS = Path(__file__).resolve().parent.parent.parent / "configs/camera_intrinsics.yaml"
TAG_SIZE = 0.05
WORLD_TAG_ID = 0
CUBE_TAG_ID = 1


def create_dynamixel_bus():
    """Initialize Dynamixel communication."""
    port = PortHandler(config.dynamixel.port_name)
    packet = PacketHandler(config.dynamixel.protocol_version)

    if not port.openPort():
        raise RuntimeError("Cannot open Dynamixel port")
    if not port.setBaudRate(config.dynamixel.baudrate):
        raise RuntimeError("Cannot set baudrate")

    print("Dynamixel connected")

    for motor_id in config.motor.motor_ids:
        model, comm, error = packet.ping(port, motor_id)
        print(
            f"Motor {motor_id}: {model}, "
            f"{packet.getTxRxResult(comm)}, "
            f"{packet.getRxPacketError(error)}"
        )

    sync_write = GroupSyncWrite(port, packet, 116, 4)

    sync_reads = {
        "positions":    GroupSyncRead(port, packet, config.dynamixel.addr_present_position, 4),
        "currents":     GroupSyncRead(port, packet, config.dynamixel.addr_present_current, 2),
        "voltages":     GroupSyncRead(port, packet, config.dynamixel.addr_present_input_voltage, 2),
        "temperatures": GroupSyncRead(port, packet, config.dynamixel.addr_present_temperature, 1),
        "hw_status":    GroupSyncRead(port, packet, config.dynamixel.addr_hardware_error_status, 1),
        "pwm":          GroupSyncRead(port, packet, config.dynamixel.addr_present_pwm, 2),
        "velocity":     GroupSyncRead(port, packet, config.dynamixel.addr_present_velocity, 4),
        "torque":       GroupSyncRead(port, packet, config.dynamixel.addr_torque_enable, 1),
    }

    for sr in sync_reads.values():
        for motor_id in config.motor.motor_ids:
            sr.addParam(motor_id)

    return port, packet, sync_write, sync_reads


def create_environment(render_mode: str = "human"):
    """Create environment with all components."""
    x_min = config.environment.x_min
    x_max = config.environment.x_max
    y_min = config.environment.y_min
    y_max = config.environment.y_max
    cell_size_m = config.environment.cell_size_m

    print(f"Board bounds: x=[{x_min:.3f}, {x_max:.3f}], y=[{y_min:.3f}, {y_max:.3f}] m")
    print(f"Cell size: {cell_size_m:.3f} m")

    print("Initializing camera...")
    cameras = CameraPair()

    print("Initializing AprilTag tracker...")
    tag_tracker = AprilTagTracker(
        yaml_file=str(CAMERA_INTRINSICS),
        tag_size=TAG_SIZE,
        world_tag_id=WORLD_TAG_ID,
        cube_tag_id=CUBE_TAG_ID,
    )

    cube_tracker = CubeTracker(camera=cameras, tracker=tag_tracker)

    port, packet, sync_write, sync_reads = create_dynamixel_bus()

    executor = DynamixelExecutor(
        port_handler=port,
        packet_handler=packet,
        motor_ids=config.motor.motor_ids,
        group_sync_write=sync_write,
        group_sync_reads=sync_reads,
    )
    executor.initialize()

    grid = OccupancyGrid(
        x_min=x_min, x_max=x_max,
        y_min=y_min, y_max=y_max,
        cell_size_m=cell_size_m,
    )

    reward_function = RewardFunction()

    observation_builder = ObservationBuilder(
        cube_tracker=cube_tracker,
        executor=executor,
        x_min=x_min, x_max=x_max,
        y_min=y_min, y_max=y_max,
    )

    safety_filter = SafetyFilter(motor_ids=config.motor.motor_ids)

    env = ResetPolicyEnv(
        executor=executor,
        observation_builder=observation_builder,
        cube_tracker=cube_tracker,
        occupancy_grid=grid,
        reward_function=reward_function,
        safety_filter=safety_filter,
        render_mode=render_mode,
        max_steps=config.environment.max_steps,
        high_current_threshold=config.environment.high_current_threshold,
        safety_penalty_weight=config.environment.safety_penalty_weight,
    )

    return env, cameras, executor, port