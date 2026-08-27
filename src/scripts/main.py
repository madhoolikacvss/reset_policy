"""
Main entry point for reset policy training.
"""

import torch
import argparse
import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)


# Keep enabled while debugging PPO numerical issues
torch.autograd.set_detect_anomaly(True)

# Perception
from reset_policy.perception.cameras import CameraPair
from reset_policy.perception.april_tag.track_apriltag import AprilTagTracker
from reset_policy.perception.cube_tracker import CubeTracker

# Control
from reset_policy.control.dynamixel_executor import DynamixelExecutor
from reset_policy.control.safety_filter import SafetyFilter

# Environment
from reset_policy.environment.occupancy_grid import OccupancyGrid
from reset_policy.environment.reward import RewardFunction
from reset_policy.environment.observation import ObservationBuilder
from reset_policy.environment.environment import ResetPolicyEnv
from reset_policy.environment.renderer import BoardRenderer

# RL
from train import train
from reset_policy.config import config

# Constants
CAMERA_INTRINSICS = Path(__file__).resolve().parent.parent / "configs/camera_intrinsics.yaml"
TAG_SIZE = 0.05
WORLD_TAG_ID = 0
CUBE_TAG_ID = 1


def create_dynamixel_bus():
    """Initialize Dynamixel communication."""
    from dynamixel_sdk import PortHandler, PacketHandler, GroupSyncWrite
    
    port = PortHandler(config.dynamixel.port_name)
    packet = PacketHandler(config.dynamixel.protocol_version)
    
    if not port.openPort():
        raise RuntimeError("Cannot open Dynamixel port")
    
    if not port.setBaudRate(config.dynamixel.baudrate):
        raise RuntimeError("Cannot set baudrate")
    
    print("Dynamixel connected")
    
    # Ping all motors
    for motor_id in config.motor.motor_ids:
        model, comm, error = packet.ping(port, motor_id)
        print(f"Motor {motor_id}: {model}, {packet.getTxRxResult(comm)}, {packet.getRxPacketError(error)}")
    
    # Create sync write for goal position (address 116, 4 bytes)
    sync_write = GroupSyncWrite(port, packet, 116, 4)
    
    return port, packet, sync_write


def create_environment():
    """Create environment with all components."""
    
    # Load board bounds from config
    x_min = config.environment.x_min
    x_max = config.environment.x_max
    y_min = config.environment.y_min
    y_max = config.environment.y_max
    cell_size_m = config.environment.cell_size_m
    
    print(f"Board bounds: x=[{x_min:.3f}, {x_max:.3f}], y=[{y_min:.3f}, {y_max:.3f}] m")
    print(f"Cell size: {cell_size_m:.3f} m")
    
    # Initialize camera
    print("Initializing camera...")
    cameras = CameraPair()
    
    # Initialize AprilTag tracker
    print("Initializing AprilTag tracker...")
    tag_tracker = AprilTagTracker(
        yaml_file=str(CAMERA_INTRINSICS),
        tag_size=TAG_SIZE,
        world_tag_id=WORLD_TAG_ID,
        cube_tag_id=CUBE_TAG_ID,
    )
    
    cube_tracker = CubeTracker(
        camera=cameras,
        tracker=tag_tracker,
    )
    
    # Initialize Dynamixels
    port, packet, sync_write = create_dynamixel_bus()
    
    executor = DynamixelExecutor(
        port_handler=port,
        packet_handler=packet,
        motor_ids=config.motor.motor_ids,
        group_sync_write=sync_write,
    )
    executor.initialize()
    
    # Initialize occupancy grid
    grid = OccupancyGrid(
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        cell_size_m=cell_size_m,
    )
    
    # Initialize reward function (uses defaults from config)
    reward_function = RewardFunction()
    
    # Initialize observation builder
    observation_builder = ObservationBuilder(
        cube_tracker=cube_tracker,
        executor=executor,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )
    
    # Create safety filter from config
    safety_filter = SafetyFilter(motor_ids=config.motor.motor_ids)
    
    # Create environment
    env = ResetPolicyEnv(
        executor=executor,
        observation_builder=observation_builder,
        cube_tracker=cube_tracker,
        occupancy_grid=grid,
        reward_function=reward_function,
        safety_filter=safety_filter,
        render_mode="human",
        max_steps=config.environment.max_steps,
        target_coverage=config.environment.target_coverage,
        high_current_threshold=config.environment.high_current_threshold,
        safety_penalty_weight=config.environment.safety_penalty_weight,
    )
    
    return env, cameras, executor, port


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--episodes', type=int, default=config.training.episodes)
    parser.add_argument('--save_every', type=int, default=config.training.save_every)
    parser.add_argument('--render_every', type=int, default=config.training.render_every)
    args = parser.parse_args()
    
    # Update config with command line args
    config.training.episodes = args.episodes
    config.training.save_every = args.save_every
    config.training.render_every = args.render_every
    config.training.resume_from = args.resume
    
    env = None
    cameras = None
    executor = None
    port = None
    
    try:
        env, cameras, executor, port = create_environment()
        print("\nEnvironment initialized successfully")
        
        # Train
        actor_critic = train(env, config)
        print("Training finished")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        
    except Exception as e:
        print(f"\nTraining stopped with error: {type(e).__name__}: {e}")
        raise
        
    finally:
        print("\n=== CLEANUP START ===")
        
        # Close environment (handles executor shutdown)
        if env is not None:
            try:
                env.close()
                print("Environment closed")
            except Exception as e:
                print(f"Environment cleanup error: {e}")
        
        # Close cameras
        if cameras is not None:
            try:
                cameras.close()
                print("Cameras closed")
            except Exception as e:
                print(f"Camera cleanup error: {e}")
        
        # Final port cleanup
        if port is not None:
            try:
                if port.is_open:
                    port.closePort()
                    print("Port closed")
            except Exception as e:
                print(f"Port cleanup error: {e}")
        
        print("=== CLEANUP COMPLETE ===")


if __name__ == "__main__":
    main()