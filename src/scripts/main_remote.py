# main.py - Modified version with live camera feed

from dynamixel_sdk import *
from pathlib import Path
import torch
import cv2  # Add this import
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Perception
from reset_policy.perception.cameras import CameraPair
from reset_policy.perception.april_tag.track_apriltag import AprilTagTracker
from reset_policy.perception.cube_tracker import CubeTracker

# Control
from reset_policy.control.dynamixel_executor import DynamixelExecutor

# Environment
from reset_policy.environment.occupancy_grid import OccupancyGrid
from reset_policy.environment.reward import RewardFunction
from reset_policy.environment.observation import ObservationBuilder
from reset_policy.environment.environment import ResetPolicyEnv

# RL
from train import train

CAMERA_INTRINSICS = "camera_intrinsics.yaml"
TAG_SIZE = 0.04
WORLD_TAG_ID = 1
CUBE_TAG_ID = 0

PORT_NAME = (
    "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT89FK0C-if00-port0"
)
BAUDRATE = 1000000
PROTOCOL_VERSION = 2.0

MOTOR_IDS = [16, 17, 18, 19]

# Dynamixel setup
def create_dynamixel_bus():
    port = PortHandler(PORT_NAME)
    packet = PacketHandler(PROTOCOL_VERSION)

    if not port.openPort():
        raise RuntimeError("Cannot open Dynamixel port")

    if not port.setBaudRate(BAUDRATE):
        raise RuntimeError("Cannot set baudrate")

    print("Dynamixel connected")

    for motor in MOTOR_IDS:
        model, comm, error = packet.ping(port, motor)
        print(motor, model, packet.getTxRxResult(comm), packet.getRxPacketError(error))

    sync_write = GroupSyncWrite(port, packet, 116, 4)

    return (port, packet, sync_write)

# Environment creation
def create_environment():
    print("Initializing camera")
    cameras = CameraPair()

    print("Initializing AprilTag tracker")
    tag_tracker = AprilTagTracker(
        yaml_file=CAMERA_INTRINSICS,
        tag_size=TAG_SIZE,
        world_tag_id=WORLD_TAG_ID,
        cube_tag_id=CUBE_TAG_ID,
    )

    cube_tracker = CubeTracker(
        camera=cameras,
        tracker=tag_tracker,
    )
    
    (port, packet, sync_write) = create_dynamixel_bus()
    
    executor = DynamixelExecutor(
        port_handler=port,
        packet_handler=packet,
        motor_ids=MOTOR_IDS,
        group_sync_write=sync_write,
    )
    executor.initialize()
    
    grid = OccupancyGrid(
        board_width_m=0.30,
        board_height_m=0.20,
        cell_size_m=0.01,
    )
    
    reward_function = RewardFunction()
    
    observation_builder = ObservationBuilder(
        cube_tracker=cube_tracker,
        executor=executor,
    )

    env = ResetPolicyEnv(
        executor=executor,
        observation_builder=observation_builder,
        cube_tracker=cube_tracker,
        occupancy_grid=grid,
        reward_function=reward_function,
        render_mode="human",
    )

    return (env, cameras, executor, port)

# Main
def main():
    env = None
    cameras = None
    executor = None
    port = None
    
    # Flag to control camera display
    show_camera = True  # Set to False to disable camera feed
    
    try:
        (env, cameras, executor, port) = create_environment()
        
        print("\nEnvironment initialized")
        print("Press 'q' in camera window to close it")
        print("Press Ctrl+C to stop training")
        
        # Create a separate thread for camera display if you want non-blocking
        # Or run the training with periodic camera updates
        actor_critic = train(
            env,
            episodes=1000,
            save_every=50,
            # Pass camera and show_camera flag to training function
            camera_feed=(cameras, show_camera) if show_camera else None,
        )
        print("Training finished")

    except KeyboardInterrupt:
        print("\nInterrupted")

    finally:
        if env is not None:
            env.close()
        if cameras is not None:
            cameras.close()
        if port is not None:
            port.closePort()
        cv2.destroyAllWindows()
        print("Shutdown complete")

if __name__ == "__main__":
    main()