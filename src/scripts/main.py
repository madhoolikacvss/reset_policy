"""
Camera
 -> AprilTagTracker
 -> CubeTracker

Dynamixels
 -> DynamixelExecutor

CubeTracker + Executor
 -> ObservationBuilder

ObservationBuilder + Reward + OccupancyGrid
 -> ResetPolicyEnv

"""

from dynamixel_sdk import *
import sys
from pathlib import Path 
sys.path.append(str(Path(__file__).resolve().parent))
from reset_policy.src.reset.perception.cameras import CameraPair
from reset_policy.src.reset.perception.april_tag.track_apriltag import AprilTagTracker
from reset_policy.src.reset.perception.cube_tracker import CubeTracker

from reset_policy.src.reset.control.dynamixel_executor import DynamixelExecutor

from reset_policy.src.reset_policy.environment.occupancy_grid import OccupancyGrid
from reset_policy.src.reset_policy.environment.reward import RewardFunction
from reset_policy.src.reset_policy.environment.observation import ObservationBuilder
from reset_policy.src.reset_policy.environment.environment import ResetPolicyEnv


import gymnasium
import numpy as np
import time
import cv2
import imageio
from pathlib import Path




# Camera / AprilTag Configuration
CAMERA_INTRINSICS = "camera_intrinsics.yaml"

TAG_SIZE = 0.04

WORLD_TAG_ID = 1
CUBE_TAG_ID = 0

# Dynamixel Configuration

PORT_NAME = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT89FK0C-if00-port0"
BAUDRATE = 1000000

PROTOCOL_VERSION = 2.0


MOTOR_IDS = [
    16,
    17,
    18,
    19,
]



# Dynamixel Communication
def create_dynamixel_bus():

    port = PortHandler(PORT_NAME)
    packet = PacketHandler(PROTOCOL_VERSION)

    if not port.openPort():
        raise RuntimeError("Cannot open Dynamixel port")

    if not port.setBaudRate(BAUDRATE):
        raise RuntimeError("Cannot set baudrate")

    for m in MOTOR_IDS:
        model, comm, dxl_error = packet.ping(port, m)
        print(
            m,
            model,
            packet.getTxRxResult(comm),
            packet.getRxPacketError(dxl_error),
        )

    sync_write = GroupSyncWrite(port, packet, 116, 4)
    return (port, packet, sync_write)


def main():
    port = None
    cameras = None
    executor = None

    try:
        # Camera
        print("Initializing camera")
        cameras = CameraPair()

        # AprilTag Tracker
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

        # Dynamixels
        (port,packet,sync_write) = create_dynamixel_bus()

        executor = DynamixelExecutor(
            port_handler=port,
            packet_handler=packet,
            motor_ids=MOTOR_IDS,
            group_sync_write=sync_write,
        )

        executor.initialize()

        # RL Components
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
        print("\nEnvironment initialized")

        # Temporary test episode
        # TODO: Replace this with PPO later
        observation, info = env.reset()
        print("Initial observation:",observation)


        frames = []
        for step in range(10):

            print(f"\nStep {step}")
            # Random action
            # Later replaced by RL policy

            action = env.action_space.sample()
            print("Action:",action)

            obs, reward, terminated, truncated, info = env.step(action)
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            time.sleep(0.5)

            print("Reward:",reward)

            print("Cube:",obs[:3])
            if terminated or truncated:
                break

        if len(frames) > 0:
            save_dir = Path("episodes")
            save_dir.mkdir(exist_ok=True)


            cv2.imwrite(
                str(save_dir / "episode_final.png"),
                frames[-1][:,:,::-1]
            )


            imageio.mimsave(
                str(save_dir / "reset_policy_episode.mp4"),
                frames,
                fps=10,
            )

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        if 'env' in locals():
            env.close()
        elif executor is not None:
            executor.shutdown()

        if cameras is not None:
            cameras.close()

        if port is not None:
            port.closePort()

        print("Shutdown complete")

if __name__ == "__main__":

    main()