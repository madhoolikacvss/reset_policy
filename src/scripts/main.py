from dynamixel_sdk import *
import torch

# Keep enabled while debugging PPO numerical issues.
torch.autograd.set_detect_anomaly(True)

import yaml
import argparse

import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)


# =========================================================
# Perception
# =========================================================

from reset_policy.perception.cameras import CameraPair
from reset_policy.perception.april_tag.track_apriltag import (
    AprilTagTracker,
)
from reset_policy.perception.cube_tracker import CubeTracker


# =========================================================
# Control
# =========================================================

from reset_policy.control.dynamixel_executor import (
    DynamixelExecutor,
)
from reset_policy.control.safety_filter import create_safety_filter


# =========================================================
# Environment
# =========================================================

from reset_policy.environment.occupancy_grid import (
    OccupancyGrid,
)
from reset_policy.environment.reward import (
    RewardFunction,
)
from reset_policy.environment.observation import (
    ObservationBuilder,
)
from reset_policy.environment.environment import (
    ResetPolicyEnv,
)


# =========================================================
# RL
# =========================================================

from train import train


# =========================================================
# Constants
# =========================================================

CAMERA_INTRINSICS = str(
    Path(__file__).resolve().parent.parent
    / "configs/camera_intrinsics.yaml"
)

TAG_SIZE = 0.04

WORLD_TAG_ID = 0
CUBE_TAG_ID = 1

PORT_NAME = (
    "/dev/serial/by-id/"
    "usb-FTDI_USB__-__Serial_Converter_FT89FK0C"
    "-if00-port0"
)

BAUDRATE = 1000000
PROTOCOL_VERSION = 2.0

MOTOR_IDS = [
    16,
    17,
    18,
    19,
]


# =========================================================
# Configuration
# =========================================================

def load_config(config_path="configs/config.yaml"):
    """
    Load configuration from YAML file.
    """

    config_file = (
        Path(__file__).resolve().parent.parent
        / config_path
    )

    if not config_file.exists():

        print(
            f"Warning: {config_file} not found. "
            "Using default board bounds."
        )

        return {
            "board_bounds": {
                "x_min": 0.035,
                "x_max": 0.830,
                "y_min": 0.140,
                "y_max": 0.654,
            },
            "grid_config": {
                "cell_size_m": 0.01,
            },
        }

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    return config


# =========================================================
# Dynamixel setup
# =========================================================

def create_dynamixel_bus():

    port = PortHandler(PORT_NAME)
    packet = PacketHandler(PROTOCOL_VERSION)

    if not port.openPort():
        raise RuntimeError(
            "Cannot open Dynamixel port"
        )

    if not port.setBaudRate(BAUDRATE):
        raise RuntimeError(
            "Cannot set baudrate"
        )

    print("Dynamixel connected")

    for motor in MOTOR_IDS:

        model, comm, error = packet.ping(
            port,
            motor,
        )

        print(
            motor,
            model,
            packet.getTxRxResult(comm),
            packet.getRxPacketError(error),
        )

    # Goal Position = 116
    # Length = 4 bytes
    sync_write = GroupSyncWrite(
        port,
        packet,
        116,
        4,
    )

    return (
        port,
        packet,
        sync_write,
    )


# =========================================================
# Environment creation
# =========================================================

def create_environment():

    # -----------------------------------------------------
    # Load configuration
    # -----------------------------------------------------

    config = load_config(
        "configs/config.yaml"
    )

    board_bounds = config.get(
        "board_bounds",
        {},
    )

    grid_config = config.get(
        "grid_config",
        {},
    )

    x_min = board_bounds.get(
        "x_min",
        0.035,
    )

    x_max = board_bounds.get(
        "x_max",
        0.830,
    )

    y_min = board_bounds.get(
        "y_min",
        0.140,
    )

    y_max = board_bounds.get(
        "y_max",
        0.654,
    )

    cell_size_m = grid_config.get(
        "cell_size_m",
        0.01,
    )

    print(
        "Board bounds loaded from config:"
    )

    print(
        f"  x: [{x_min:.3f}, {x_max:.3f}] m"
    )

    print(
        f"  y: [{y_min:.3f}, {y_max:.3f}] m"
    )

    print(
        f"  Cell size: {cell_size_m:.3f} m"
    )


    # -----------------------------------------------------
    # Camera
    # -----------------------------------------------------

    print("Initializing camera")

    cameras = CameraPair()


    # -----------------------------------------------------
    # AprilTag
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # Dynamixels
    # -----------------------------------------------------

    (
        port,
        packet,
        sync_write,
    ) = create_dynamixel_bus()

    executor = DynamixelExecutor(
        port_handler=port,
        packet_handler=packet,
        motor_ids=MOTOR_IDS,
        group_sync_write=sync_write,
    )

    executor.initialize()


    # -----------------------------------------------------
    # Occupancy grid
    # -----------------------------------------------------

    grid = OccupancyGrid(
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        cell_size_m=cell_size_m,
    )


    # -----------------------------------------------------
    # Reward
    # -----------------------------------------------------
    #
    # IMPORTANT:
    #
    # The reward function is now threshold-based.
    #
    # We therefore DO NOT pass:
    #
    #   current_weight
    #   current_change_weight
    #
    # Those belonged to the previous reward formulation.
    #
    # The current thresholds/rewards are defined inside
    # RewardFunction.
    #
    # -----------------------------------------------------

    reward_function = RewardFunction(
        coverage_weight=1.0,
    )


    # -----------------------------------------------------
    # Observation
    # -----------------------------------------------------

    observation_builder = ObservationBuilder(
        cube_tracker=cube_tracker,
        executor=executor,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )

    # =====================================================
    # Create Safety Filter
    # =====================================================
    
    safety_config = {
        'pair_horizontal': [16, 17],
        'pair_vertical': [18, 19],
        'max_pair_current': 800.0,
        'max_single_current': 600.0,
        'max_position': 8000.0,
        'min_voltage': 4.0,
        'critical_voltage': 3.5,
        'voltage_scale_factor': 0.3,
        'heavy_load_threshold': 400.0,       # <-- NEW
        'critical_load_threshold': 700.0,    # <-- NEW
        'heavy_load_scale': 0.3,             # <-- NEW
        'moderate_load_scale': 0.6,          # <-- NEW
        'current_scale_factor': 0.25,
        'position_scale_factor': 0.5,
        'tension_threshold': 0.7,
        'enable_current_safety': True,
        'enable_position_safety': True,
        'enable_tension_safety': True,
        'enable_voltage_safety': True,
        'enable_current_aware_safety': True, # <-- NEW
        'log_interventions': True,
    }
    
    safety_filter = create_safety_filter(
        motor_ids=MOTOR_IDS,
        config=safety_config,
    )
    
    # -----------------------------------------------------
    # Environment
    # -----------------------------------------------------

    env = ResetPolicyEnv(
        executor=executor,
        observation_builder=observation_builder,
        cube_tracker=cube_tracker,
        occupancy_grid=grid,
        reward_function=reward_function,
        render_mode="human",

        # Current safety thresholds
        #
        # < 500 mA       -> safe
        # 500-1800 mA     -> moderate
        # 1800-2500 mA    -> high
        # > 2500 mA       -> terminate
        safe_current_threshold=500.0,
        moderate_current_threshold=1800.0,
        high_current_threshold=2500.0,

        # Episode configuration
        max_steps=100,
        target_coverage=0.95,
        safety_filter=safety_filter,
        safety_penalty_weight=2.0,
    )


    return (
        env,
        cameras,
        executor,
        port,
    )


# =========================================================
# Main
# =========================================================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, help='Path to checkpoint file to resume from')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes to train')
    parser.add_argument('--save_every', type=int, default=5, help='Save checkpoint every N episodes')
    args = parser.parse_args()

    env = None
    cameras = None
    executor = None
    port = None

    try:

        (
            env,
            cameras,
            executor,
            port,
        ) = create_environment()

        print(
            "\nEnvironment initialized"
        )

        # -------------------------------------------------
        # PPO Training
        # -------------------------------------------------

        actor_critic = train(
            env,
            episodes=args.episodes,
            save_every=args.save_every,
            resume_from=args.resume,
        )
        print(
            "Training finished"
        )


    except KeyboardInterrupt:

        print(
            "\nInterrupted"
        )


    except Exception as e:

        print(
            "\nTraining stopped error:"
        )

        print(
            f"{type(e).__name__}: {e}"
        )

        # Re-raise so the traceback is still available
        # during development.
        raise


    finally:
        print("\n=== CLEANUP START ===")
        
        # =====================================================
        # 1. Close environment (this handles executor shutdown)
        # =====================================================
        
        if env is not None:
            try:
                env.close()
                print("Environment closed")
            except Exception as e:
                print(f"Environment cleanup error: {e}")
        
        # =====================================================
        # 2. Double-check executor (if env.close() failed)
        # =====================================================
        
        if executor is not None:
            # Check if executor still has a port
            try:
                if hasattr(executor, 'port') and executor.port is not None:
                    if executor.port.is_open:
                        print("Executor port still open, force closing...")
                        executor.port.closePort()
            except Exception as e:
                print(f"Executor force close error: {e}")
        
        # =====================================================
        # 3. Close cameras
        # =====================================================
        
        if cameras is not None:
            try:
                cameras.close()
                print("Cameras closed")
            except Exception as e:
                print(f"Camera cleanup error: {e}")
        
        # =====================================================
        # 4. Final port cleanup
        # =====================================================
        
        if port is not None:
            try:
                if port.is_open:
                    port.closePort()
                    print("Port closed")
                else:
                    print("Port already closed")
            except Exception as e:
                print(f"Port cleanup error: {e}")
        
        print("=== CLEANUP COMPLETE ===")


# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":
    main()