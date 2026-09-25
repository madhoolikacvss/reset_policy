"""
Autonomous test harness for a trained PPO policy.

Runs N episodes back-to-back in a single session.

Each episode:
  - Writes per-step data to logs/test_results/per_episode/episode_NNN.csv
  - Appends one summary row to logs/test_results/results.csv

Episode counter auto-increments by reading existing files in per_episode/.

Usage:
    python scripts/test_policy.py --checkpoint checkpoints/ppo_checkpoint_50.pth
    python scripts/test_policy.py --checkpoint checkpoints/ppo_checkpoint_50.pth --episodes 10 --max-steps 200 --pause 5
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# ---- project imports ----
sys.path.append(str(Path(__file__).resolve().parent.parent))

from reset_policy.perception.cameras import CameraPair
from reset_policy.perception.april_tag.track_apriltag import AprilTagTracker
from reset_policy.perception.cube_tracker import CubeTracker

from reset_policy.control.dynamixel_executor import DynamixelExecutor
from reset_policy.control.safety_filter import SafetyFilter

from reset_policy.environment.occupancy_grid import OccupancyGrid
from reset_policy.environment.reward import RewardFunction
from reset_policy.environment.observation import ObservationBuilder
from reset_policy.environment.environment import ResetPolicyEnv

from reset_policy.rl.actor_critic import ActorCritic

from reset_policy.config import config


# ---------------- Constants ----------------

CAMERA_INTRINSICS = str(
    Path(__file__).resolve().parent.parent / "configs/camera_intrinsics.yaml"
)
TAG_SIZE = 0.05
WORLD_TAG_ID = 0
CUBE_TAG_ID = 1

SUCCESS_LEFT_X = config.environment.x_min + 0.02

TEST_RESULTS_DIR = Path("/home/madhoolika/workspace/reset_policy/src/logs/test_results")
PER_EPISODE_DIR = TEST_RESULTS_DIR / "per_episode"
RESULTS_CSV = TEST_RESULTS_DIR / "results.csv"


# ---------------- Setup ----------------

def create_dynamixel_bus():
    from dynamixel_sdk import PortHandler, PacketHandler, GroupSyncWrite, GroupSyncRead

    port = PortHandler(config.dynamixel.port_name)
    packet = PacketHandler(config.dynamixel.protocol_version)

    if not port.openPort():
        raise RuntimeError("Cannot open Dynamixel port")
    if not port.setBaudRate(config.dynamixel.baudrate):
        raise RuntimeError("Cannot set baudrate")

    print("Dynamixel connected")
    for motor_id in config.motor.motor_ids:
        model, comm, error = packet.ping(port, motor_id)
        print(f"Motor {motor_id}: {model}, "
              f"{packet.getTxRxResult(comm)}, {packet.getRxPacketError(error)}")

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
        for m in config.motor.motor_ids:
            sr.addParam(m)

    return port, packet, sync_write, sync_reads


def create_env():
    x_min = config.environment.x_min
    x_max = config.environment.x_max
    y_min = config.environment.y_min
    y_max = config.environment.y_max
    cell_size_m = config.environment.cell_size_m

    cameras = CameraPair()
    tag_tracker = AprilTagTracker(
        yaml_file=CAMERA_INTRINSICS,
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
        render_mode=None,
        max_steps=config.environment.max_steps,
        high_current_threshold=config.environment.high_current_threshold,
        safety_penalty_weight=config.environment.safety_penalty_weight,
    )
    return env, cameras, executor, port


# ---------------- Bookkeeping ----------------

def initialize_results_files():
    TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PER_EPISODE_DIR.mkdir(parents=True, exist_ok=True)

    if not RESULTS_CSV.exists():
        with open(RESULTS_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode_number",
                "timestamp",
                "checkpoint",
                "cube_start_x", "cube_start_y",
                "cube_end_x", "cube_end_y",
                "cube_dx", "cube_dy",
                "vx_avg", "vy_avg",
                "total_velocity_reward",
                "total_steps",
                "avg_velocity_reward_per_step",
                "avg_v_error",
                "termination_reason",
                "direction_correct",
                "success",
            ])
        print(f"Created {RESULTS_CSV}")


def get_next_episode_number():
    existing = sorted(PER_EPISODE_DIR.glob("episode_*.csv"))
    return len(existing) + 1


# ---------------- Episode runner ----------------

def run_single_episode(env, actor_critic, checkpoint_path: Path, max_steps: int):
    """
    Run one episode of the test. Returns nothing; writes files as side effect.
    `env` and `actor_critic` are already constructed and reused.
    """
    initialize_results_files()
    episode_number = get_next_episode_number()
    episode_path = PER_EPISODE_DIR / f"episode_{episode_number:03d}.csv"

    print(f"\n===== TEST EPISODE {episode_number} =====")
    print(f"Per-step file: {episode_path}")
    print(f"Summary file:  {RESULTS_CSV}")

    # ---- reset ----
    state, _ = env.reset(options={"episode_num": episode_number})

    step_rows = []
    total_velocity_reward = 0.0
    total_v_error = 0.0
    n_steps_logged = 0

    if env.last_valid_observation is not None:
        start_x = env.last_valid_observation.cube_x
        start_y = env.last_valid_observation.cube_y
        print(f"Start cube: ({start_x:.3f}, {start_y:.3f})")
    else:
        start_x = start_y = None

    terminated = False
    truncated = False
    steps = 0
    final_info = {}

    while not (terminated or truncated) and steps < max_steps:
        state_t = torch.tensor(state, dtype=torch.float32,
                               device=config.training.device)
        with torch.no_grad():
            action, _, _ = actor_critic.act(state_t)

        action_np = action.cpu().numpy().astype(np.float32)
        next_state, reward, terminated, truncated, info = env.step(action_np)

        cube_pos = info.get("cube_position", (None, None))
        v_error = info.get("v_error", None)
        velocity_reward = info.get("velocity_reward", 0.0)

        step_rows.append({
            "step": steps + 1,
            "cube_x": cube_pos[0],
            "cube_y": cube_pos[1],
            "velocity_reward": velocity_reward,
            "v_error": v_error,
            "termination_reason": info.get("termination_reason", ""),
        })

        total_velocity_reward += float(velocity_reward) if velocity_reward is not None else 0.0
        if v_error is not None:
            total_v_error += float(v_error)
            n_steps_logged += 1

        steps += 1
        final_info = info
        state = next_state

        if steps % 20 == 0 and cube_pos[0] is not None:
            print(f"  step {steps:3d}  cube=({cube_pos[0]:.3f}, {cube_pos[1]:.3f})  "
                  f"v_err={v_error if v_error is not None else 0:.4f}  "
                  f"reason={info.get('termination_reason', '?')}")

    # ---- final cube position ----
    if env.last_valid_observation is not None:
        end_x = env.last_valid_observation.cube_x
        end_y = env.last_valid_observation.cube_y
    else:
        end_x = end_y = None

    dx = (end_x - start_x) if (start_x is not None and end_x is not None) else None
    dy = (end_y - start_y) if (start_y is not None and end_y is not None) else None

    # ---- achieved velocity (average over whole episode) ----
    elapsed = steps * config.dynamixel.action_duration
    vx_avg = (dx / elapsed) if (dx is not None and elapsed > 0) else None
    vy_avg = (dy / elapsed) if (dy is not None and elapsed > 0) else None

    # ---- success ----
    direction_correct = (dx is not None and dx < 0)
    success = (end_x is not None and end_x <= SUCCESS_LEFT_X)

    # ---- write per-episode CSV ----
    with open(episode_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["step", "cube_x", "cube_y", "velocity_reward",
                        "v_error", "termination_reason"],
        )
        writer.writeheader()
        writer.writerows(step_rows)
    print(f"Wrote per-step data to {episode_path}")

    # ---- append summary row ----
    avg_vr_per_step = (total_velocity_reward / steps) if steps > 0 else 0.0
    avg_v_error = (total_v_error / n_steps_logged) if n_steps_logged > 0 else None

    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            episode_number,
            datetime.now().isoformat(timespec="seconds"),
            str(checkpoint_path),
            start_x, start_y,
            end_x, end_y,
            dx, dy,
            round(vx_avg, 5) if vx_avg is not None else "",
            round(vy_avg, 5) if vy_avg is not None else "",
            round(total_velocity_reward, 4),
            steps,
            round(avg_vr_per_step, 5),
            round(avg_v_error, 5) if avg_v_error is not None else "",
            final_info.get("termination_reason", ""),
            direction_correct,
            success,
        ])
    print(f"Appended summary row to {RESULTS_CSV}")

    # ---- print summary ----
    print(f"\n----- Episode {episode_number} summary -----")
    print(f"  Steps:                    {steps}")
    print(f"  Cube start:               ({start_x}, {start_y})")
    print(f"  Cube end:                 ({end_x}, {end_y})")
    print(f"  Displacement:             dx={dx}, dy={dy}")
    print(f"  Avg velocity:             vx={vx_avg}, vy={vy_avg}")
    print(f"  Total velocity reward:    {total_velocity_reward:.3f}")
    print(f"  Avg velocity reward/step: {avg_vr_per_step:.4f}")
    print(f"  Avg v_error:              {avg_v_error}")
    print(f"  Termination reason:       {final_info.get('termination_reason', '?')}")
    print(f"  Direction correct:        {direction_correct}")
    print(f"  Success (reached left):   {success}")


# ---------------- Main ----------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/ppo_checkpoint_50.pth",
        help="Path to checkpoint .pth file",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of test episodes to run in this session",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=1.0,
        help="Seconds to pause between episodes (for repositioning the cube)",
    )
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    initialize_results_files()

    print(f"\n===== TEST SESSION =====")
    print(f"Checkpoint:       {ckpt}")
    print(f"Episodes:         {args.episodes}")
    print(f"Max steps/ep:     {args.max_steps}")
    print(f"Pause between:    {args.pause} s")

    # ---- build env ONCE for the whole session ----
    env, cameras, executor, port = create_env()

    try:
        # ---- load model ONCE ----
        actor_critic = ActorCritic().to(config.training.device)
        ckpt_data = torch.load(ckpt, map_location=config.training.device)
        actor_critic.load_state_dict(ckpt_data["model_state_dict"])
        actor_critic.eval()
        print(f"Loaded checkpoint (episode {ckpt_data.get('episode', '?')})")

        # ---- loop over episodes ----
        for i in range(args.episodes):
            print(f"\n########## Session episode {i + 1}/{args.episodes} ##########")
            try:
                run_single_episode(env, actor_critic, ckpt, args.max_steps)
            except KeyboardInterrupt:
                print("\n[KeyboardInterrupt] Stopping session early.")
                break
            except Exception as e:
                print(f"[ERROR] Episode failed with: {type(e).__name__}: {e}")
                print("[ERROR] Continuing to next episode...")
                continue

            # ---- pause between episodes ----
            if i < args.episodes - 1:
                print(f"\nPausing for {args.pause} s before next episode...")
                time.sleep(args.pause)

    finally:
        print("\n=== CLEANUP ===")
        try:
            if env is not None:
                env.close()
        except Exception as e:
            print(f"env.close error: {e}")
        try:
            if cameras is not None:
                cameras.close()
        except Exception as e:
            print(f"cameras.close error: {e}")
        try:
            if port is not None and port.is_open:
                port.closePort()
        except Exception as e:
            print(f"port.close error: {e}")
        print("Cleanup done.")


if __name__ == "__main__":
    main()