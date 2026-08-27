"""
Reset Policy Environment for 4-motor cube control system.

Coordinate System:
- Y-axis: horizontal (left/right)
- X-axis: vertical (up/down)

Motor Mapping:
- Motors 16, 17: horizontal pair (Y-axis)
- Motors 18, 19: vertical pair (X-axis)
- Action: +1 = pull (shorten string), -1 = release (lengthen string)

Error Handling Philosophy:
- Hardware errors are FATAL - no recovery is possible
- When hardware error occurs, terminate episode with penalty
- Reset should raise error if hardware error persists
"""

from __future__ import annotations

import time
import gymnasium as gym
import numpy as np
from gymnasium import spaces

import sys
from pathlib import Path
sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from reset_policy.control.safety_filter import SafetyFilter
from reset_policy.environment.observation import ObservationBuilder
from reset_policy.environment.reward import RewardFunction
from reset_policy.environment.occupancy_grid import OccupancyGrid
from reset_policy.environment.renderer import BoardRenderer
from reset_policy.config import config


class ResetPolicyEnv(gym.Env):
    """Gymnasium environment for cube control with safety filtering."""
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(
        self,
        executor,
        observation_builder: ObservationBuilder,
        cube_tracker,
        occupancy_grid: OccupancyGrid,
        reward_function: RewardFunction,
        safety_filter: SafetyFilter = None,
        render_mode: str = None,
        **kwargs,
    ):
        super().__init__()
        
        # Core components
        self.executor = executor
        self.obs_builder = observation_builder
        self.cube_tracker = cube_tracker
        self.grid = occupancy_grid
        self.reward_fn = reward_function
        self.safety_filter = safety_filter
        
        # Configuration (from config.py, overridable via kwargs)
        self.max_steps = kwargs.get('max_steps', config.environment.max_steps)
        self.target_coverage = kwargs.get('target_coverage', config.environment.target_coverage)
        self.action_duration = kwargs.get('action_duration', config.dynamixel.action_duration)
        self.safety_penalty_weight = kwargs.get('safety_penalty_weight', config.environment.safety_penalty_weight)
        self.high_current_threshold = kwargs.get('high_current_threshold', config.environment.high_current_threshold)
        
        # Spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        
        # Rendering
        self.render_mode = render_mode
        self.renderer = None
        if render_mode in ("human", "rgb_array"):
            self.renderer = BoardRenderer(
                board_width_cm=(self.grid.x_max - self.grid.x_min) * 100,
                board_height_cm=(self.grid.y_max - self.grid.y_min) * 100,
                cell_size_cm=self.grid.cell_size * 100,
            )
        
        # State tracking
        self.step_count = 0
        self.episode_count = 0
        self.last_valid_observation = None
        self.needs_reposition = False
        self.safety_interventions_total = 0
        
        # Hardware error tracking (fatal - no recovery)
        self.hardware_error_occurred = False
        self.hardware_error_details = None
        
        # Out of bounds penalty
        self.out_of_bound_penalty = kwargs.get('out_of_bound_penalty', -1.0)
    
    
    # Reset
    
    
    def reset(self, *, seed=None, options=None):
        """Reset environment for new episode."""
        super().reset(seed=seed)
        self.episode_count += 1
        time.sleep(5)
        
        print("\n================ ENV RESET ================")
        self.reward_fn.prev_currents = None
        
        # Check if hardware error occurred previously
        if self.hardware_error_occurred:
            raise RuntimeError(
                f"Hardware error occurred in previous episode. "
                f"Cannot continue training. Details: {self.hardware_error_details}"
            )
        
        # Reposition if needed
        if self.needs_reposition:
            print("Repositioning cube to center...")
            self.reposition_cube_to_center()
            self.needs_reposition = False
            self._sync_targets_with_actual_positions()
        
        # Reset bookkeeping
        self.grid.reset()
        self.step_count = 0
        
        # Establish initial positions
        self.obs_builder.reset()
        
        # Get first observation
        result = self.obs_builder.get_observation_result()
        
        if result.hardware_error:
            self.hardware_error_occurred = True
            self.hardware_error_details = {
                'error_ids': result.hardware_error_ids,
                'error_status': result.hardware_error_status,
                'message': result.error_message,
            }
            raise RuntimeError(
                f"Hardware error detected during reset: {result.error_message}. "
                f"Motor IDs: {result.hardware_error_ids}"
            )
        
        if result.observation is None:
            raise RuntimeError(f"Failed to get observation during reset: {result.error_message}")
        
        observation = result.observation
        
        # Mark starting position
        self.grid.visit(observation.cube_x, observation.cube_y)
        self.last_valid_observation = observation
        
        return observation.as_numpy(), {}
    
    
    # Step
    
    
    def step(self, action):
        """Execute one action step."""
        action = np.clip(action, -1.0, 1.0)
        self.step_count += 1
        
        # Apply safety filter
        safety_info = self._apply_safety_filter(action)
        action = safety_info['action']
        
        # Execute action
        execution_result = self.executor.execute(action)
        
        # Handle hardware errors during execution
        if execution_result.hardware_error:
            return self._handle_hardware_error(
                execution_result, safety_info, "action_execution"
            )
        
        # Non-hardware execution failure
        if not execution_result.success:
            raise RuntimeError(execution_result.error_message)
        
        # Wait for movement
        time.sleep(self.action_duration)
        
        # Get observation
        observation_result = self.obs_builder.get_observation_result()
        
        # Handle hardware errors during observation
        if observation_result.hardware_error:
            return self._handle_hardware_error(
                observation_result, safety_info, "observation"
            )
        
        # Handle observation failure (non-hardware)
        if observation_result.observation is None:
            return self._handle_observation_failure(safety_info)
        
        observation = observation_result.observation
        self.last_valid_observation = observation
        
        # Update renderer
        if self.renderer is not None:
            cube_y_cm = (observation.cube_y - self.grid.y_min) * 100
            cube_x_cm = (observation.cube_x - self.grid.x_min) * 100
            self.renderer.update_step(self.step_count, cube_y_cm, cube_x_cm)
        
        # Check out of bounds
        x, y = observation.cube_x, observation.cube_y
        out_of_bounds = (
            x < self.grid.x_min or x > self.grid.x_max or
            y < self.grid.y_min or y > self.grid.y_max
        )
        
        # Update grid
        out_of_bound_penalty = 0.0
        if out_of_bounds:
            visits = 0
            print(f"Cube out of bounds at ({x:.3f}, {y:.3f})")
            self.needs_reposition = True
            out_of_bound_penalty = self.out_of_bound_penalty
        else:
            visits = self.grid.visit(x, y)
        
        # Compute reward
        reward_info = self.reward_fn.compute(
            visitation_count=visits,
            motor_currents=observation.motor_currents,
            hardware_error=False,
        )
        reward = reward_info.total + out_of_bound_penalty
        
        # Apply safety penalty
        safety_penalty = self._compute_safety_penalty(safety_info)
        reward += safety_penalty
        
        # Check termination conditions
        coverage = self.grid.coverage()
        max_current = max(abs(float(i)) for i in observation.motor_currents)
        
        terminated = False
        truncated = False
        termination_reason = None
        
        if max_current > self.high_current_threshold:
            terminated = True
            termination_reason = "high_current"
        elif out_of_bounds:
            terminated = True
            termination_reason = "out_of_bounds"
        elif coverage >= self.target_coverage:
            terminated = True
            termination_reason = "coverage"
        elif self.step_count >= self.max_steps:
            truncated = True
            termination_reason = "max_steps"
        
        # Build info dict
        info = {
            "coverage": coverage,
            "visits": visits,
            "coverage_reward": reward_info.coverage_reward,
            "current_reward": reward_info.current_reward,
            "current_change_penalty": reward_info.current_change_penalty,
            "hardware_error_penalty": reward_info.hardware_error_penalty,
            "max_current": max_current,
            "motor_currents": observation.motor_currents.copy(),
            "motor_voltages": self.executor.read_voltages() if hasattr(self.executor, 'read_voltages') else [None]*4,
            "motor_temperatures": self.executor.read_temperatures() if hasattr(self.executor, 'read_temperatures') else [None]*4,
            "motor_currents": observation.motor_currents.copy(),
            "hardware_error": False,
            "hardware_error_ids": [],
            "hardware_error_status": {},
            "execution_success": True,
            "cube_position": (x, y),
            "termination_reason": termination_reason,
            "out_of_bounds_penalty": out_of_bound_penalty,
            "action_modified": safety_info['modified'],
            "safety_reason": safety_info['reason'],
            "safety_detail": safety_info['detail'],
            "safety_penalty": safety_penalty,
            "modification_magnitude": safety_info['magnitude'],
            "tension_penalty": reward_info.tension_penalty,
        }
        
        return observation.as_numpy(), reward, terminated, truncated, info
    
    
    # Helper methods
    
    
    def _apply_safety_filter(self, action):
        """Apply safety filter and return safety info."""
        safety_info = {
            'action': action,
            'modified': False,
            'reason': 'no_filter',
            'detail': '',
            'magnitude': 0.0,
        }
        
        if self.safety_filter is None:
            return safety_info
        
        # Get current motor state
        positions = self.executor.read_positions()
        currents = self.executor.read_currents()
        voltages = self.executor.read_voltages()
        temperatures = self.executor.read_temperatures()
        initial_positions = self.obs_builder.initial_motor_positions
        
        if positions is None or currents is None or initial_positions is None:
            print("WARNING: Missing motor state, skipping safety filter")
            return safety_info
        
        safety_result = self.safety_filter.filter(
            action=action,
            currents=np.array(currents),
            positions=np.array(positions),
            initial_positions=initial_positions,
            voltages=np.array(voltages) if voltages is not None else None,
            temperatures=np.array(temperatures) if temperatures is not None else None,
            action_count=self.step_count,
        )
        
        safety_info['action'] = safety_result.safe_action
        safety_info['modified'] = safety_result.modified
        safety_info['reason'] = safety_result.reason.value if safety_result.reason else "safe"
        safety_info['detail'] = safety_result.detail
        
        if safety_result.modified:
            self.safety_interventions_total += 1
            safety_info['magnitude'] = np.mean(
                np.abs(safety_result.raw_action - safety_result.safe_action)
            )
        
        return safety_info
    
    def _compute_safety_penalty(self, safety_info):
        """Compute safety penalty based on intervention."""
        if not safety_info['modified'] or self.safety_penalty_weight <= 0:
            return 0.0
        
        penalty = -self.safety_penalty_weight * (safety_info['magnitude'])
        
        # Extra penalty for dangerous interventions
        dangerous_keywords = ["over_current", "tension", "opposing"]
        if any(kw in safety_info['reason'] for kw in dangerous_keywords):
            penalty *= 1.2

        penalty = max(-1.0, penalty)
        
        if penalty != 0:
            print(f"  [SAFETY PENALTY] {penalty:.2f} "
                  f"(reason: {safety_info['reason']}, "
                  f"magnitude: {safety_info['magnitude']:.3f})")
        
        return penalty
    
    def _create_zero_observation(self):
        """Create a zero observation for error cases."""
        from reset_policy.environment.observation import Observation
        return Observation(
            cube_x=0.0,
            cube_y=0.0,
            cube_yaw=0.0,
            motor_positions=np.zeros(4, dtype=np.float32),
            motor_currents=np.zeros(4, dtype=np.float32),
            initial_motor_positions=np.zeros(4, dtype=np.float32),
            cube_x_norm=0.0,
            cube_y_norm=0.0,
        )
    
    def _handle_hardware_error(self, result, safety_info, source):
        """
        Handle hardware errors (FATAL - no recovery possible).
        
        This method:
        1. Logs the hardware error details
        2. Computes reward with heavy penalty
        3. Returns terminated=True with hardware_error info
        4. Sets flag so next reset() will raise error
        """
        # Record hardware error for next reset
        self.hardware_error_occurred = True
        self.hardware_error_details = {
            'source': source,
            'error_ids': result.hardware_error_ids,
            'error_status': result.hardware_error_status,
            'message': result.error_message,
        }
        
        print(f"\n{'!'*60}")
        print(f"FATAL HARDWARE ERROR during {source}")
        print(f"Error IDs: {result.hardware_error_ids}")
        print(f"Error status: {result.hardware_error_status}")
        print(f"Message: {result.error_message}")
        print(f"Training cannot continue - no recovery possible")
        print(f"{'!'*60}\n")
        
        # Use last valid observation
        observation = self.last_valid_observation
        if observation is None:
            print("WARNING: No valid observation available, using zeros")
            observation = self._create_zero_observation()
        
        # Compute reward with hardware error penalty
        reward_info = self.reward_fn.compute(
            visitation_count=0,
            motor_currents=observation.motor_currents,
            hardware_error=True,
        )
        reward = reward_info.total
        
        # Apply safety penalty
        safety_penalty = 0.0
        if safety_info['modified'] and self.safety_penalty_weight > 0:
            safety_penalty = self._compute_safety_penalty(safety_info)
            safety_penalty = max(-1.0, safety_penalty * 1.5) 
            reward += safety_penalty
            print(f"  [SAFETY PENALTY ON HARDWARE ERROR] {safety_penalty:.2f}")
        
        info = {
            "coverage": self.grid.coverage(),
            "visits": 0,
            "hardware_error": True,
            "hardware_error_ids": result.hardware_error_ids,
            "hardware_error_status": result.hardware_error_status,
            "execution_success": False if source == "action_execution" else True,
            "execution_error_message": result.error_message,
            "termination_reason": "hardware_error",
            "action_modified": safety_info['modified'],
            "safety_reason": safety_info['reason'],
            "safety_detail": safety_info['detail'],
            "safety_penalty": safety_penalty,
            "modification_magnitude": safety_info['magnitude'],
            "tension_penalty": reward_info.tension_penalty,
        }
        
        return observation.as_numpy(), reward, True, False, info
    
    def _handle_observation_failure(self, safety_info):
        """
        Handle observation failures (non-hardware).
        This could be AprilTag detection failure or communication issue.
        """
        print("WARNING: Observation result is None (non-hardware failure)")
        
        observation = self.last_valid_observation
        if observation is None:
            print("WARNING: No valid observation available, using zeros")
            observation = self._create_zero_observation()
        
        reward_info = self.reward_fn.compute(
            visitation_count=0,
            motor_currents=observation.motor_currents,
            hardware_error=False,
        )
        
        info = {
            "coverage": self.grid.coverage(),
            "visits": 0,
            "coverage_reward": 0.0,
            "current_reward": reward_info.current_reward,
            "current_change_penalty": reward_info.current_change_penalty,
            "hardware_error_penalty": reward_info.hardware_error_penalty,
            "max_current": 0.0,
            "motor_currents": np.zeros(4),
            "hardware_error": False,
            "hardware_error_ids": [],
            "hardware_error_status": {},
            "execution_success": True,
            "cube_position": (observation.cube_x, observation.cube_y),
            "termination_reason": "observation_failed",
            "action_modified": safety_info['modified'],
            "safety_reason": safety_info['reason'],
            "safety_detail": safety_info['detail'],
            "safety_penalty": 0.0,
            "modification_magnitude": safety_info['magnitude'],
            "tension_penalty": reward_info.tension_penalty,
        }
        
        return observation.as_numpy(), reward_info.total, True, False, info
    
    
    # Repositioning
    
    
    def reposition_cube_to_center(self, max_attempts=3):
        """Reposition cube to center using deterministic motor pulls."""
        print("\n========== REPOSITIONING CUBE ==========")
        
        for attempt in range(max_attempts):
            # Get current position
            result = self.obs_builder.get_observation_result()
            if result.observation is None:
                print("ERROR: Cannot get cube position")
                return False
            
            x, y = result.observation.cube_x, result.observation.cube_y
            print(f"Current position: ({x:.3f}, {y:.3f})")
            print(f"Board bounds: x=[{self.grid.x_min:.3f}, {self.grid.x_max:.3f}], "
                  f"y=[{self.grid.y_min:.3f}, {self.grid.y_max:.3f}]")
            
            # Check if already in bounds
            in_bounds = (self.grid.x_min <= x <= self.grid.x_max and 
                        self.grid.y_min <= y <= self.grid.y_max)
            
            if in_bounds and attempt > 0:
                print("Cube is in bounds, repositioning complete")
                break
            
            # Determine motor to pull
            motor_to_pull = self._determine_motor_to_pull(x, y)
            if motor_to_pull is None:
                print("Cube is centered, no repositioning needed")
                break
            
            # Execute repositioning
            self._pull_motor_to_reposition(motor_to_pull)
            
            # Restore tension
            self.restore_tension_after_reposition()
            
            # Sync targets
            self._sync_targets_with_actual_positions()
        
        print("========== REPOSITION COMPLETE ==========")
        return True
    
    def _determine_motor_to_pull(self, x, y):
        """Determine which motor to pull based on position."""
        center_x = (self.grid.x_min + self.grid.x_max) / 2
        center_y = (self.grid.y_min + self.grid.y_max) / 2
        dx = center_x - x
        dy = center_y - y
        
        # Priority: whichever is further from center
        if abs(dx) > abs(dy):
            if dx > 0:
                print(f"Pulling Motor 18 to move +X (up)")
                return 18
            else:
                print(f"Pulling Motor 19 to move -X (down)")
                return 19
        else:
            if dy > 0:
                print(f"Pulling Motor 17 to move +Y (right)")
                return 17
            else:
                print(f"Pulling Motor 16 to move -Y (left)")
                return 16
    
    def _pull_motor_to_reposition(self, motor_to_pull, steps=10, step_delta=50):
        """Execute repositioning pull."""
        # Release other motors
        print(f"Releasing other motors (except Motor {motor_to_pull})...")
        for motor in self.executor.motor_ids:
            if motor != motor_to_pull:
                try:
                    self.executor.write1(motor, 64, 0)  # TORQUE_DISABLE
                    print(f"  Released Motor {motor}")
                except Exception as e:
                    print(f"  Failed to release motor {motor}: {e}")
        
        time.sleep(0.05)
        
        # Pull motor in steps
        print(f"Moving Motor {motor_to_pull}: {steps} steps of {step_delta} ticks")
        for step in range(steps):
            self.executor.move_motor_by_delta(motor_to_pull, step_delta)
            time.sleep(0.05)
        
        # Re-enable motors
        print("Re-enabling all motors...")
        for motor in self.executor.motor_ids:
            try:
                self.executor.write1(motor, 64, 1)
            except Exception as e:
                print(f"  Failed to enable motor {motor}: {e}")
        
        time.sleep(0.5)
    
    
    # Tension Restoration
    def restore_tension_after_reposition(self, target_tension=10.0, max_steps=5):
        """Gradually restore tension after repositioning."""
        print("\n========== RESTORING TENSION ==========")
        print(f"Target tension: {target_tension} mA")
        
        currents = self.executor.read_currents()
        if currents is None:
            print("WARNING: Cannot read currents for tension restoration")
            return False
        
        # Calculate tensions using correct pairs
        h_idx = config.get_motor_indices(self.executor.motor_ids)['horizontal']
        v_idx = config.get_motor_indices(self.executor.motor_ids)['vertical']
        
        horizontal_tension = abs(currents[h_idx[0]]) + abs(currents[h_idx[1]])
        vertical_tension = abs(currents[v_idx[0]]) + abs(currents[v_idx[1]])
        
        print(f"Current horizontal tension: {horizontal_tension:.1f} mA")
        print(f"Current vertical tension: {vertical_tension:.1f} mA")
        
        if (horizontal_tension >= target_tension * 0.7 and 
            vertical_tension >= target_tension * 0.7):
            print("Tension already sufficient")
            return True
        
        step_delta = 30
        wait_time = 0.5
        
        for step in range(max_steps):
            currents = self.executor.read_currents()
            if currents is None:
                break
            
            horizontal_tension = abs(currents[h_idx[0]]) + abs(currents[h_idx[1]])
            vertical_tension = abs(currents[v_idx[0]]) + abs(currents[v_idx[1]])
            
            if (horizontal_tension >= target_tension * 0.8 and 
                vertical_tension >= target_tension * 0.8):
                print(f"  Target tension reached at step {step+1}")
                break
            
            # Tighten horizontal pair (16, 17)
            if horizontal_tension < target_tension * 0.8:
                # Pull both motors to create tension
                self.executor.move_motor_by_delta(16, step_delta)
                self.executor.move_motor_by_delta(17, step_delta)
                print(f"  Step {step+1}: Tightening horizontal")
            
            # Tighten vertical pair (18, 19)
            if vertical_tension < target_tension * 0.8:
                # Pull both motors to create tension
                self.executor.move_motor_by_delta(18, step_delta)
                self.executor.move_motor_by_delta(19, step_delta)
                print(f"  Step {step+1}: Tightening vertical")
            
            time.sleep(wait_time)
        
        print("========== TENSION RESTORATION COMPLETE ==========")
        return True
    
    
    # Utility
    
    
    def _sync_targets_with_actual_positions(self):
        """Sync executor targets with actual motor positions."""
        print("  Syncing targets with actual positions...")
        positions = self.executor.read_positions()
        if positions is not None:
            for motor_id, pos in zip(self.executor.motor_ids, positions):
                old_target = self.executor.targets.get(motor_id, pos)
                self.executor.targets[motor_id] = pos
                print(f"    Motor {motor_id}: target {old_target} → {pos}")
        else:
            print("  WARNING: Could not read positions for sync!")
    
    # Rendering
    def render(self):
        """Render current state."""
        if self.renderer is None:
            return None
        
        observation = self.obs_builder.get_observation()
        if observation is None:
            print("[RENDER] Warning: No observation available")
            return None
        
        cube_x_cm = (observation.cube_x - self.grid.x_min) * 100
        cube_y_cm = (observation.cube_y - self.grid.y_min) * 100
        
        return self.renderer.render(
            cube_y=cube_y_cm,
            cube_x=cube_x_cm,
            occupancy_grid=self.grid.as_numpy(),
            coverage=self.grid.coverage(),
            mode=self.render_mode,
        )
    
    def save_render(self):
        """Save current render to disk."""
        if self.renderer is None:
            return None
        
        observation = self.obs_builder.get_observation()
        if observation is None:
            return None
        
        cube_x_cm = (observation.cube_x - self.grid.x_min) * 100
        cube_y_cm = (observation.cube_y - self.grid.y_min) * 100
        
        return self.renderer.render(
            cube_y=cube_y_cm,
            cube_x=cube_x_cm,
            occupancy_grid=self.grid.as_numpy(),
            coverage=self.grid.coverage(),
            mode="human",
        )
    
    def close(self):
        """Close environment and cleanup."""
        if hasattr(self, 'executor') and self.executor is not None:
            try:
                if not hasattr(self.executor, '_shutdown_done'):
                    self.executor.shutdown()
                    self.executor._shutdown_done = True
            except Exception as e:
                print(f"Executor shutdown error: {e}")
        
        if self.renderer is not None:
            self.renderer.close()
        
        print("Environment closed")