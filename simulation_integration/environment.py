"""Stateful online adapter for the existing simulator, with no hardware imports."""
import sys
from pathlib import Path
import numpy as np
import mujoco
from config import SIMULATOR, GOALS, TrainingConfig

# Suppress bytecode writes in the read-only source simulator as well as imports.
sys.dont_write_bytecode = True
sys.path.insert(0, str(SIMULATOR))
import sim_core as core
from params import load_params

def normalize_goal(raw):
    return core.raw_to_norm(*np.asarray(raw)).astype(np.float32)

def distance_reward(achieved, desired, cfg):
    distance = float(np.linalg.norm(np.asarray(achieved) - np.asarray(desired)))
    success = distance < cfg.success_threshold
    # Match actual hardware env: exponential distance plus distance-based success bonus.
    return float(np.exp(-cfg.distance_scale * distance)) + float(success), success

class SimulatorAdapter:
    """Same MuJoCo substep sequence as sim_core._simulate_force, kept alive per episode."""
    def __init__(self, params_path=None):
        self.p = load_params(params_path or SIMULATOR / 'params.yaml')
        if self.p.force_mode == 'current':
            raise ValueError('Current-driven physics needs hardware observations; use the calibrated saturating model.')
        self.model = mujoco.MjModel.from_xml_string(core._make_xml(self.p))
        self.model.dof_damping[:2] = self.p.damping
        self.data = mujoco.MjData(self.model)
        self.bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'cube')
        self.scales = np.array([getattr(self.p, 'force_scale_' + m) for m in core.MOTORS])
        self.substeps = max(1, int(round(self.p.row_dt / core.PHYSICS_DT)))
        self.pull_steps = max(1, int(round(self.p.pull_fraction * self.substeps)))

    def reset(self, start_norm):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = [*core.norm_to_board(start_norm), 0.0]
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.lengths = core._string_lengths(self.data.qpos[:2], self.p) + self.p.slack_initial
        self.history = []
        self.off_board_rows = 0

    @property
    def position_norm(self):
        return core.board_to_norm(self.data.qpos[:2]).copy()

    def step(self, action):
        self.history.append(np.asarray(action, dtype=float).copy())
        # Only lag+1 past/current actions are needed; never read recorded future states.
        actions = np.asarray(self.history[-(self.p.motor_lag_rows + 1):])
        dummy = np.zeros((len(actions), 2))
        ep = core.Episode('online', dummy, dummy, actions, np.zeros_like(actions))
        activation = core.compute_activation(ep, self.p)[-1]
        hit = False
        for s in range(self.substeps):
            if self.p.string_model == 'slack':
                f4 = core._slack_forces(action, activation, self.data.qpos[:2],
                                       self.lengths, self.scales, self.p, s < self.pull_steps)
            elif s < self.pull_steps:
                f4 = core._motor_forces(activation, self.data.qpos[:2], self.scales, self.p)
            else:
                f4 = np.zeros((4, 2))
            self.data.xfrc_applied[self.bid, :] = 0.0
            self.data.xfrc_applied[self.bid, :2] = f4.sum(axis=0)
            mujoco.mj_step(self.model, self.data)
            for ax, half in ((0, core.HALF_X), (1, core.HALF_Y)):
                if abs(self.data.qpos[ax]) > half:
                    self.data.qpos[ax] = np.sign(self.data.qpos[ax]) * half
                    self.data.qvel[ax] = 0.0
                    hit = True
        self.off_board_rows += int(hit)
        return self.position_norm, hit

class ResetPolicyEnv:
    def __init__(self, cfg=None, params_path=None):
        self.cfg = cfg or TrainingConfig()
        self.sim = SimulatorAdapter(params_path)
        self.rng = np.random.default_rng(self.cfg.seed)

    def observation(self):
        return np.concatenate((self.sim.position_norm, normalize_goal(self.goal_position))).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = options or {}
        episode = options.get('episode_num', 1)
        self.goal_position = np.asarray(options.get('goal', GOALS[(episode - 1) % len(GOALS)]), dtype=float)
        # Independent interior starts; no logged trajectories or future observations.
        self.sim.reset(options.get('start_norm', self.rng.uniform(0.15, 0.85, 2)))
        self.step_count = 0
        return self.observation(), {'goal_position': self.goal_position.copy()}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (4,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
            raise ValueError('Expected four finite actions in [-1, 1].')
        position, hit = self.sim.step(action)
        self.step_count += 1
        raw = np.array([core.B.x_min, core.B.y_min]) + position * np.array([2*core.HALF_X, 2*core.HALF_Y])
        reward, success = distance_reward(raw, self.goal_position, self.cfg)
        truncated = self.step_count >= self.cfg.max_steps and not success
        info = dict(cube_position=raw, goal_position=self.goal_position.copy(),
                    distance_to_goal=float(np.linalg.norm(raw-self.goal_position)),
                    goal_reached=success, total_reward=reward,
                    distance_reward=reward-float(success), executed_actions=action.copy(),
                    current_change_penalty=0., hardware_error_penalty=0., tension_penalty=0.,
                    safety_filtering_penalty=0., action_modified=False, hardware_error=False,
                    off_board=hit, termination_reason='goal_reached' if success else 'max_steps' if truncated else '')
        return self.observation(), reward, success, truncated, info
