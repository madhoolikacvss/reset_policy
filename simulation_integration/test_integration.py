import sys
sys.dont_write_bytecode = True
import unittest
import numpy as np
import torch
from config import TrainingConfig
from environment import SimulatorAdapter, ResetPolicyEnv, core, distance_reward, normalize_goal
from rl.replay_buffer import ReplayBuffer
from rl.her import HERBuffer
from rl.sac import SACAgent, SACConfig

class IntegrationTests(unittest.TestCase):
    def test_online_matches_original_open_loop(self):
        rng = np.random.default_rng(12)
        actions = rng.uniform(-1, 1, (120, 4))
        actions[:30] = [1, -1, -1, -1]
        for lag in (0, 2):
            sim = SimulatorAdapter()
            sim.p.motor_lag_rows = lag
            start = np.tile([0.5, 0.3], (len(actions), 1))
            ep = core.Episode('test', start, np.zeros_like(start), actions, np.zeros_like(actions))
            expected = core.simulate_episode(ep, sim.p)
            sim.reset(start[0])
            got = np.array([sim.step(a)[0] for a in actions])
            np.testing.assert_allclose(got, expected.pred_norm, atol=1e-12, rtol=0)
            self.assertEqual(sim.off_board_rows, expected.off_board_rows)

    def test_success_and_timeout(self):
        env = ResetPolicyEnv(TrainingConfig(max_steps=1))
        goal = [-0.285, -0.415]
        env.reset(options={'goal': goal, 'start_norm': normalize_goal(goal)})
        _, reward, terminated, truncated, _ = env.step(np.zeros(4))
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertAlmostEqual(reward, 2.)
        env.reset(options={'goal': goal, 'start_norm': [0.2, 0.2]})
        _, _, terminated, truncated, _ = env.step(np.zeros(4))
        self.assertFalse(terminated)
        self.assertTrue(truncated)

    def test_her_reward_goal_and_terminal_consistency(self):
        cfg = TrainingConfig()
        base = ReplayBuffer(state_dim=4)
        her = HERBuffer(base, cfg)
        for i, goal in enumerate(([-0.30, -0.40], [-0.30, -0.40], [-0.20, -0.40])):
            her.add(state=np.array([.5,.5,.7,.7]), next_state=np.array([.4,.4,.7,.7]),
                    action=np.zeros(4), executed_action=np.zeros(4), reward=0.,
                    done=float(i==2), achieved_goal=np.array(goal), desired_goal=np.array([-.1,-.2]))
        self.assertEqual(her.finalize_episode(), 8)
        for i in range(3, len(base)):
            reward, terminal = distance_reward(base._achieved_goals[i], base._desired_goals[i], cfg)
            self.assertAlmostEqual(base._rewards[i], reward, places=6)
            self.assertEqual(base._dones[i], float(terminal))
            np.testing.assert_allclose(base._states[i][2:], normalize_goal(base._desired_goals[i]), atol=1e-7)
            np.testing.assert_allclose(base._next_states[i][2:], base._states[i][2:])
        self.assertEqual(her.finalize_episode(), 0)
        torch.set_num_threads(1)
        agent = SACAgent(SACConfig(obs_dim=4), torch.device('cpu'))
        for _ in range(4):
            metrics = agent.update(base.sample(8))
            self.assertTrue(np.isfinite(metrics['q1_loss']))

if __name__ == '__main__':
    unittest.main(verbosity=2)
