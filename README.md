# Reset Policy for Cube-String System

## Project Goal

**The Problem**: Robotic manipulation systems often fail during operation and require manual intervention to reset. For cable-driven systems (like our cube on 4 strings), a harsh pull or unexpected disturbance can leave the system in an unrecoverable state, requiring a human to physically reposition the object.

**Our Solution**: We train a reinforcement learning (PPO) policy to **autonomously reset** a cube to the board center from any arbitrary position. Reset policy learns to control 4 Dynamixel motors that manipulate strings to reposition the cube, essentially teaching the robot how to recover from failures without human intervention.

**Why This Matters**:
- **Autonomous Recovery**: Removes the need for human intervention when a manipulation task fails
- **Cable-Driven Systems**: Addresses the challenging dynamics of string-based manipulation
- **Safe Exploration**: The policy must learn to explore while respecting hardware constraints (current, voltage, temperature limits)
- **Real-World RL**: Trains on physical hardware with real sensor feedback, not simulation

The system uses AprilTag tracking for cube position and motor telemetry for state estimation, demonstrating that RL can learn safe manipulation policies directly on hardware.

## System Overview

- **4 Dynamixel XL330 motors** control the cube via strings
- **Coordinate System**: Y-axis = horizontal (left/right), X-axis = vertical (up/down)
- **Motor Mapping**:
  - Motors 16, 17: horizontal pair (Y-axis movement)
  - Motors 18, 19: vertical pair (X-axis movement)
  - Action: +1 = pull (shorten string), -1 = release (lengthen string)
- **Perception**: AprilTag tracking (world tag ID 0, cube tag ID 1)
- **Safety Filter**: 7-layer protection (current, voltage, temperature, tension, position, stuck motor, opposing motors)

## MDP Formulation

| Component | Description |
|-----------|-------------|
| **State** | 14-dim: cube position (x, y, yaw), motor position deltas (4), motor currents (4), tension metrics (3) |
| **Action** | 4-dim continuous [-1, 1], converted to encoder deltas (±150 ticks) |
| **Reward** | Coverage (+1 new cell), Current (+0.5 safe, -1.0 high), Current change (-0.3), Tension (-0.3), Safety penalty (tiered), Hardware error (-1.0 terminal) |
| **Termination** | High current (>2500mA), out-of-bounds, coverage ≥95%, max_steps (200) |
| **Truncation** | max_steps reached |

## Hardware Safety

Hardware errors are **FATAL** - motors cannot be recovered without physical restart. The safety filter is the primary defense:
- Current-aware scaling (proactive)
- Voltage monitoring (prevents communication loss)
- Temperature protection
- Tension constraint (prevents slack strings)
- Stuck motor detection
- Opposing motor prevention

## Installation

```bash
cd /home/.../reset_policy
pip install -r requirements.txt
```

## Training 
1. From scratch:
   ```
    cd /home/.../reset_policy/src/scripts
    nohup python main.py --episodes 1000 --save_every 5 --render_every 5 > training.log 2>&1 &
   ```
2. Resume from checkpoint
   ```
   cd /home/.../reset_policy/src/scripts
   nohup python main.py --resume /home/.../reset_policy/checkpoints/ppo_checkpoint_70.pth --episodes 1000 --save_every 5 > training_resume.log 2>&1 &
   ```
