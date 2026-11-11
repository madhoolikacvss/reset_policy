"""Run inside the existing reset conda environment: python -B main_sac.py."""
import sys
sys.dont_write_bytecode = True
import argparse
from datetime import datetime
from config import TrainingConfig
from train_sac import train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--max-steps', type=int, default=400)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--warmup-steps', type=int, default=5000)
    parser.add_argument('--run-name', default=datetime.now().strftime('sac_sim_%Y%m%d_%H%M%S'))
    args = parser.parse_args()
    if args.episodes < 1 or args.max_steps < 1 or args.warmup_steps < 0:
        parser.error('Episodes/steps must be positive and warmup nonnegative.')
    if not args.run_name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.run_name):
        parser.error('Run name must contain only letters, digits, underscores or hyphens.')
    cfg = TrainingConfig(episodes=args.episodes, max_steps=args.max_steps, seed=args.seed,
                         warmup_steps=args.warmup_steps)
    train(cfg, args.run_name)

if __name__ == '__main__':
    main()
