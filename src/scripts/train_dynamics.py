"""
PETS-style 1-step forward dynamics model for the cube-string system.

Inputs:  [cube_x, cube_y, action_m16, action_m17, action_m18, action_m19]
Outputs: [delta_cube_x, delta_cube_y]

Model:   ensemble of 5 MLPs (3 layers, 200 units, Swish)
Loss:    MSE on delta

Validation:
    - Split by episode (not by step) to avoid leakage
    - Report MSE per motor + goal-directedness check

Usage:
    python scripts/train_dynamics_model.py
    python scripts/train_dynamics_model.py --epochs 100 --ensemble 5
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ---------------- Config ----------------

MOTOR_LOG_DIR = Path(
    "/home/madhoolika/workspace/reset_policy/src/logs/motor_logs"
)
OUT_DIR = Path(
    "/home/madhoolika/workspace/reset_policy/src/logs/dynamics_model"
)

INPUT_COLS = [
    "cube_x", "cube_y",
    "action_m16", "action_m17", "action_m18", "action_m19",
]
OUTPUT_COLS = ["cube_x", "cube_y"]   # we predict the delta of these

VAL_FRACTION = 0.10          # 10% of episodes held out
SEED = 0


# ---------------- Data loading ----------------

def load_all_episodes(log_dir: Path, verbose: bool = True):
    """
    Load all episodes, build (s_t, a_t) -> delta_s_t pairs.
    Returns:
        X: (N, 6) float32
        Y: (N, 2) float32  (delta of cube_x, cube_y)
        episode_ids: (N,) int  (which episode each row came from)
    """
    files = sorted(log_dir.glob("episode_*_motors.csv"))
    if not files:
        raise FileNotFoundError(f"No episodes found in {log_dir}")

    Xs, Ys, eps = [], [], []

    for ep_idx, f in enumerate(files):
        df = pd.read_csv(f)

        # Need at least 2 rows to make one transition
        if len(df) < 2:
            continue

        # Check required columns
        missing = [c for c in INPUT_COLS + OUTPUT_COLS if c not in df.columns]
        if missing:
            print(f"WARNING: {f.name} missing columns {missing}, skipping")
            continue

        # Build pairs: row t (input state + action) -> row t+1 (target cube pos)
        for t in range(len(df) - 1):
            s_t = df.iloc[t][INPUT_COLS].to_numpy(dtype=np.float32)
            cube_next = df.iloc[t + 1][OUTPUT_COLS].to_numpy(dtype=np.float32)
            cube_t = df.iloc[t][OUTPUT_COLS].to_numpy(dtype=np.float32)
            delta = cube_next - cube_t

            # Skip rows with NaN
            if not (np.isfinite(s_t).all() and np.isfinite(delta).all()):
                continue

            Xs.append(s_t)
            Ys.append(delta)
            eps.append(ep_idx)

    X = np.stack(Xs, axis=0)
    Y = np.stack(Ys, axis=0)
    episode_ids = np.array(eps, dtype=np.int64)

    if verbose:
        print(f"Loaded {len(files)} files, "
              f"{len(X)} transitions, {X.shape[1]}-dim inputs")
        print(f"  X range: [{X.min(axis=0)}, {X.max(axis=0)}]")
        print(f"  Y range: [{Y.min(axis=0)}, {Y.max(axis=0)}]")

    return X, Y, episode_ids


def train_val_split(X, Y, episode_ids, val_fraction, seed):
    """Split by episode to avoid leakage."""
    rng = np.random.default_rng(seed)
    unique_eps = np.unique(episode_ids)
    n_val = max(1, int(len(unique_eps) * val_fraction))
    val_eps = set(rng.choice(unique_eps, size=n_val, replace=False))

    val_mask = np.isin(episode_ids, list(val_eps))
    train_mask = ~val_mask

    return (
        X[train_mask], Y[train_mask],
        X[val_mask], Y[val_mask],
        sorted(val_eps),
    )


# ---------------- Model ----------------

class MLP(nn.Module):
    """One PETS ensemble member (deterministic, 3 layers)."""

    def __init__(self, in_dim, out_dim, hidden=200, depth=3):
        super().__init__()
        layers = []
        prev = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.SiLU())  # Swish
            prev = hidden
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Ensemble(nn.Module):
    """Ensemble of MLPs — PETS style."""

    def __init__(self, in_dim, out_dim, n_members=5, hidden=200, depth=3):
        super().__init__()
        self.members = nn.ModuleList([
            MLP(in_dim, out_dim, hidden, depth)
            for _ in range(n_members)
        ])

    def forward(self, x):
        """
        Returns: (n_members, batch, out_dim) predictions.
        """
        return torch.stack([m(x) for m in self.members], dim=0)

    def predict_mean(self, x):
        """Mean prediction across ensemble members."""
        preds = self.forward(x)
        return preds.mean(dim=0)

    def predict_std(self, x):
        """Disagreement (std) across ensemble members."""
        preds = self.forward(x)
        return preds.std(dim=0)


# ---------------- Training ----------------

def train_one_epoch(model, optimizer, loader, device):
    model.train()
    total_loss = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        preds = model(xb)                # (n_members, batch, out_dim)
        loss = ((preds - yb.unsqueeze(0)) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        n += xb.size(0)

    return total_loss / n


@torch.no_grad()
def evaluate(model, X, Y, device):
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    Y_t = torch.tensor(Y, dtype=torch.float32, device=device)

    # Split into chunks to avoid OOM
    batch = 4096
    preds_all = []
    for i in range(0, len(X_t), batch):
        p = model.predict_mean(X_t[i:i+batch])
        preds_all.append(p.cpu().numpy())
    pred = np.concatenate(preds_all, axis=0)

    mse = ((pred - Y) ** 2).mean(axis=0)
    mae = np.abs(pred - Y).mean(axis=0)
    return mse, mae, pred


# ---------------- Goal-directedness check ----------------

def goal_directedness_report(X_val, Y_val, pred_val):
    """
    X_val columns: [cube_x, cube_y, a16..19]
    Y_val columns: [delta_x, delta_y]

    We don't have the goal in X_val, so we infer it from the episode-level
    behavior. For now, we just report:
        - Fraction of steps where |pred_delta| moves cube "inward"
          (proxy: reduce the magnitude of cube position)
        - Fraction of steps where ground truth does the same

    The real goal-directedness check requires the goal, which is loaded
    separately in `check_goal_direction`.
    """
    pred_delta = pred_val
    true_delta = Y_val

    # Approximate "moved toward center" by checking the sign change of
    # cube_x * delta_x (moving toward x=0)
    cube_x = X_val[:, 0]
    cube_y = X_val[:, 1]

    pred_toward_x = (cube_x * pred_delta[:, 0]) < 0
    true_toward_x = (cube_x * true_delta[:, 0]) < 0
    pred_toward_y = (cube_y * pred_delta[:, 1]) < 0
    true_toward_y = (cube_y * true_delta[:, 1]) < 0

    print(f"\nGoal-directedness proxy (toward origin):")
    print(f"  X-axis  — model: {pred_toward_x.mean():.2%}, "
          f"ground truth: {true_toward_x.mean():.2%}")
    print(f"  Y-axis  — model: {pred_toward_y.mean():.2%}, "
          f"ground truth: {true_toward_y.mean():.2%}")


def check_goal_direction(X_val_raw, Y_val, pred_val, episode_ids_val, log_dir):
    """
    Load each episode's goal from episodes.csv, then compute whether the
    model's predicted next cube position is closer to the goal than the
    current cube position.

    X_val_raw columns: [cube_x, cube_y, a16, a17, a18, a19]
    episode_ids_val: array of episode indices (0-based, matching sorted file order)
    """
    episodes_csv = log_dir.parent / "episodes.csv"
    if not episodes_csv.exists():
        print(f"\n[goal check] episodes.csv not found at {episodes_csv}")
        return

    # Load episodes.csv and dedupe by episode number (keep last occurrence)
    ep_df = pd.read_csv(episodes_csv)
    ep_df = ep_df.drop_duplicates(subset="episode", keep="last")
    ep_df = ep_df.set_index("episode")

    # Build a mapping from episode-index -> goal
    # We need to know which actual episode number each val-index corresponds to.
    # The sorted file order in motor_logs is episode_0001, episode_0002, ...
    # But your files might not be contiguous. Rebuild the file list.
    motor_files = sorted(log_dir.glob("episode_*_motors.csv"))

    counts = {"closer": 0, "farther": 0, "equal": 0, "skipped": 0}

    for i in range(len(X_val_raw)):
        # episode_ids_val[i] is the file index (0-based) among sorted files
        file_idx = int(episode_ids_val[i])
        if file_idx >= len(motor_files):
            counts["skipped"] += 1
            continue

        # Extract the episode number from the filename
        fname = motor_files[file_idx].stem  # e.g. "episode_0001_motors"
        try:
            ep_num = int(fname.split("_")[1])
        except (IndexError, ValueError):
            counts["skipped"] += 1
            continue

        if ep_num not in ep_df.index:
            counts["skipped"] += 1
            continue

        row = ep_df.loc[ep_num]
        goal_x = float(row["goal_x"])
        goal_y = float(row["goal_y"])

        if not np.isfinite(goal_x) or not np.isfinite(goal_y):
            counts["skipped"] += 1
            continue

        cx = float(X_val_raw[i, 0])
        cy = float(X_val_raw[i, 1])
        dx = float(pred_val[i, 0])
        dy = float(pred_val[i, 1])

        d_before = np.hypot(cx - goal_x, cy - goal_y)
        d_after = np.hypot(cx + dx - goal_x, cy + dy - goal_y)

                # Model prediction
        d_after_pred = np.hypot(cx + dx - goal_x, cy + dy - goal_y)

        # Ground truth next position
        cx_next = cx + float(Y_val[i, 0])
        cy_next = cy + float(Y_val[i, 1])
        d_after_true = np.hypot(cx_next - goal_x, cy_next - goal_y)

        # Count for both
        if d_after_pred < d_before - 1e-9:
            counts["closer"] += 1
        elif d_after_pred > d_before + 1e-9:
            counts["farther"] += 1
        else:
            counts["equal"] += 1

        if d_after_true < d_before - 1e-9:
            counts["true_closer"] += 1
        elif d_after_true > d_before + 1e-9:
            counts["true_farther"] += 1
        else:
            counts["true_equal"] += 1


    total = counts["closer"] + counts["farther"] + counts["equal"]
    if total == 0:
        print("\n[goal check] no valid rows")
        print(f"  skipped: {counts['skipped']}")
        return

    print(f"\nGoal-directedness (using episodes.csv goals):")
    print(f"  Predicted step moves closer to goal: "
          f"{counts['closer'] / total:.2%}")
    print(f"  Predicted step moves farther from goal: "
          f"{counts['farther'] / total:.2%}")
    print(f"  Predicted step moves same distance: "
          f"{counts['equal'] / total:.2%}")
    print(f"  Skipped: {counts['skipped']} rows")

# ---------------- Main ----------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--ensemble", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=200)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Data ----
    X, Y, ep_ids = load_all_episodes(MOTOR_LOG_DIR)
    X_tr, Y_tr, X_val, Y_val, val_eps = train_val_split(
        X, Y, ep_ids, args.val_fraction, args.seed
    )
    print(f"Train: {len(X_tr)} samples  "
          f"Val: {len(X_val)} samples (episodes: {val_eps})")

    # ---- Normalization (compute on train only) ----
    X_mean = X_tr.mean(axis=0)
    X_std = X_tr.std(axis=0) + 1e-8
    Y_mean = Y_tr.mean(axis=0)
    Y_std = Y_tr.std(axis=0) + 1e-8

    X_tr_n = (X_tr - X_mean) / X_std
    X_val_n = (X_val - X_mean) / X_std
    Y_tr_n = (Y_tr - Y_mean) / Y_std
    Y_val_n = (Y_val - Y_mean) / Y_std

    # ---- DataLoaders ----
    train_ds = TensorDataset(
        torch.tensor(X_tr_n, dtype=torch.float32),
        torch.tensor(Y_tr_n, dtype=torch.float32),
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, drop_last=True,
    )

    # ---- Model ----
    model = Ensemble(
        in_dim=X.shape[1], out_dim=Y.shape[1],
        n_members=args.ensemble,
        hidden=args.hidden, depth=args.depth,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Ensemble of {args.ensemble} MLPs, "
          f"{n_params:,} params total")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ---- Train ----
    t0 = time.time()
    best_val_mse = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, optimizer, train_loader, device)
        val_mse, val_mae, pred_val_n = evaluate(model, X_val_n, Y_val_n, device)
        val_mse_mean = float(val_mse.mean())

        if val_mse_mean < best_val_mse:
            best_val_mse = val_mse_mean
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:4d}  "
                  f"train MSE (norm): {tr_loss:.5f}  "
                  f"val MSE (norm): {val_mse_mean:.5f}  "
                  f"val MAE (norm): {val_mae.mean():.5f}")

    elapsed = time.time() - t0
    print(f"\nTraining done in {elapsed:.1f}s")
    print(f"Best val MSE (norm): {best_val_mse:.5f}")

    # ---- Restore best ----
    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- Evaluate in physical units ----
    val_mse, val_mae, pred_val_n = evaluate(model, X_val_n, Y_val_n, device)
    pred_val = pred_val_n * Y_std + Y_mean

    print(f"\nValidation MSE (physical units):")
    print(f"  Δcube_x: {val_mse[0]:.8f} m²  "
          f"(RMSE {np.sqrt(val_mse[0]) * 1000:.3f} mm)")
    print(f"  Δcube_y: {val_mse[1]:.8f} m²  "
          f"(RMSE {np.sqrt(val_mse[1]) * 1000:.3f} mm)")

    print(f"Validation MAE (physical units):")
    print(f"  Δcube_x: {val_mae[0] * 1000:.3f} mm")
    print(f"  Δcube_y: {val_mae[1] * 1000:.3f} mm")

    # Reference: how big are the deltas?
    print(f"\nDelta magnitude reference:")
    print(f"  Mean |Δcube_x|: {np.abs(Y_val[:, 0]).mean() * 1000:.3f} mm")
    print(f"  Mean |Δcube_y|: {np.abs(Y_val[:, 1]).mean() * 1000:.3f} mm")

    # ---- Goal-directedness ----
    goal_directedness_report(X_val, Y_val, pred_val)

    # Also try the episodes.csv-based check
    # Rebuild the episode_ids for the val set
    X_all_eps = ep_ids[np.isin(ep_ids, val_eps)]
    # X_val_raw = original unscaled X_val
    check_goal_direction(X_val, Y_val, pred_val, X_all_eps, MOTOR_LOG_DIR)

    # ---- Save ----
    torch.save({
        "model_state_dict": model.state_dict(),
        "X_mean": X_mean, "X_std": X_std,
        "Y_mean": Y_mean, "Y_std": Y_std,
        "input_cols": INPUT_COLS,
        "output_cols": OUTPUT_COLS,
        "val_episodes": val_eps,
    }, OUT_DIR / "dynamics_model.pt")
    print(f"\nSaved model to {OUT_DIR / 'dynamics_model.pt'}")


if __name__ == "__main__":
    main()