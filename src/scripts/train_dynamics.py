"""
PETS-style 1-step forward dynamics model for the cube-string system.

Inputs:  [cube_x, cube_y, action_m16, action_m17, action_m18, action_m19]
Outputs: [delta_cube_x, delta_cube_y]

Model:   ensemble of 5 MLPs (3 layers, 200 units, Swish)
Loss:    MSE on delta

Validation:
    - Split by episode (not by step) to avoid leakage
    - Report RMSE in physical units + goal-directedness check
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
OUTPUT_COLS = ["cube_x", "cube_y"]

VAL_FRACTION = 0.10
SEED = 0


# ---------------- Data loading ----------------

def load_all_episodes(log_dir: Path, verbose: bool = True):
    """Build (s_t, a_t) -> delta_s_t pairs from all episode CSVs."""
    files = sorted(log_dir.glob("episode_*_motors.csv"))
    if not files:
        raise FileNotFoundError(f"No episodes found in {log_dir}")

    Xs, Ys, eps = [], [], []

    for ep_idx, f in enumerate(files):
        df = pd.read_csv(f)
        if len(df) < 2:
            continue

        missing = [c for c in INPUT_COLS + OUTPUT_COLS if c not in df.columns]
        if missing:
            print(f"WARNING: {f.name} missing columns {missing}, skipping")
            continue

        for t in range(len(df) - 1):
            s_t = df.iloc[t][INPUT_COLS].to_numpy(dtype=np.float32)
            cube_next = df.iloc[t + 1][OUTPUT_COLS].to_numpy(dtype=np.float32)
            cube_t = df.iloc[t][OUTPUT_COLS].to_numpy(dtype=np.float32)
            delta = cube_next - cube_t

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
            layers.append(nn.SiLU())
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
        return torch.stack([m(x) for m in self.members], dim=0)

    def predict_mean(self, x):
        preds = self.forward(x)
        return preds.mean(dim=0)

    def predict_std(self, x):
        preds = self.forward(x)
        return preds.std(dim=0)


# ---------------- Training / Evaluation ----------------

def train_one_epoch(model, optimizer, loader, device):
    model.train()
    total_loss = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        preds = model(xb)                      # (n_members, batch, out_dim)
        loss = ((preds - yb.unsqueeze(0)) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        n += xb.size(0)

    return total_loss / n


@torch.no_grad()
def evaluate(model, X, Y, device):
    """Returns MSE, MAE (in whatever units X/Y are in) and predictions."""
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32, device=device)

    batch = 4096
    preds_all = []
    for i in range(0, len(X_t), batch):
        p = model.predict_mean(X_t[i:i+batch])
        preds_all.append(p.cpu().numpy())
    pred = np.concatenate(preds_all, axis=0)

    mse = ((pred - Y) ** 2).mean(axis=0)
    mae = np.abs(pred - Y).mean(axis=0)
    return mse, mae, pred


# ---------------- Goal-directedness ----------------

def goal_directedness_report(X_val, Y_val, pred_val):
    """Proxy check: does the model move the cube toward the origin?"""
    cube_x = X_val[:, 0]
    cube_y = X_val[:, 1]

    pred_toward_x = (cube_x * pred_val[:, 0]) < 0
    true_toward_x = (cube_x * Y_val[:, 0]) < 0
    pred_toward_y = (cube_y * pred_val[:, 1]) < 0
    true_toward_y = (cube_y * Y_val[:, 1]) < 0

    print(f"\nGoal-directedness proxy (toward origin):")
    print(f"  X-axis  — model: {pred_toward_x.mean():.2%}, "
          f"ground truth: {true_toward_x.mean():.2%}")
    print(f"  Y-axis  — model: {pred_toward_y.mean():.2%}, "
          f"ground truth: {true_toward_y.mean():.2%}")


def check_goal_direction(X_val_raw, Y_val, pred_val, episode_ids_val, log_dir):
    """Compare model-predicted and ground-truth goal-directedness."""
    episodes_csv = log_dir.parent / "episodes.csv"
    if not episodes_csv.exists():
        print(f"\n[goal check] episodes.csv not found at {episodes_csv}")
        return

    ep_df = pd.read_csv(episodes_csv)
    ep_df = ep_df.drop_duplicates(subset="episode", keep="last")
    ep_df = ep_df.set_index("episode")

    motor_files = sorted(log_dir.glob("episode_*_motors.csv"))

    counts = {
        "closer": 0, "farther": 0, "equal": 0,
        "true_closer": 0, "true_farther": 0, "true_equal": 0,
        "skipped": 0,
    }

    for i in range(len(X_val_raw)):
        file_idx = int(episode_ids_val[i])
        if file_idx >= len(motor_files):
            counts["skipped"] += 1
            continue

        fname = motor_files[file_idx].stem
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

        # Model-predicted next position
        d_after_pred = np.hypot(cx + dx - goal_x, cy + dy - goal_y)

        # Ground-truth next position
        d_after_true = np.hypot(
            cx + float(Y_val[i, 0]) - goal_x,
            cy + float(Y_val[i, 1]) - goal_y,
        )

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

    total_pred = counts["closer"] + counts["farther"] + counts["equal"]
    total_true = counts["true_closer"] + counts["true_farther"] + counts["true_equal"]

    print(f"\nGoal-directedness (using episodes.csv goals):")
    if total_pred > 0:
        print(f"  Model predictions:")
        print(f"    closer  : {counts['closer'] / total_pred:.2%}")
        print(f"    farther : {counts['farther'] / total_pred:.2%}")
        print(f"    equal   : {counts['equal'] / total_pred:.2%}")
    if total_true > 0:
        print(f"  Ground truth:")
        print(f"    closer  : {counts['true_closer'] / total_true:.2%}")
        print(f"    farther : {counts['true_farther'] / total_true:.2%}")
        print(f"    equal   : {counts['true_equal'] / total_true:.2%}")
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

    # ---- Normalization (train stats only) ----
    X_mean = X_tr.mean(axis=0)
    X_std = X_tr.std(axis=0) + 1e-8
    Y_mean = Y_tr.mean(axis=0)
    Y_std = Y_tr.std(axis=0) + 1e-8

    X_tr_n = (X_tr - X_mean) / X_std
    X_val_n = (X_val - X_mean) / X_std
    Y_tr_n = (Y_tr - Y_mean) / Y_std
    Y_val_n = (Y_val - Y_mean) / Y_std

    # ---- DataLoader ----
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
        val_mse, val_mae, _ = evaluate(model, X_val_n, Y_val_n, device)
        val_mse_mean = float(val_mse.mean())

        if val_mse_mean < best_val_mse:
            best_val_mse = val_mse_mean
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}

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
    _, _, pred_val_n = evaluate(model, X_val_n, Y_val_n, device)
    pred_val = pred_val_n * Y_std + Y_mean

    residual = pred_val - Y_val
    val_mse_phys = (residual ** 2).mean(axis=0)
    val_mae_phys = np.abs(residual).mean(axis=0)

    print(f"\nValidation MSE (physical units):")
    print(f"  Δcube_x: {val_mse_phys[0]:.12f} m²  "
          f"(RMSE {np.sqrt(val_mse_phys[0]) * 1000:.4f} mm)")
    print(f"  Δcube_y: {val_mse_phys[1]:.12f} m²  "
          f"(RMSE {np.sqrt(val_mse_phys[1]) * 1000:.4f} mm)")

    print(f"Validation MAE (physical units):")
    print(f"  Δcube_x: {val_mae_phys[0] * 1000:.4f} mm")
    print(f"  Δcube_y: {val_mae_phys[1] * 1000:.4f} mm")

    # Baseline (predict mean)
    baseline_mse = ((Y_val - Y_val.mean(axis=0)) ** 2).mean(axis=0)
    print(f"\nBaseline (predict mean) RMSE:")
    print(f"  Δcube_x: {np.sqrt(baseline_mse[0]) * 1000:.4f} mm")
    print(f"  Δcube_y: {np.sqrt(baseline_mse[1]) * 1000:.4f} mm")

    print(f"\nDelta magnitude reference:")
    print(f"  Mean |Δcube_x|: {np.abs(Y_val[:, 0]).mean() * 1000:.4f} mm")
    print(f"  Mean |Δcube_y|: {np.abs(Y_val[:, 1]).mean() * 1000:.4f} mm")

    # ---- Goal-directedness ----
    goal_directedness_report(X_val, Y_val, pred_val)

    # Rebuild the episode index list for the val set
    X_all_eps = ep_ids[np.isin(ep_ids, val_eps)]
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