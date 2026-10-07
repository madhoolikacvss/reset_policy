"""
PETS-style dynamics model for the string-driven cube board.

Changes vs. previous version
  1. full_eval() now normalizes X, and de-normalizes predictions back to physical units.
  2. Goal-directedness uses the goal-relative features (dist_to_goal_x/y), not the origin.
  3. Episode-level K-fold cross validation. Each fold has:
        train episodes  -> fit weights
        inner-val eps   -> pick best epoch (early stopping)
        held-out fold   -> reported metrics (never used for any decision)
  4. Simple baselines per fold (predict-mean, predict-zero, prev_delta, ridge).
  5. Optional final model trained on all episodes after CV.

Usage:
    python train_dynamics_kfold.py --log-dir /path/to/motor_logs --out-dir ./dyn_out --k-folds 5
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- Config ----------------

BASE_INPUT_COLS = [
    "cube_x", "cube_y",
    "dist_to_goal_x", "dist_to_goal_y", "dist_to_goal",
    "obs_goal_x_norm", "obs_goal_y_norm",
    "obs_motor_16_pos_delta_norm", "obs_motor_17_pos_delta_norm",
    "obs_motor_18_pos_delta_norm", "obs_motor_19_pos_delta_norm",
    "obs_motor_16_current_norm", "obs_motor_17_current_norm",
    "obs_motor_18_current_norm", "obs_motor_19_current_norm",
    "obs_horizontal_tension_norm", "obs_vertical_tension_norm", "obs_total_tension_norm",
    "obs_motor_16_target_error_norm", "obs_motor_17_target_error_norm",
    "obs_motor_18_target_error_norm", "obs_motor_19_target_error_norm",
    "action_m16", "action_m17", "action_m18", "action_m19",
]
LAG_INPUT_COLS = ["prev_delta_cube_x", "prev_delta_cube_y"]
INPUT_COLS = BASE_INPUT_COLS + LAG_INPUT_COLS   # 27 dims (25 base + 2 lag)
OUTPUT_COLS = ["cube_x", "cube_y"]

IDX_DIST_X = INPUT_COLS.index("dist_to_goal_x")
IDX_DIST_Y = INPUT_COLS.index("dist_to_goal_y")
IDX_PREV_DX = INPUT_COLS.index("prev_delta_cube_x")
IDX_PREV_DY = INPUT_COLS.index("prev_delta_cube_y")

SEED = 0


# ---------------- Data loading ----------------

def load_all_episodes(log_dir: Path, verbose: bool = True):
    """Build (s_t, a_t) -> cube(t+1) - cube(t) pairs. Skips first/last row of each episode."""
    files = sorted(log_dir.glob("episode_*_motors.csv"))
    if not files:
        raise FileNotFoundError(f"No episodes found in {log_dir}")

    Xs, Ys, eps = [], [], []
    skipped_files = skipped_rows = 0

    for ep_idx, f in enumerate(files):
        df = pd.read_csv(f)
        if len(df) < 3:
            skipped_files += 1
            continue
        missing = [c for c in BASE_INPUT_COLS + OUTPUT_COLS if c not in df.columns]
        if missing:
            print(f"WARNING: {f.name} missing columns {missing}, skipping")
            skipped_files += 1
            continue

        cube = df[OUTPUT_COLS].to_numpy(dtype=np.float32)       # (T, 2)
        base = df[BASE_INPUT_COLS].to_numpy(dtype=np.float32)   # (T, 25)

        # t = 1 .. T-2 (need t-1 and t+1)
        prev_delta = cube[1:-1] - cube[:-2]
        delta = cube[2:] - cube[1:-1]
        s = np.concatenate([base[1:-1], prev_delta], axis=1).astype(np.float32)

        ok = np.isfinite(s).all(axis=1) & np.isfinite(delta).all(axis=1)
        skipped_rows += int((~ok).sum())
        Xs.append(s[ok])
        Ys.append(delta[ok].astype(np.float32))
        eps.append(np.full(ok.sum(), ep_idx, dtype=np.int64))

    if not Xs:
        raise RuntimeError("No valid transitions loaded.")

    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)
    episode_ids = np.concatenate(eps, axis=0)

    if verbose:
        print(f"Loaded {len(files)} files ({skipped_files} skipped), "
              f"{len(X)} transitions ({skipped_rows} rows skipped for NaN), "
              f"{X.shape[1]}-dim inputs, {len(np.unique(episode_ids))} usable episodes")
    return X, Y, episode_ids


def make_folds(episode_ids, k, seed):
    """Return list of k arrays of held-out episode ids (episode-level split, no leakage)."""
    rng = np.random.default_rng(seed)
    unique_eps = rng.permutation(np.unique(episode_ids))
    if k > len(unique_eps):
        raise ValueError(f"k-folds={k} > number of episodes={len(unique_eps)}")
    return np.array_split(unique_eps, k)


def split_inner_val(train_eps, frac, rng):
    """Carve an inner validation set (by episode) out of the training episodes."""
    train_eps = rng.permutation(train_eps)
    n_val = max(1, int(round(len(train_eps) * frac)))
    return train_eps[n_val:], train_eps[:n_val]   # (train, inner_val)


# ---------------- Normalization ----------------

class Normalizer:
    def __init__(self, X_tr, Y_tr):
        self.X_mean = X_tr.mean(axis=0)
        self.X_std = X_tr.std(axis=0)
        self.X_std = np.where(self.X_std < 1e-8, 1.0, self.X_std)   # constant features
        self.Y_mean = Y_tr.mean(axis=0)
        self.Y_std = np.where(Y_tr.std(axis=0) < 1e-8, 1.0, Y_tr.std(axis=0))

    def x(self, X):
        return ((X - self.X_mean) / self.X_std).astype(np.float32)

    def y(self, Y):
        return ((Y - self.Y_mean) / self.Y_std).astype(np.float32)

    def y_inv(self, Yn):
        return Yn * self.Y_std + self.Y_mean

    def state_dict(self):
        return dict(X_mean=self.X_mean, X_std=self.X_std, Y_mean=self.Y_mean, Y_std=self.Y_std)


# ---------------- Model ----------------

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=256, depth=3):
        super().__init__()
        layers, prev = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(prev, hidden), nn.SiLU()]
            prev = hidden
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Ensemble(nn.Module):
    def __init__(self, in_dim, out_dim, n_members=5, hidden=256, depth=3):
        super().__init__()
        self.members = nn.ModuleList(
            [MLP(in_dim, out_dim, hidden, depth) for _ in range(n_members)]
        )

    def forward(self, x):
        return torch.stack([m(x) for m in self.members], dim=0)

    def predict_mean(self, x):
        return self.forward(x).mean(dim=0)

    def predict_std(self, x):
        return self.forward(x).std(dim=0)


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ---------------- LR schedule ----------------

def make_lr_scheduler(optimizer, args):
    if args.lr_schedule == "cosine":
        def lr_lambda(epoch):
            if epoch < args.warmup_epochs:
                return (epoch + 1) / max(1, args.warmup_epochs)
            progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if args.lr_schedule == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    if args.lr_schedule == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=15
        )
    return None


# ---------------- Training / Evaluation ----------------

def train_one_epoch(model, optimizer, loader, device):
    model.train()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb)                                  # (members, batch, out)
        loss = ((preds - yb.unsqueeze(0)) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * xb.size(0)
        n += xb.size(0)
    return total / max(1, n)


@torch.no_grad()
def predict_normalized(model, X_n, device, batch=4096):
    """X_n: NORMALIZED inputs. Returns NORMALIZED ensemble-mean predictions."""
    model.eval()
    X_t = torch.tensor(X_n, dtype=torch.float32, device=device)
    out = [model.predict_mean(X_t[i:i + batch]).cpu().numpy()
           for i in range(0, len(X_t), batch)]
    return np.concatenate(out, axis=0)


def regression_metrics(pred, Y):
    err = pred - Y
    rmse = np.sqrt((err ** 2).mean(axis=0))
    mae = np.abs(err).mean(axis=0)
    ss_res = (err ** 2).sum(axis=0)
    ss_tot = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1 - ss_res / np.where(ss_tot == 0, 1e-12, ss_tot)
    return {"rmse": rmse, "mae": mae, "r2": r2}


def full_eval(model, X, Y, device, norm):
    """
    X, Y are RAW (physical units). Inputs are normalized internally and the
    predictions are de-normalized, so all returned metrics are in physical units.
    """
    pred_n = predict_normalized(model, norm.x(X), device)
    pred = norm.y_inv(pred_n)
    out = regression_metrics(pred, Y)
    out["pred"] = pred
    return out


def normalized_mse(model, X, Y, device, norm):
    pred_n = predict_normalized(model, norm.x(X), device)
    return float(((pred_n - norm.y(Y)) ** 2).mean())


# ---------------- Baselines ----------------

def ridge_baseline(X_tr, Y_tr, X_te, norm, lam=1.0):
    Xn = norm.x(X_tr)
    Xb = np.hstack([Xn, np.ones((len(Xn), 1), dtype=np.float32)])
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    A[-1, -1] -= lam                       # don't regularize bias
    W = np.linalg.solve(A, Xb.T @ norm.y(Y_tr))
    Xte = np.hstack([norm.x(X_te), np.ones((len(X_te), 1), dtype=np.float32)])
    return norm.y_inv(Xte @ W)


def baseline_report(X_tr, Y_tr, X_te, Y_te, norm):
    prev = X_te[:, [IDX_PREV_DX, IDX_PREV_DY]]
    preds = {
        "predict train-mean": np.tile(Y_tr.mean(axis=0), (len(Y_te), 1)),
        "predict zero": np.zeros_like(Y_te),
        "predict prev_delta": prev,
        "ridge (linear)": ridge_baseline(X_tr, Y_tr, X_te, norm),
    }
    return {name: regression_metrics(p, Y_te) for name, p in preds.items()}


# ---------------- Goal-directedness ----------------

def resolve_goal_sign(X, Y, mode):
    """
    Returns s in {+1, -1} such that  s * dist_to_goal_{x,y}  points TOWARD the goal.
      goal_minus_cube : dist = goal - cube  -> s = +1
      cube_minus_goal : dist = cube - goal  -> s = -1
      auto            : infer from data (sign of E[dist . delta_cube]); only a
                        sanity heuristic, set it explicitly if you know your logging.
    """
    if mode == "goal_minus_cube":
        return 1.0
    if mode == "cube_minus_goal":
        return -1.0
    score = float((X[:, IDX_DIST_X] * Y[:, 0] + X[:, IDX_DIST_Y] * Y[:, 1]).mean())
    s = 1.0 if score >= 0 else -1.0
    print(f"[goal sign auto-detect] mean(dist . delta_cube) = {score:+.3e} -> "
          f"{'goal - cube' if s > 0 else 'cube - goal'} convention (s={s:+.0f})")
    return s


def goal_directedness(X, Y, pred, goal_sign, min_dist=0.005):
    """
    Fraction of samples where the cube's step is toward the goal along each axis,
    for ground truth vs. model, restricted to samples at least `min_dist` away
    from the goal on that axis (otherwise 'toward' is meaningless).
    Also reports sign agreement between model and ground truth.
    """
    out = {}
    for ax, idx, name in [(0, IDX_DIST_X, "x"), (1, IDX_DIST_Y, "y")]:
        toward = goal_sign * X[:, idx]                 # >0 means goal is in +axis direction
        mask = np.abs(toward) > min_dist
        if mask.sum() == 0:
            out[name] = dict(n=0, model=np.nan, truth=np.nan, agree=np.nan)
            continue
        t, y, p = toward[mask], Y[mask, ax], pred[mask, ax]
        out[name] = dict(
            n=int(mask.sum()),
            model=float(((t * p) > 0).mean()),
            truth=float(((t * y) > 0).mean()),
            agree=float((np.sign(p) == np.sign(y)).mean()),
        )
    return out


# ---------------- Single training run ----------------

def train_model(X_tr, Y_tr, X_es, Y_es, args, device, tag=""):
    """
    Fit on (X_tr, Y_tr), choose best epoch on (X_es, Y_es) [early-stopping set].
    Returns best model (restored), normalizer, history dict.
    """
    norm = Normalizer(X_tr, Y_tr)

    train_ds = TensorDataset(torch.tensor(norm.x(X_tr)), torch.tensor(norm.y(Y_tr)))
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, drop_last=len(train_ds) > args.batch_size)

    model = Ensemble(X_tr.shape[1], Y_tr.shape[1], args.ensemble,
                     args.hidden, args.depth).to(device)
    model.apply(init_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = make_lr_scheduler(optimizer, args)

    best_mse, best_state, best_epoch = float("inf"), None, -1
    hist = dict(epochs=[], train_mse=[], val_mse=[], train_r2=[], val_r2=[])
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, optimizer, loader, device)
        es_mse = normalized_mse(model, X_es, Y_es, device, norm)

        if es_mse < best_mse:
            best_mse, best_epoch = es_mse, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            tr_m = full_eval(model, X_tr, Y_tr, device, norm)
            es_m = full_eval(model, X_es, Y_es, device, norm)
            hist["epochs"].append(epoch)
            hist["train_mse"].append(tr_loss)
            hist["val_mse"].append(es_mse)
            hist["train_r2"].append(tr_m["r2"].copy())
            hist["val_r2"].append(es_m["r2"].copy())
            print(f"{tag}epoch {epoch:4d} | lr {optimizer.param_groups[0]['lr']:.2e} | "
                  f"MSE(n) tr {tr_loss:.4f} val {es_mse:.4f} | "
                  f"RMSE mm tr x={tr_m['rmse'][0]*1000:.2f} y={tr_m['rmse'][1]*1000:.2f} "
                  f"val x={es_m['rmse'][0]*1000:.2f} y={es_m['rmse'][1]*1000:.2f} | "
                  f"R2 tr x={tr_m['r2'][0]:+.3f} y={tr_m['r2'][1]:+.3f} "
                  f"val x={es_m['r2'][0]:+.3f} y={es_m['r2'][1]:+.3f}")

        if args.lr_schedule == "plateau":
            scheduler.step(es_mse)
        elif scheduler is not None:
            scheduler.step()

    print(f"{tag}done in {time.time() - t0:.1f}s, best early-stop MSE(n) "
          f"{best_mse:.5f} at epoch {best_epoch}")
    model.load_state_dict(best_state)
    hist["best_epoch"] = best_epoch
    return model, norm, hist


def plot_learning_curves(hist, out_path):
    ep = hist["epochs"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(ep, hist["train_mse"], label="train")
    axes[0].plot(ep, hist["val_mse"], label="early-stop val")
    axes[0].set_yscale("log")
    axes[0].set_title("Loss (normalized MSE)")
    for i, name in enumerate(["x", "y"], start=1):
        axes[i].plot(ep, [r[i - 1] for r in hist["train_r2"]], label="train")
        axes[i].plot(ep, [r[i - 1] for r in hist["val_r2"]], label="early-stop val")
        axes[i].axhline(0, color="k", ls="--", alpha=0.3)
        axes[i].set_title(f"R2 for delta_cube_{name}")
    for a in axes:
        a.set_xlabel("Epoch")
        a.grid(True, alpha=0.3)
        a.legend()
        a.axvline(hist["best_epoch"], color="r", ls=":", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()


# ---------------- Reporting ----------------

def mean_std(vals):
    v = np.array(vals, dtype=np.float64)
    return v.mean(axis=0), v.std(axis=0)


def print_fold_table(fold, model_m, base_m, gd):
    print(f"\n--- Fold {fold}: held-out results (physical units) ---")
    hdr = f"{'Model':<24} {'RMSE x (mm)':>12} {'RMSE y (mm)':>12} {'MAE x':>8} {'MAE y':>8} {'R2 x':>8} {'R2 y':>8}"
    print(hdr)
    print("-" * len(hdr))

    def row(name, m):
        print(f"{name:<24} {m['rmse'][0]*1000:>12.3f} {m['rmse'][1]*1000:>12.3f} "
              f"{m['mae'][0]*1000:>8.3f} {m['mae'][1]*1000:>8.3f} "
              f"{m['r2'][0]:>8.4f} {m['r2'][1]:>8.4f}")

    row("MLP ensemble", model_m)
    for name, m in base_m.items():
        row(name, m)
    for ax in ("x", "y"):
        g = gd[ax]
        print(f"goal-directed {ax}: n={g['n']}, model {g['model']:.1%}, "
              f"truth {g['truth']:.1%}, model/truth sign agreement {g['agree']:.1%}")


def print_cv_summary(results, base_names):
    print("\n" + "=" * 90)
    print(f"CROSS-VALIDATION SUMMARY  ({len(results)} folds, mean ± std across folds)")
    print("=" * 90)
    hdr = f"{'Model':<24} {'RMSE x (mm)':>16} {'RMSE y (mm)':>16} {'R2 x':>14} {'R2 y':>14}"
    print(hdr)
    print("-" * len(hdr))

    def line(name, getter):
        rm, rs = mean_std([getter(r)["rmse"] * 1000 for r in results])
        r2m, r2s = mean_std([getter(r)["r2"] for r in results])
        print(f"{name:<24} {rm[0]:>9.3f}±{rs[0]:<6.3f} {rm[1]:>9.3f}±{rs[1]:<6.3f} "
              f"{r2m[0]:>7.4f}±{r2s[0]:<6.4f} {r2m[1]:>7.4f}±{r2s[1]:<6.4f}")

    line("MLP ensemble (test)", lambda r: r["model"])
    line("MLP ensemble (train)", lambda r: r["train"])
    for b in base_names:
        line(b, lambda r, b=b: r["base"][b])

    mean_name = "predict train-mean"
    red = np.array([1 - r["model"]["rmse"] / r["base"][mean_name]["rmse"] for r in results])
    print(f"\nRMSE reduction vs predict-mean: x {red[:, 0].mean():.1%} ± {red[:, 0].std():.1%}, "
          f"y {red[:, 1].mean():.1%} ± {red[:, 1].std():.1%}")

    print("\nGoal-directedness (fraction of steps moving toward goal):")
    for ax in ("x", "y"):
        m = np.nanmean([r["gd"][ax]["model"] for r in results])
        t = np.nanmean([r["gd"][ax]["truth"] for r in results])
        a = np.nanmean([r["gd"][ax]["agree"] for r in results])
        print(f"  {ax}-axis: model {m:.1%} | truth {t:.1%} | model/truth sign agreement {a:.1%}")

    r2 = np.array([r["model"]["r2"] for r in results])
    tr_r2 = np.array([r["train"]["r2"] for r in results])
    gap = (tr_r2 - r2).mean()
    print("\nINTERPRETATION")
    print(f"  mean test R2 = {r2.mean():.3f}, mean train-test R2 gap = {gap:.3f}")
    if r2.mean() < 0.1:
        print("  -> Little predictable signal. Check ridge/prev_delta baselines, label noise,")
        print("     and consider multi-step targets or more history.")
    elif gap > 0.2:
        print("  -> Overfitting. Reduce size, raise weight decay, or add data.")
    else:
        print("  -> Reasonable fit. Compare against the ridge baseline to see how much the MLP adds.")


# ---------------- Main ----------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log-dir", type=str, required=True, help="Dir with episode_*_motors.csv")
    p.add_argument("--out-dir", type=str, default="./dynamics_out")
    p.add_argument("--k-folds", type=int, default=5)
    p.add_argument("--inner-val-fraction", type=float, default=0.10,
                   help="Fraction of TRAIN episodes used for early stopping in each fold")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--ensemble", type=int, default=5)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--lr-schedule", type=str, default="cosine",
                   choices=["cosine", "step", "plateau", "none"])
    p.add_argument("--goal-convention", type=str, default="auto",
                   choices=["auto", "goal_minus_cube", "cube_minus_goal"],
                   help="How dist_to_goal_{x,y} is defined in your logs")
    p.add_argument("--goal-min-dist", type=float, default=0.005,
                   help="Ignore samples closer than this to the goal (per axis) in goal-directedness")
    p.add_argument("--no-final-model", action="store_true",
                   help="Skip training a final model on all data after CV")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--log-every", type=int, default=25)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    X, Y, ep_ids = load_all_episodes(Path(args.log_dir))
    goal_sign = resolve_goal_sign(X, Y, args.goal_convention)

    folds = make_folds(ep_ids, args.k_folds, args.seed)
    rng = np.random.default_rng(args.seed + 1)
    all_eps = np.unique(ep_ids)

    results, fold_states = [], []
    for k, test_eps in enumerate(folds):
        print("\n" + "=" * 90)
        print(f"FOLD {k + 1}/{len(folds)}  (held-out episodes: {len(test_eps)})")
        print("=" * 90)

        rest = np.setdiff1d(all_eps, test_eps)
        tr_eps, es_eps = split_inner_val(rest, args.inner_val_fraction, rng)

        m_tr, m_es, m_te = (np.isin(ep_ids, e) for e in (tr_eps, es_eps, test_eps))
        X_tr, Y_tr = X[m_tr], Y[m_tr]
        X_es, Y_es = X[m_es], Y[m_es]
        X_te, Y_te = X[m_te], Y[m_te]
        print(f"train {len(X_tr)} ({len(tr_eps)} eps) | early-stop {len(X_es)} ({len(es_eps)} eps) "
              f"| test {len(X_te)} ({len(test_eps)} eps)")

        model, norm, hist = train_model(X_tr, Y_tr, X_es, Y_es, args, device, tag=f"[f{k + 1}] ")
        plot_learning_curves(hist, out_dir / f"learning_curves_fold{k + 1}.png")

        te = full_eval(model, X_te, Y_te, device, norm)
        tr = full_eval(model, X_tr, Y_tr, device, norm)
        base = baseline_report(X_tr, Y_tr, X_te, Y_te, norm)
        gd = goal_directedness(X_te, Y_te, te["pred"], goal_sign, args.goal_min_dist)
        print_fold_table(k + 1, te, base, gd)

        results.append(dict(model=te, train=tr, base=base, gd=gd))
        fold_states.append(dict(model_state_dict=model.state_dict(), norm=norm.state_dict(),
                                test_episodes=test_eps.tolist(), best_epoch=hist["best_epoch"]))

    print_cv_summary(results, list(results[0]["base"].keys()))

    ckpt = dict(input_cols=INPUT_COLS, output_cols=OUTPUT_COLS, args=vars(args))
    torch.save({**ckpt, "folds": fold_states}, out_dir / "dynamics_cv_folds.pt")
    print(f"\nSaved per-fold models to {out_dir / 'dynamics_cv_folds.pt'}")

    # ---- Final model on all episodes (inner split only for epoch selection) ----
    if not args.no_final_model:
        print("\n" + "=" * 90)
        print("FINAL MODEL (all episodes; inner split for early stopping only)")
        print("=" * 90)
        tr_eps, es_eps = split_inner_val(all_eps, args.inner_val_fraction, rng)
        m_tr, m_es = np.isin(ep_ids, tr_eps), np.isin(ep_ids, es_eps)
        model, norm, hist = train_model(X[m_tr], Y[m_tr], X[m_es], Y[m_es], args, device, tag="[final] ")
        plot_learning_curves(hist, out_dir / "learning_curves_final.png")
        torch.save({**ckpt, "model_state_dict": model.state_dict(), **norm.state_dict(),
                    "best_epoch": hist["best_epoch"], "early_stop_episodes": es_eps.tolist()},
                   out_dir / "dynamics_model.pt")
        print(f"Saved final model to {out_dir / 'dynamics_model.pt'}")
        print("NOTE: CV metrics above are the honest performance estimate; the final model "
              "has no held-out test set.")


if __name__ == "__main__":
    main()