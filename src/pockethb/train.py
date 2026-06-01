"""Training loop for the Hb regressor."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 6           # early stop on val MAE
    image_size: int = 224
    num_workers: int = 0        # colab safer at 0–2; windows defaults to 0
    device: str = "auto"        # "auto", "cuda", "cpu"


@dataclass
class TrainResult:
    best_epoch: int
    best_val_mae: float
    best_state_dict: dict
    history: list[dict] = field(default_factory=list)
    test_metrics: dict | None = None


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _epoch(model, loader, optim, device, train: bool):
    model.train(train)
    loss_fn = torch.nn.MSELoss()
    losses, preds_all, y_all, pid_all, cidx_all = [], [], [], [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y, pid, cidx in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x)
            loss = loss_fn(pred, y)
            if train:
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()
            losses.append(float(loss.item()) * x.size(0))
            preds_all.extend(pred.detach().cpu().numpy().tolist())
            y_all.extend(y.detach().cpu().numpy().tolist())
            pid_all.extend([int(p) for p in pid])
            cidx_all.extend([int(c) for c in cidx])
    n = len(loader.dataset)
    mean_loss = float(sum(losses) / max(n, 1))
    return {
        "loss": mean_loss,
        "preds": preds_all,
        "y": y_all,
        "pid": pid_all,
        "cidx": cidx_all,
    }


def _metrics(preds, y, pid):
    """Return crop-level and patient-level (mean over a patient's crops) metrics."""
    preds = np.asarray(preds)
    y = np.asarray(y)
    pid = np.asarray(pid)
    crop = {
        "MAE": float(mean_absolute_error(y, preds)),
        "RMSE": float(np.sqrt(mean_squared_error(y, preds))),
        "R2": float(r2_score(y, preds)),
        "n": int(len(y)),
    }
    df = pd.DataFrame({"pid": pid, "y": y, "pred": preds})
    agg = df.groupby("pid").agg(y=("y", "first"), pred=("pred", "mean"))
    patient = {
        "MAE": float(mean_absolute_error(agg["y"], agg["pred"])),
        "RMSE": float(np.sqrt(mean_squared_error(agg["y"], agg["pred"]))),
        "R2": float(r2_score(agg["y"], agg["pred"])),
        "n": int(len(agg)),
    }
    return {"crop": crop, "patient": patient}


def train_model(
    model,
    train_ds,
    val_ds,
    test_ds=None,
    cfg: TrainConfig | None = None,
) -> TrainResult:
    cfg = cfg or TrainConfig()
    device = _device(cfg.device)
    model = model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"))

    best_val_mae = float("inf")
    best_epoch = -1
    best_state = None
    bad_epochs = 0
    history = []

    print(f"training on {device}  |  train={len(train_ds)} val={len(val_ds)} "
          f"epochs={cfg.epochs} bs={cfg.batch_size} lr={cfg.lr}")

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr = _epoch(model, train_loader, optim, device, train=True)
        va = _epoch(model, val_loader, optim, device, train=False)
        m_tr = _metrics(tr["preds"], tr["y"], tr["pid"])
        m_va = _metrics(va["preds"], va["y"], va["pid"])
        val_mae_p = m_va["patient"]["MAE"]
        dt = time.time() - t0
        print(f"epoch {epoch:3d}  tr_loss={tr['loss']:.3f}  "
              f"tr_MAE_p={m_tr['patient']['MAE']:.3f}  "
              f"va_MAE_p={val_mae_p:.3f}  "
              f"va_R2_p={m_va['patient']['R2']:+.3f}  "
              f"({dt:.1f}s)")

        history.append({
            "epoch": epoch,
            "tr_loss": tr["loss"],
            "tr_MAE_p": m_tr["patient"]["MAE"],
            "va_MAE_p": val_mae_p,
            "va_R2_p": m_va["patient"]["R2"],
            "wall_s": dt,
        })

        if val_mae_p < best_val_mae - 1e-3:
            best_val_mae = val_mae_p
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print(f"early stop at epoch {epoch} (no val improvement for {cfg.patience} epochs)")
                break

    print(f"\nbest val patient-MAE = {best_val_mae:.3f} at epoch {best_epoch}")

    # restore best
    model.load_state_dict(best_state)

    test_metrics = None
    if test_ds is not None:
        test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                                 num_workers=cfg.num_workers)
        te = _epoch(model, test_loader, optim, device, train=False)
        test_metrics = _metrics(te["preds"], te["y"], te["pid"])
        print(f"\nTEST  crop    MAE={test_metrics['crop']['MAE']:.3f}  "
              f"RMSE={test_metrics['crop']['RMSE']:.3f}  R²={test_metrics['crop']['R2']:+.3f}  "
              f"n={test_metrics['crop']['n']}")
        print(f"TEST  patient MAE={test_metrics['patient']['MAE']:.3f}  "
              f"RMSE={test_metrics['patient']['RMSE']:.3f}  R²={test_metrics['patient']['R2']:+.3f}  "
              f"n={test_metrics['patient']['n']}")

    return TrainResult(
        best_epoch=best_epoch,
        best_val_mae=best_val_mae,
        best_state_dict=best_state,
        history=history,
        test_metrics=test_metrics,
    )
