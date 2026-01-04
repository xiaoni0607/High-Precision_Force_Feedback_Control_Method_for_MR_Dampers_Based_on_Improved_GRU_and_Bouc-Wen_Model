# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils.loss import LossPack, compute_metrics


# ----------------------------
# Utils
# ----------------------------
def _get(args, name, default=None):
    return getattr(args, name, default)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_device(device_arg: str = "auto") -> torch.device:
    if device_arg == "cuda":
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# Dataset
# ----------------------------
class SequenceCSVDataset(Dataset):
    """
    Sliding-window sequence dataset from a single CSV.

    If file_col exists -> group by file id.
    Each sample:
        X: (seq_len, input_dim)
        y: (pred_len,)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        input_cols: List[str],
        target_col: str,
        seq_len: int,
        pred_len: int,
        stride: int,
        file_col: Optional[str] = None,
    ):
        self.input_cols = input_cols
        self.target_col = target_col
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.stride = stride
        self.file_col = file_col if file_col and file_col in df.columns else None

        # numeric clean
        for c in input_cols + [target_col]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=input_cols + [target_col]).reset_index(drop=True)

        self.samples: List[Tuple[np.ndarray, np.ndarray]] = []
        if self.file_col:
            for _, g in df.groupby(self.file_col, sort=False):
                self._add_group(g)
        else:
            self._add_group(df)

    def _add_group(self, g: pd.DataFrame) -> None:
        X_all = g[self.input_cols].to_numpy(dtype=np.float32)
        y_all = g[self.target_col].to_numpy(dtype=np.float32)

        n = len(g)
        L = self.seq_len
        P = self.pred_len
        if n < L + P:
            return

        for start in range(0, n - (L + P) + 1, self.stride):
            x = X_all[start:start + L]
            y = y_all[start + L:start + L + P]
            self.samples.append((x, y))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, y = self.samples[idx]
        return torch.from_numpy(x), torch.from_numpy(y)


def split_indices(n: int, train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    idx = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return train_idx, val_idx, test_idx


# ----------------------------
# Model factory
# ----------------------------
def build_model(args, input_dim: int) -> nn.Module:
    model_name = _get(args, "model", "Attention_GRU")

    if model_name == "GRU":
        from inverse_model_VMD_GRU_Attention.models.GRU_model import GRUModel
        return GRUModel(
            input_dim=input_dim,
            hidden_size=_get(args, "hidden_size", 128),
            num_layers=_get(args, "num_layers", 2),
            dropout=_get(args, "dropout", 0.1),
            output_dim=_get(args, "pred_len", 1),
        )

    if model_name == "LSTM":
        from inverse_model_VMD_GRU_Attention.models.LSTM_model import LSTMModel
        return LSTMModel(
            input_dim=input_dim,
            hidden_size=_get(args, "hidden_size", 128),
            num_layers=_get(args, "num_layers", 2),
            dropout=_get(args, "dropout", 0.1),
            output_dim=_get(args, "pred_len", 1),
        )

    if model_name in ["Attention_GRU", "GRU_Attention"]:
        from inverse_model_VMD_GRU_Attention.models.Attention_GRU_model import AttentionGRUModel
        return AttentionGRUModel(
            input_dim=input_dim,
            hidden_size=_get(args, "hidden_size", 128),
            num_layers=_get(args, "num_layers", 2),
            dropout=_get(args, "dropout", 0.1),
            attn_dim=_get(args, "attn_dim", 128),
            output_dim=_get(args, "pred_len", 1),
        )

    raise ValueError(f"Unknown model: {model_name}")


# ----------------------------
# Train / Eval
# ----------------------------
def to_device(batch, device):
    x, y = batch
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    losses = []
    preds_all, trues_all = [], []

    for batch in loader:
        x, y = to_device(batch, device)
        pred = model(x).view(y.shape)
        loss = criterion(pred, y)
        losses.append(loss.item())
        preds_all.append(pred.detach().cpu())
        trues_all.append(y.detach().cpu())

    if not losses:
        return {"loss": 1e9, "rmse": 1e9, "mae": 1e9, "r2": -1e9}

    y_pred = torch.cat(preds_all, dim=0)
    y_true = torch.cat(trues_all, dim=0)

    metrics = compute_metrics(y_pred, y_true)
    metrics["loss"] = float(np.mean(losses))
    return metrics


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                    optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    losses = []

    for batch in loader:
        x, y = to_device(batch, device)
        optimizer.zero_grad()
        pred = model(x).view(y.shape)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return float(np.mean(losses)) if losses else 1e9


def save_ckpt(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_metric: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state": model.state_dict(),
         "optim_state": optimizer.state_dict(),
         "epoch": epoch,
         "best_metric": best_metric},
        path
    )


def load_ckpt(path: Path, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optim_state" in ckpt:
        optimizer.load_state_dict(ckpt["optim_state"])
    return int(ckpt.get("epoch", 0)), float(ckpt.get("best_metric", 1e9))


# ----------------------------
# Public API
# ----------------------------
def train(args):
    """
    Entry for main.py:
        from inverse_model_...train import train
        train(args)
    """

    set_seed(_get(args, "seed", 42))
    device = pick_device(_get(args, "device", "auto"))

    data_csv = Path(_get(args, "data_csv", "force_control_dataset.csv"))
    out_dir = Path(_get(args, "out_dir", "fig_GRU_Attention_MRB"))
    ckpt_path = Path(_get(args, "ckpt_path", "best_model.pth"))

    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_csv.exists():
        raise FileNotFoundError(f"CSV not found: {data_csv.resolve()}")

    df = pd.read_csv(data_csv, encoding="utf-8-sig")

    input_cols = [c.strip() for c in _get(args, "input_cols", "force").split(",") if c.strip()]
    target_col = _get(args, "target_col", "current")

    for c in input_cols + [target_col]:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in CSV. Columns={list(df.columns)}")

    file_col = _get(args, "file_col", "file")
    file_col = file_col if file_col in df.columns else None

    dataset = SequenceCSVDataset(
        df=df,
        input_cols=input_cols,
        target_col=target_col,
        seq_len=int(_get(args, "seq_len", 64)),
        pred_len=int(_get(args, "pred_len", 1)),
        stride=int(_get(args, "stride", 1)),
        file_col=file_col,
    )

    if len(dataset) == 0:
        raise RuntimeError("No samples built. Check seq_len/pred_len/stride and your CSV content.")

    train_ratio = float(_get(args, "train_ratio", 0.8))
    val_ratio = float(_get(args, "val_ratio", 0.1))
    test_ratio = float(_get(args, "test_ratio", 0.1))

    train_idx, val_idx, _test_idx = split_indices(len(dataset), train_ratio, val_ratio, test_ratio, _get(args, "seed", 42))

    def make_loader(indices, shuffle: bool):
        sub = torch.utils.data.Subset(dataset, indices.tolist())
        return DataLoader(
            sub,
            batch_size=int(_get(args, "batch_size", 64)),
            shuffle=shuffle,
            num_workers=int(_get(args, "num_workers", 0)),
            drop_last=False,
        )

    train_loader = make_loader(train_idx, shuffle=True)
    val_loader = make_loader(val_idx, shuffle=False)

    model = build_model(args, input_dim=len(input_cols)).to(device)

    # loss
    criterion = LossPack(
        name=_get(args, "loss", "rmse"),
        huber_delta=float(_get(args, "huber_delta", 1.0)),
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(_get(args, "lr", 1e-3)),
        weight_decay=float(_get(args, "weight_decay", 0.0)),
    )

    epochs = int(_get(args, "epochs", 200))
    save_best_only = bool(_get(args, "save_best_only", True))
    metric_best = _get(args, "metric_best", "rmse")
    resume = bool(_get(args, "resume", False))

    start_epoch = 0
    best_metric = 1e9

    if resume and ckpt_path.exists():
        start_epoch, best_metric = load_ckpt(ckpt_path, model, optimizer)
        print(f"[OK] Resume ckpt={ckpt_path} start_epoch={start_epoch} best_metric={best_metric:.6f}")

    print(f"[INFO] device={device} samples={len(dataset)} train={len(train_idx)} val={len(val_idx)}")

    for ep in range(start_epoch, epochs):
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        current = val_metrics["rmse"] if metric_best == "rmse" else val_metrics["loss"]
        improved = current < best_metric

        if improved:
            best_metric = current
            save_ckpt(ckpt_path, model, optimizer, epoch=ep + 1, best_metric=best_metric)

        print(f"[Epoch {ep+1:03d}/{epochs}] "
              f"train_loss={tr_loss:.6f} | "
              f"val_loss={val_metrics['loss']:.6f} "
              f"val_rmse={val_metrics['rmse']:.6f} "
              f"val_mae={val_metrics['mae']:.6f} "
              f"val_r2={val_metrics['r2']:.6f} "
              f"{'| BEST ✓' if improved else ''}")

    if not save_best_only:
        save_ckpt(ckpt_path, model, optimizer, epoch=epochs, best_metric=best_metric)

    print(f"[DONE] Best({metric_best})={best_metric:.6f} ckpt={ckpt_path.resolve()}")
