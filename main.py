# main.py
# -*- coding: utf-8 -*-


from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils.loss import LossPack, compute_metrics


# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------
# Args
# ----------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MR brake force control inverse model (VMD-GRU-Attention)")

    # mode
    p.add_argument("--mode", type=str, default="train", choices=["train", "test", "infer"],
                   help="Run mode: train/test/infer")
    p.add_argument("--seed", type=int, default=42)

    # paths
    p.add_argument("--data_csv", type=str, default="force_control_dataset.csv",
                   help="Path to processed dataset CSV")
    p.add_argument("--out_dir", type=str, default="fig_GRU_Attention_MRB",
                   help="Output directory for figures/logs/results")
    p.add_argument("--ckpt_path", type=str, default="best_model.pth",
                   help="Checkpoint path to save/load")

    # dataset columns
    p.add_argument("--file_col", type=str, default="file",
                   help="Column name for file id (optional). If not exists, whole CSV treated as one sequence.")
    p.add_argument("--time_col", type=str, default="time")
    p.add_argument("--input_cols", type=str, default="force",
                   help="Comma-separated input columns, e.g. 'force' or 'force,force_dot'")
    p.add_argument("--target_col", type=str, default="current",
                   help="Target column name, e.g. 'current' (desired -> current mapping).")

    # sequence settings
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--pred_len", type=int, default=1)  # here we do 1-step prediction by default
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.1)

    # training hyperparams
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--num_workers", type=int, default=0)

    # device
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    # model selection
    p.add_argument("--model", type=str, default="Attention_GRU",
                   choices=["GRU", "LSTM", "Attention_GRU"],
                   help="Choose model architecture")

    # model hyperparams
    p.add_argument("--hidden_size", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--attn_dim", type=int, default=128)

    # loss
    p.add_argument("--loss", type=str, default="rmse", choices=["rmse", "mse", "mae", "huber"])
    p.add_argument("--huber_delta", type=float, default=1.0)

    # checkpoint behavior
    p.add_argument("--resume", action="store_true", help="Resume from ckpt if exists (train mode).")
    p.add_argument("--save_best_only", action="store_true", help="Save only best model based on val RMSE.")
    p.add_argument("--metric_best", type=str, default="rmse", choices=["rmse", "loss"])

    # inference output
    p.add_argument("--infer_out", type=str, default="infer_results.npz",
                   help="NPZ output for inference results")

    return p


# ----------------------------
# Dataset
# ----------------------------
class SequenceCSVDataset(Dataset):


    def __init__(
        self,
        df: pd.DataFrame,
        input_cols: List[str],
        target_col: str,
        seq_len: int = 64,
        pred_len: int = 1,
        stride: int = 1,
        file_col: Optional[str] = None,
    ):
        self.input_cols = input_cols
        self.target_col = target_col
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.stride = stride
        self.file_col = file_col if file_col and file_col in df.columns else None

        # Ensure numeric
        for c in input_cols + [target_col]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=input_cols + [target_col]).reset_index(drop=True)

        self.samples: List[Tuple[np.ndarray, np.ndarray]] = []

        if self.file_col:
            for fid, g in df.groupby(self.file_col, sort=False):
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


def split_dataset_indices(n: int, train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
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


    if args.model == "GRU":
        from inverse_model_VMD_GRU_Attention.models.GRU_model import GRUModel
        model = GRUModel(input_dim=input_dim,
                         hidden_size=args.hidden_size,
                         num_layers=args.num_layers,
                         dropout=args.dropout,
                         output_dim=args.pred_len)
        return model

    if args.model == "LSTM":
        from inverse_model_VMD_GRU_Attention.models.LSTM_model import LSTMModel
        model = LSTMModel(input_dim=input_dim,
                          hidden_size=args.hidden_size,
                          num_layers=args.num_layers,
                          dropout=args.dropout,
                          output_dim=args.pred_len)
        return model

    if args.model == "Attention_GRU":
        from inverse_model_VMD_GRU_Attention.models.Attention_GRU_model import AttentionGRUModel
        model = AttentionGRUModel(input_dim=input_dim,
                                  hidden_size=args.hidden_size,
                                  num_layers=args.num_layers,
                                  dropout=args.dropout,
                                  attn_dim=args.attn_dim,
                                  output_dim=args.pred_len)
        return model

    raise ValueError(f"Unknown model: {args.model}")


# ----------------------------
# Train / Eval loops
# ----------------------------
def to_device(batch, device):
    x, y = batch
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Dict[str, float]:
    model.eval()
    losses = []
    preds_all = []
    trues_all = []
    for batch in loader:
        x, y = to_device(batch, device)
        pred = model(x)  # expected (B, pred_len) or (B, pred_len, 1)
        pred = pred.view(y.shape)  # align shape
        loss = criterion(pred, y)
        losses.append(loss.item())
        preds_all.append(pred.detach().cpu())
        trues_all.append(y.detach().cpu())

    pred_cat = torch.cat(preds_all, dim=0)
    true_cat = torch.cat(trues_all, dim=0)

    metrics = compute_metrics(pred_cat, true_cat)
    metrics["loss"] = float(np.mean(losses)) if losses else 1e9
    return metrics


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                    optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    losses = []
    for batch in loader:
        x, y = to_device(batch, device)
        optimizer.zero_grad()
        pred = model(x)
        pred = pred.view(y.shape)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else 1e9


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_metric: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
    }, path)


def load_checkpoint(path: Path, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optim_state" in ckpt:
        optimizer.load_state_dict(ckpt["optim_state"])
    epoch = int(ckpt.get("epoch", 0))
    best_metric = float(ckpt.get("best_metric", 1e9))
    return epoch, best_metric


# ----------------------------
# Main
# ----------------------------
def main():
    args = build_argparser().parse_args()

    set_seed(args.seed)

    # device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # paths
    data_csv = Path(args.data_csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    ckpt_path = Path(args.ckpt_path).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_csv.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {data_csv}")

    # load data
    df = pd.read_csv(data_csv, encoding="utf-8-sig")
    input_cols = [c.strip() for c in args.input_cols.split(",") if c.strip()]
    target_col = args.target_col

    for c in input_cols + [target_col]:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in CSV. Available columns: {list(df.columns)}")

    file_col = args.file_col if args.file_col in df.columns else None

    dataset = SequenceCSVDataset(
        df=df,
        input_cols=input_cols,
        target_col=target_col,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        stride=args.stride,
        file_col=file_col,
    )

    if len(dataset) == 0:
        raise RuntimeError("No samples generated. Check seq_len/pred_len/stride or CSV content.")

    # split
    train_idx, val_idx, test_idx = split_dataset_indices(
        n=len(dataset),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    def subset_loader(indices, shuffle: bool) -> DataLoader:
        sub = torch.utils.data.Subset(dataset, indices.tolist())
        return DataLoader(sub, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.num_workers, drop_last=False)

    train_loader = subset_loader(train_idx, shuffle=True)
    val_loader = subset_loader(val_idx, shuffle=False)
    test_loader = subset_loader(test_idx, shuffle=False)

    # model
    input_dim = len(input_cols)
    model = build_model(args, input_dim=input_dim).to(device)

    # loss
    loss_pack = LossPack(name=args.loss, huber_delta=args.huber_delta)
    criterion = loss_pack  # callable

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # resume / load for test/infer
    start_epoch = 0
    best_metric = 1e9

    if args.mode in ["test", "infer"]:
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found for {args.mode}: {ckpt_path}")
        load_checkpoint(ckpt_path, model, optimizer=None)
        print(f"[OK] Loaded checkpoint: {ckpt_path}")

    if args.mode == "train":
        if args.resume and ckpt_path.exists():
            start_epoch, best_metric = load_checkpoint(ckpt_path, model, optimizer=optimizer)
            print(f"[OK] Resumed from {ckpt_path}, start_epoch={start_epoch}, best_metric={best_metric:.6f}")

        print(f"[INFO] device={device}, samples={len(dataset)} "
              f"(train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)})")

        # training loop
        for epoch in range(start_epoch, args.epochs):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_metrics = evaluate(model, val_loader, criterion, device)

            # choose best metric
            if args.metric_best == "rmse":
                current = val_metrics["rmse"]
            else:
                current = val_metrics["loss"]

            improved = current < best_metric
            if improved:
                best_metric = current
                if args.save_best_only:
                    save_checkpoint(ckpt_path, model, optimizer, epoch=epoch + 1, best_metric=best_metric)

            # logging
            print(f"[Epoch {epoch+1:03d}/{args.epochs}] "
                  f"train_loss={train_loss:.6f} | "
                  f"val_loss={val_metrics['loss']:.6f} "
                  f"val_rmse={val_metrics['rmse']:.6f} "
                  f"val_mae={val_metrics['mae']:.6f} "
                  f"val_r2={val_metrics['r2']:.6f} "
                  f"{'| BEST ✓' if improved else ''}")

        # If not save_best_only, you may still want to save final:
        if not args.save_best_only:
            save_checkpoint(ckpt_path, model, optimizer, epoch=args.epochs, best_metric=best_metric)

        print(f"[DONE] Training finished. Best ({args.metric_best}) = {best_metric:.6f}. ckpt={ckpt_path}")

    elif args.mode == "test":
        metrics = evaluate(model, test_loader, criterion, device)
        print("[TEST] loss={loss:.6f}, rmse={rmse:.6f}, mae={mae:.6f}, r2={r2:.6f}".format(**metrics))

        # save predictions for plotting
        model.eval()
        preds_all = []
        trues_all = []
        for batch in test_loader:
            x, y = to_device(batch, device)
            pred = model(x).view(y.shape)
            preds_all.append(pred.detach().cpu().numpy())
            trues_all.append(y.detach().cpu().numpy())

        y_pred = np.concatenate(preds_all, axis=0)
        y_true = np.concatenate(trues_all, axis=0)

        out_npz = out_dir / "test_results.npz"
        np.savez(out_npz, y_true=y_true, y_pred=y_pred)
        print(f"[OK] Saved test predictions: {out_npz}")

    elif args.mode == "infer":

        model.eval()
        preds_all = []
        for batch in test_loader:
            x, _y = to_device(batch, device)
            pred = model(x)
            preds_all.append(pred.detach().cpu().numpy())

        y_pred = np.concatenate(preds_all, axis=0)
        out_npz = out_dir / args.infer_out
        np.savez(out_npz, y_pred=y_pred)
        print(f"[OK] Saved inference output: {out_npz}")

    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
