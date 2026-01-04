# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils.loss import LossPack, compute_metrics


def _get(args, name, default=None):
    return getattr(args, name, default)


def pick_device(device_arg: str = "auto") -> torch.device:
    if device_arg == "cuda":
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SequenceCSVDataset(Dataset):
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
        self.samples = []
        self.input_cols = input_cols
        self.target_col = target_col
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.stride = stride
        self.file_col = file_col if file_col and file_col in df.columns else None

        for c in input_cols + [target_col]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=input_cols + [target_col]).reset_index(drop=True)

        if self.file_col:
            for _, g in df.groupby(self.file_col, sort=False):
                self._add_group(g)
        else:
            self._add_group(df)

    def _add_group(self, g: pd.DataFrame):
        X_all = g[self.input_cols].to_numpy(dtype=np.float32)
        y_all = g[self.target_col].to_numpy(dtype=np.float32)
        n = len(g)
        L, P = self.seq_len, self.pred_len
        if n < L + P:
            return
        for start in range(0, n - (L + P) + 1, self.stride):
            x = X_all[start:start + L]
            y = y_all[start + L:start + L + P]
            self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.from_numpy(x), torch.from_numpy(y)


def split_indices(n: int, train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    idx = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


def build_model(args, input_dim: int) -> nn.Module:
    model_name = _get(args, "model", "Attention_GRU")

    if model_name == "GRU":
        from inverse_model_VMD_GRU_Attention.models.GRU_model import GRUModel
        return GRUModel(input_dim=input_dim,
                        hidden_size=_get(args, "hidden_size", 128),
                        num_layers=_get(args, "num_layers", 2),
                        dropout=_get(args, "dropout", 0.1),
                        output_dim=_get(args, "pred_len", 1))

    if model_name == "LSTM":
        from inverse_model_VMD_GRU_Attention.models.LSTM_model import LSTMModel
        return LSTMModel(input_dim=input_dim,
                         hidden_size=_get(args, "hidden_size", 128),
                         num_layers=_get(args, "num_layers", 2),
                         dropout=_get(args, "dropout", 0.1),
                         output_dim=_get(args, "pred_len", 1))

    if model_name in ["Attention_GRU", "GRU_Attention"]:
        from inverse_model_VMD_GRU_Attention.models.Attention_GRU_model import AttentionGRUModel
        return AttentionGRUModel(input_dim=input_dim,
                                 hidden_size=_get(args, "hidden_size", 128),
                                 num_layers=_get(args, "num_layers", 2),
                                 dropout=_get(args, "dropout", 0.1),
                                 attn_dim=_get(args, "attn_dim", 128),
                                 output_dim=_get(args, "pred_len", 1))

    raise ValueError(f"Unknown model: {model_name}")


def load_ckpt(path: Path, model: nn.Module):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])


@torch.no_grad()
def test(args):
    device = pick_device(_get(args, "device", "auto"))

    data_csv = Path(_get(args, "data_csv", "force_control_dataset.csv"))
    out_dir = Path(_get(args, "out_dir", "fig_GRU_Attention_MRB"))
    ckpt_path = Path(_get(args, "ckpt_path", "best_model.pth"))

    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_csv.exists():
        raise FileNotFoundError(f"CSV not found: {data_csv.resolve()}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path.resolve()}")

    df = pd.read_csv(data_csv, encoding="utf-8-sig")

    input_cols = [c.strip() for c in _get(args, "input_cols", "force").split(",") if c.strip()]
    target_col = _get(args, "target_col", "current")

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
        raise RuntimeError("No samples built. Check seq_len/pred_len/stride and your CSV.")

    train_idx, val_idx, test_idx = split_indices(
        len(dataset),
        float(_get(args, "train_ratio", 0.8)),
        float(_get(args, "val_ratio", 0.1)),
        float(_get(args, "test_ratio", 0.1)),
        int(_get(args, "seed", 42)),
    )

    test_loader = DataLoader(
        torch.utils.data.Subset(dataset, test_idx.tolist()),
        batch_size=int(_get(args, "batch_size", 64)),
        shuffle=False,
        num_workers=int(_get(args, "num_workers", 0)),
        drop_last=False,
    )

    model = build_model(args, input_dim=len(input_cols)).to(device)
    load_ckpt(ckpt_path, model)
    model.eval()

    criterion = LossPack(name=_get(args, "loss", "rmse"), huber_delta=float(_get(args, "huber_delta", 1.0)))

    losses = []
    preds_all, trues_all = [], []

    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x).view(y.shape)
        loss = criterion(pred, y)
        losses.append(loss.item())
        preds_all.append(pred.detach().cpu())
        trues_all.append(y.detach().cpu())

    y_pred = torch.cat(preds_all, dim=0)
    y_true = torch.cat(trues_all, dim=0)

    metrics = compute_metrics(y_pred, y_true)
    metrics["loss"] = float(np.mean(losses)) if losses else 1e9

    print("[TEST] loss={loss:.6f}, rmse={rmse:.6f}, mae={mae:.6f}, r2={r2:.6f}".format(**metrics))

    out_npz = out_dir / "test_results.npz"
    np.savez(out_npz, y_true=y_true.numpy(), y_pred=y_pred.numpy())
    print(f"[OK] Saved: {out_npz.resolve()}")
