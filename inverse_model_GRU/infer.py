# -*- coding: utf-8 -*-
"""
MR Damper 多步预测（H步）完整脚本（去除注意力机制模块）
- 目标: 预测 [y_{t+1}..y_{t+H}]（level）或对应 Δy 向量（delta），训练在 z 空间，评估/作图还原物理单位
- 采样: WeightedRandomSampler（基于窗口内最大目标幅值 + 事件阈值）
- 损失: 事件加权 MSE（可关），日志打印未加权 z-MSE（与基线同口径）
- 诊断: 打印 z 空间 pred/std/MSE，辅助判断是否塌到常数
- 可视化: 额外输出“只看一步(h=1)”与“近似去重叠的平均轨迹”
"""

import os, math
import numpy as np
import pandas as pd
from typing import List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------- Utils ----------------
def set_seed(seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def metrics(y_true, y_pred, name=""):
    y_true = np.asarray(y_true).reshape(-1); y_pred = np.asarray(y_pred).reshape(-1)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{name}  MAE:{mae:.6f}  RMSE:{rmse:.6f}  R2:{r2:.6f}")
    return mae, rmse, r2

def persistence_baseline(last_values: np.ndarray, horizon: int) -> np.ndarray:
    """
    持久化基线：将每个时间点的最后一个值（last_value）重复用于预测接下来的 H 个时间步。
    """
    return np.repeat(last_values, horizon, axis=1)  # 复制列以生成持久化预测


def estimate_delay(true_series: np.ndarray, pred_series: np.ndarray, max_lag: int = 200):
    """
    返回最佳滞后（>0 表示预测“落后”于真值；单位=采样点）及对应相关系数。
    先做整数栅格搜索，再用 3 点抛物线插值到亚采样精度。
    """
    x = np.asarray(true_series, dtype=float).ravel()
    y = np.asarray(pred_series, dtype=float).ravel()
    x = x - x.mean(); y = y - y.mean()
    lags = np.arange(-max_lag, max_lag + 1)
    corr = []
    for l in lags:
        if l >= 0:
            a = x[l:]; b = y[:len(y)-l]
        else:
            a = x[:len(x)+l]; b = y[-l:]
        if len(a) == 0 or len(b) == 0:
            corr.append(-np.inf); continue
        denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
        corr.append(float(np.dot(a, b) / denom))
    corr = np.asarray(corr, dtype=float)

    # 整数栅格最优
    k = int(np.argmax(corr))
    best_lag_int = float(lags[k])
    best_corr = float(corr[k])

    # 亚采样插值：要求不在边界，且相邻两侧有限
    if 0 < k < len(lags) - 1 and np.isfinite(corr[k-1]) and np.isfinite(corr[k+1]):
        y1, y2, y3 = corr[k-1], corr[k], corr[k+1]
        denom = (y1 - 2.0*y2 + y3)
        if abs(denom) > 1e-12:
            delta = 0.5 * (y1 - y3) / denom   # 顶点相对中点的偏移量（单位=采样点）
            best_lag = best_lag_int + float(delta)
            best_corr = float(y2 - 0.25 * (y1 - y3) * delta)  # 顶点处相关近似
            return best_lag, best_corr

    return best_lag_int, best_corr

def stitch_windows(windows: np.ndarray) -> np.ndarray:
    """
    将 [N, H] 的多步窗口拼接成一条连续序列；重叠处做简单平均。
    """
    N, H = windows.shape
    L = N + H - 1
    acc = np.zeros(L, dtype=float)
    cnt = np.zeros(L, dtype=float)
    for i in range(N):
        acc[i:i+H] += windows[i]
        cnt[i:i+H] += 1.0
    return acc / np.maximum(cnt, 1.0)


# ---------------- Data ----------------
class MRDamperData:
    """
    - 历史特征/历史 y：StandardScaler（仅作为输入）
    - 目标：z-score（支持 'level' 多步 or 'delta' 多步）
    - 注入：t+H 的外生控制量（及其与 t 的差），增强可辨识性
    - 划分：按 block 随机分块，避免时间泄漏
    """
    def __init__(
        self,
        csv_path: str,
        feature_cols: List[str],
        target_col: str,
        future_exog_cols: Optional[List[str]] = None,   # 注入的外生控制量列名（在 t+H）
        seq_len: int = 50,
        horizon: int = 3,                    # 预测步长 H，同时也是输出维度
        predict_mode: str = "level",         # "level": 直接预测 y；"delta": 预测多步增量
        test_size: float = 0.2,
        val_size: float = 0.1,
        rng_seed: int = 42,
        use_diff_features: bool = True,      # 历史一阶差分通道
        include_future_exog: bool = True,    # 注入 t+H 的外生控制量
        block_size: int = 500,
        device: torch.device = torch.device("cpu"),
    ):
        assert predict_mode in ("level", "delta")
        self.csv_path   = csv_path
        self.feature_cols = list(feature_cols)
        self.target_col = target_col
        self.seq_len    = seq_len
        self.horizon    = horizon
        self.predict_mode = predict_mode
        self.test_size  = test_size
        self.val_size   = val_size
        self.rng_seed   = rng_seed
        self.use_diff_features = use_diff_features
        self.include_future_exog = include_future_exog
        self.block_size = block_size
        self.device     = device

        if future_exog_cols is None:
            future_exog_cols = ["Commanded Displacement", "Commanded Current"]
        self.future_exog_cols = [c for c in future_exog_cols if c in self.feature_cols]
        self.exog_idx = None

        self.feature_scaler = StandardScaler()
        self.pasty_scaler   = StandardScaler()
        self.target_scaler  = StandardScaler()   # 针对目标（level 或 delta）

        self._load()
        self._build_and_split()

    def _load(self):
        df = pd.read_csv(self.csv_path)
        need = self.feature_cols + [self.target_col]
        miss = [c for c in need if c not in df.columns]
        if miss: raise ValueError(f"CSV 缺少列: {miss}\n现有列: {list(df.columns)}")

        feat = df[self.feature_cols].apply(pd.to_numeric, errors="coerce")
        tgt  = df[[self.target_col]].apply(pd.to_numeric, errors="coerce")
        self.df = pd.concat([feat, tgt], axis=1).dropna(axis=0).reset_index(drop=True)

        self.X_all = self.df[self.feature_cols].values  # [T,F]
        self.y_all = self.df[[self.target_col]].values  # [T,1]
        self.exog_idx = [self.feature_cols.index(c) for c in self.future_exog_cols]

        print(f"[INFO] Loaded: {self.df.shape}")
        print(f"[INFO] y range: ({self.y_all.min():.6f}, {self.y_all.max():.6f})  std: {self.y_all.std():.6f}")

    def _sliding(self, X: np.ndarray, y: np.ndarray):
        """
        返回：
          - 历史特征序列 X_list: [N, L, F]
          - 历史 y 序列      PY_list: [N, L, 1]
          - 未来目标向量     Y_list: [N, H]  （level 模式直接是 y；delta 模式后续再处理）
          - t+H 的完整特征   Xnext_list: [N, F]
        """
        L, H, T = self.seq_len, self.horizon, len(X)
        X_list, PY_list, Y_list, Xnext_list = [], [], [], []
        for i in range(T - L - H + 1):
            X_list.append(X[i:i+L, :])                 # 历史特征窗口
            PY_list.append(y[i:i+L, 0:1])              # 历史 y
            Y_list.append(y[i+L:i+L+H, 0])             # 未来 H 步 y
            Xnext_list.append(X[i+L+H-1, :])           # t+H 的完整特征
        return (np.asarray(X_list),
                np.asarray(PY_list),
                np.asarray(Y_list),        # [N, H]
                np.asarray(Xnext_list))    # [N, F]

    def _build_and_split(self):
        X_feat_raw, pasty_raw, Y_future_raw, Xnext_full_raw = self._sliding(self.X_all, self.y_all)
        N, F = X_feat_raw.shape[0], X_feat_raw.shape[-1]
        H = self.horizon

        # ---- Block-level split ----
        num_blocks = int(math.ceil(N / self.block_size))
        rng = np.random.default_rng(self.rng_seed)
        blocks = np.arange(num_blocks); rng.shuffle(blocks)
        test_B = int(np.floor(num_blocks * self.test_size))
        val_B  = int(np.floor((num_blocks - test_B) * self.val_size))
        tr_blocks = blocks[:num_blocks - test_B - val_B]
        va_blocks = blocks[num_blocks - test_B - val_B : num_blocks - test_B]
        te_blocks = blocks[num_blocks - test_B :]

        def to_idx(blks):
            ids=[]
            for b in blks:
                s=b*self.block_size; e=min((b+1)*self.block_size, N); ids.extend(range(s,e))
            return np.array(ids, dtype=int)

        tr_idx, val_idx, te_idx = to_idx(tr_blocks), to_idx(va_blocks), to_idx(te_blocks)
        print(f"[SPLIT-BLOCK] train:{len(tr_idx)}  val:{len(val_idx)}  test:{len(te_idx)} (block_size={self.block_size})")

        # 只在训练集拟合 scaler
        self.feature_scaler.fit(X_feat_raw[tr_idx].reshape(-1, F))
        self.pasty_scaler.fit(pasty_raw[tr_idx].reshape(-1, 1))

        def build(idxs):
            Xf, PY = X_feat_raw[idxs], pasty_raw[idxs]              # [N,L,F], [N,L,1]
            Yf     = Y_future_raw[idxs]                             # [N,H]
            Xn_full= Xnext_full_raw[idxs]                           # [N,F] —— t+H 全量特征

            # 标准化
            Xf_s = self.feature_scaler.transform(Xf.reshape(-1, F)).reshape(Xf.shape)
            PY_s = self.pasty_scaler.transform(PY.reshape(-1, 1)).reshape(PY.shape)
            Xn_full_s = self.feature_scaler.transform(Xn_full)      # [N,F]

            # 历史差分通道
            feats = [Xf_s, PY_s]
            if self.use_diff_features:
                dXf_s = np.diff(Xf_s, axis=1, prepend=Xf_s[:, :1, :])
                dPY_s = np.diff(PY_s, axis=1, prepend=PY_s[:, :1, :])
                feats.extend([dXf_s, dPY_s])

            # t+H 外生通道（已知控制量）：数值 & 相对最后一步的差
            if self.include_future_exog and len(self.exog_idx) > 0:
                Xn_exog_s = Xn_full_s[:, self.exog_idx]                 # [N,E]
                last_exog = Xf_s[:, -1, :][:, self.exog_idx]            # [N,E]
                dXn_exog  = Xn_exog_s - last_exog                        # [N,E]
                rep = Xf_s.shape[1]
                feats.extend([
                    np.repeat(Xn_exog_s[:, None, :], rep, axis=1),      # [N,L,E]
                    np.repeat(dXn_exog[:, None, :], rep, axis=1)        # [N,L,E]
                ])

            X_in = np.concatenate(feats, axis=2)                        # [N,L,C]

            # 目标构造
            last_raw = PY[:, -1, 0:1]                                   # [N,1] = y_t
            if self.predict_mode == "level":
                y_model_raw = Yf                                        # [N,H]
                target_fit  = y_model_raw
            else:  # "delta": 多步增量 Δ1=y_{t+1}-y_t, Δ2=y_{t+2}-y_{t+1}, ...
                prev = np.concatenate([last_raw, Yf[:, :-1]], axis=1)   # [N,H]
                y_model_raw = Yf - prev                                 # [N,H]
                target_fit  = y_model_raw

            return X_in, y_model_raw, last_raw, target_fit, Yf  # Yf: 物理单位（GT）

        Xtr, ytr_raw, last_tr, target_tr, _     = build(tr_idx)
        Xva, yva_raw, last_va, target_va, _     = build(val_idx)
        Xte, yte_raw, last_te, target_te, Yte   = build(te_idx)

        # 目标标准化（只用训练集）
        self.target_scaler.fit(target_tr.reshape(-1, 1))
        ytr_z = self.target_scaler.transform(ytr_raw.reshape(-1, 1)).reshape(ytr_raw.shape)
        yva_z = self.target_scaler.transform(yva_raw.reshape(-1, 1)).reshape(yva_raw.shape)
        yte_z = self.target_scaler.transform(yte_raw.reshape(-1, 1)).reshape(yte_raw.shape)

        print(f"[CHECK] model target z train std:{ytr_z.std():.4f}  val std:{yva_z.std():.4f}")

        # -> torch
        self.X_train      = torch.tensor(Xtr,   dtype=torch.float32, device=self.device)
        self.y_train_model= torch.tensor(ytr_z, dtype=torch.float32, device=self.device)
        self.last_train   = torch.tensor(last_tr,dtype=torch.float32, device=self.device)

        self.X_val        = torch.tensor(Xva,   dtype=torch.float32, device=self.device)
        self.y_val_model  = torch.tensor(yva_z, dtype=torch.float32, device=self.device)
        self.last_val     = torch.tensor(last_va,dtype=torch.float32, device=self.device)

        self.X_test       = torch.tensor(Xte,   dtype=torch.float32, device=self.device)
        self.y_test_model = torch.tensor(yte_z, dtype=torch.float32, device=self.device)
        self.last_test    = torch.tensor(last_te,dtype=torch.float32, device=self.device)

        # 物理单位真值（用于评估/作图）：y_{t+1..t+H}
        self.Y_test_phys  = torch.tensor(Yte, dtype=torch.float32, device=self.device)

        # ---- Baselines (z-space) ----
        zero_mse = float(np.mean(yva_z ** 2))
        if self.predict_mode == "level":
            persist_last = np.repeat(self.last_val.cpu().numpy(), H, axis=1)  # [N,H]
            persist_z_val = self.target_scaler.transform(persist_last.reshape(-1,1)).reshape(persist_last.shape)
        else:  # delta
            persist_z_val = np.zeros_like(yva_z)
        persist_mse = float(np.mean((yva_z - persist_z_val) ** 2))
        print(f"[SANITY] zero-baseline MSE on val (z): {zero_mse:.4f}")
        print(f"[SANITY] persistence-baseline MSE on val (z): {persist_mse:.4f}")

        # 训练采样权重：窗口内最大幅值 + 事件放大（前20%）
        strength = np.max(np.abs(ytr_z), axis=1)       # [N_train]
        thr = np.percentile(strength, 80)
        w = 1.0 + 4.0 * (strength / (strength.mean() + 1e-8)) + 10.0 * (strength > thr)
        self.train_weights = torch.tensor(w, dtype=torch.double, device="cpu")

    def to_physical(self, y_model_z: np.ndarray, last_phys: np.ndarray) -> np.ndarray:
        """
        z → 物理单位；支持多步 delta 还原（累加）
        y_model_z: [N,H], last_phys: [N,1]
        return: 预测的 y_{t+1..t+H}（物理单位）: [N,H]
        """
        target_phys = self.target_scaler.inverse_transform(y_model_z.reshape(-1,1)).reshape(y_model_z.shape)
        if self.predict_mode == "level":
            return target_phys
        else:  # delta: 累加增量
            N, H = target_phys.shape
            ys = []
            cur = last_phys.copy()
            for h in range(H):
                cur = cur + target_phys[:, h:h+1]
                ys.append(cur)
            return np.concatenate(ys, axis=1)

    @property
    def input_size(self):
        return self.X_train.shape[-1]  # 通道数

# ---------------- Model ----------------
class GRUHead(nn.Module):
    """GRU head（没有 Attention）"""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, horizon: int, dropout: float = 0.0):
        super().__init__()
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size + input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, horizon)
        )

    def forward(self, x):
        B = x.size(0)
        h0 = torch.zeros(self.num_layers, B, self.hidden_size, device=x.device)
        g_out, _ = self.gru(x, h0)
        last_inp = x[:, -1, :]
        out = self.head(torch.cat([g_out[:, -1, :], last_inp], dim=-1))  # [B,H]

        # 返回两个值，第一个是预测值，第二个是dummy值（None）
        return out, None  # 这里第二个返回值是None，您可以根据实际需要调整


# ---------------- Trainer ----------------
class Trainer:
    def __init__(
        self,
        model,
        lr=3e-3,
        weight_decay=1e-4,
        grad_clip=5.0,
        device=None,
        use_event_weight: bool = True,   # 事件加权 MSE（可关）
        weight_alpha: float = 3.0,
        weight_gamma: float = 1.0,
        weight_cap: float = 10.0,
    ):
        self.model = model
        self.device = device or torch.device("cpu")
        self.crit = nn.MSELoss(reduction="none")  # 我们手动做加权；日志统一用未加权
        self.opt  = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.grad_clip = grad_clip
        self.max_lr = lr  # for OneCycle
        self.use_event_weight = use_event_weight
        self.weight_alpha = weight_alpha
        self.weight_gamma = weight_gamma
        self.weight_cap   = weight_cap

    def _weighted_loss(self, pr, yb):
        if not self.use_event_weight:
            return ((pr - yb) ** 2).mean()
        se = (pr - yb) ** 2                       # [B,H]
        w  = 1.0 + self.weight_alpha * (yb.abs() ** self.weight_gamma)
        w  = w / (w.mean() + 1e-8)
        if self.weight_cap is not None:
            w = torch.clamp(w, max=self.weight_cap)
        return (w * se).mean()

    @torch.no_grad()
    def _epoch_mse(self, loader):
        self.model.eval(); mse_sum, n = 0.0, 0
        for xb, _, yb in loader:
            pr,_ = self.model(xb)
            mse_sum += torch.mean((pr - yb) ** 2).item(); n += 1
        return mse_sum / max(1,n)

    def fit(self, train_loader, val_loader, epochs=100, debug_std=False):
        steps = max(1, len(train_loader))
        sched = optim.lr_scheduler.OneCycleLR(
            self.opt, max_lr=self.max_lr, epochs=epochs, steps_per_epoch=steps,
            pct_start=0.25, div_factor=10.0, final_div_factor=10.0
        )

        tr_hist, va_hist = [], []
        for ep in range(1, epochs+1):
            self.model.train()
            for xb, _, yb in train_loader:
                self.opt.zero_grad()
                pr,_ = self.model(xb)
                loss = self._weighted_loss(pr, yb)  # ← 训练用加权/不加权
                loss.backward()
                if self.grad_clip is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                self.opt.step(); sched.step()

            train_mse = self._epoch_mse(train_loader)  # 日志口径：未加权 z-MSE
            val_mse   = self._epoch_mse(val_loader)
            tr_hist.append(train_mse); va_hist.append(val_mse)

            if debug_std:
                xb, _, yb = next(iter(train_loader))
                with torch.no_grad():
                    pr,_ = self.model(xb)
                    print(f"[DBG] pred std:{pr.std().item():.3f}  target std:{yb.std().item():.3f}")

            print(f"Epoch[{ep}/{epochs}] train_loss:{train_mse:.4f} | val_loss:{val_mse:.4f} | lr:{self.opt.param_groups[0]['lr']:.5f}")

        return tr_hist, va_hist

    @torch.no_grad()
    def predict_phys(self, loader, to_physical_fn):
        self.model.eval(); preds_phys, trues_phys = [], []
        for xb, lastb, yb in loader:
            head,_ = self.model(xb)                                # z, [B,H]
            y_hat_phys = to_physical_fn(head.cpu().numpy(), lastb.cpu().numpy())  # [B,H]
            preds_phys.append(y_hat_phys)
            y_true_phys = to_physical_fn(yb.cpu().numpy(),  lastb.cpu().numpy())  # [B,H]
            trues_phys.append(y_true_phys)
        return np.concatenate(preds_phys,0), np.concatenate(trues_phys,0)


# ---------------- Plot ----------------
def plot_all(tr_loss, va_loss, trues, preds, base, outdir="figs", interval=500):
    """
    修改后的 plot_all 函数，增加了 interval 参数，用于调整时间步长的间隔
    """
    os.makedirs(outdir, exist_ok=True)
    trues = np.asarray(trues); preds = np.asarray(preds); base = np.asarray(base)

    # 0) Loss（z-space/MSE）
    plt.figure(figsize=(10,5))
    plt.plot(tr_loss, label="train_loss"); plt.plot(va_loss, label="val_loss")
    plt.title("Training & Validation Loss (z-MSE)"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "loss_curve.png"), dpi=150)

    # 1) 全量展开（注意包含重叠）
    plt.figure(figsize=(12,5))
    idx = np.linspace(0, trues.size-1, min(1000, trues.size), dtype=int)
    # 使用 interval 参数调整展示的时间间隔
    interval_idx = np.arange(0, len(trues), interval)
    plt.plot(interval_idx, trues.reshape(-1)[interval_idx], linestyle="--", linewidth=2, label="Actual")
    plt.plot(interval_idx, preds.reshape(-1)[interval_idx], linewidth=1, label="Predicted")
    plt.title("MR Damper Prediction (Test) [Measured Current]"); plt.xlabel("Time Steps"); plt.ylabel("Measured Current (A)"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "pred_vs_true.png"), dpi=150)

    # 2) 只看一步（h=1）
    plt.figure(figsize=(12,4))
    plt.plot(trues[:,0], "--", label="Actual(h=1)")
    plt.plot(preds[:,0], label="Pred(h=1)")
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(outdir, "step1_only.png"), dpi=150)

    # 3) 简易“去重叠平均”可视化（每 H 个窗口取一条，近似）
    H = preds.shape[1]
    p_avg = preds[::H, 0] if preds.shape[0] >= H else preds[:,0]
    t_avg = trues[::H, 0] if trues.shape[0] >= H else trues[:,0]
    plt.figure(figsize=(12,4))
    plt.plot(t_avg, "--", label="Actual(avg)")
    plt.plot(p_avg, label="Pred(avg)")
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(outdir, "avg_track.png"), dpi=150)

    # 4) 残差（展开）
    res = trues.reshape(-1) - preds.reshape(-1)
    plt.figure(figsize=(12,5))
    plt.plot(res); plt.axhline(0, linestyle="--")
    plt.title("Residuals (Actual - Predicted)"); plt.xlabel("Time Steps"); plt.ylabel("Residual (A)")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "residuals.png"), dpi=150)

    # 5) 持久化基线
    plt.figure(figsize=(12,5))
    idx = np.linspace(0, trues.size-1, min(1000, trues.size), dtype=int)
    plt.plot(idx, trues.reshape(-1)[idx], linestyle="--", label="Actual")
    plt.plot(idx, base.reshape(-1)[idx], label="Persistence")
    plt.title("Persistence Baseline (Test)"); plt.xlabel("Time Steps"); plt.ylabel("Measured Current (A)"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "persist_vs_true.png"), dpi=150)

    print(f"[Saved] {outdir}/loss_curve.png")
    print(f"[Saved] {outdir}/pred_vs_true.png")
    print(f"[Saved] {outdir}/step1_only.png")
    print(f"[Saved] {outdir}/avg_track.png")
    print(f"[Saved] {outdir}/residuals.png")
    print(f"[Saved] {outdir}/persist_vs_true.png")


# ---------------- Main ----------------
def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === 改成你的 CSV 路径 ===
    csv_path = "D:/desktop/杂/个人有关/毕业论文/倪浩君大论文/force_control_dataset（符合理想）.csv"

    # 关键：包含控制量（外生输入）
    feature_cols = ["F", "displacement"]
    target_col   = "i_cmd"

    # —— 推荐设置 ——
    seq_len, horizon = 50, 3      # 多步预测
    predict_mode = "level"        # "level" 或 "delta"
    include_future_exog = True    # 建议 True；需要严格因果就改 False
    use_diff_features   = True
    block_size = 500

    # 训练超参
    batch_size = 128
    hidden_size, num_layers = 256, 2
    epochs = 50
    lr = 1e-4
    weight_decay = 1e-4
    grad_clip = 5.0
    dropout = 0.0

    # 采样周期 Δt（秒/点）。如果你的采样频率是 100 Hz，请设为 0.01；请按真实频率修改！
    sampling_period: Optional[float] = 0.01
    # 最大滞后搜索窗口（采样点）
    delay_max_lag = 500

    data = MRDamperData(
        csv_path, feature_cols, target_col,
        future_exog_cols=["force"],
        seq_len=seq_len, horizon=horizon,
        predict_mode=predict_mode,
        test_size=0.2, val_size=0.1,
        rng_seed=42,
        use_diff_features=use_diff_features,
        include_future_exog=include_future_exog,
        block_size=block_size,
        device=device
    )

    # Datasets / Loaders（注意 y_*_model 是 z-space 多步目标）
    label_shuffle = False  # ← “泄漏否定测试”时设 True
    if label_shuffle:
        perm = torch.randperm(data.y_train_model.shape[0])
        y_train = data.y_train_model[perm]
    else:
        y_train = data.y_train_model

    train_ds = TensorDataset(data.X_train, data.last_train, y_train)
    val_ds   = TensorDataset(data.X_val,   data.last_val,  data.y_val_model)
    test_ds  = TensorDataset(data.X_test,  data.last_test, data.y_test_model)

    # 训练采样器（不改变损失标尺，只影响抽样次序）
    sampler = WeightedRandomSampler(weights=data.train_weights.cpu(), num_samples=len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False)

    model = GRUHead(
        input_size=data.input_size, hidden_size=hidden_size,
        num_layers=num_layers, horizon=horizon, dropout=dropout
    ).to(device)

    trainer = Trainer(
        model, lr=lr, weight_decay=weight_decay, grad_clip=grad_clip, device=device,
        use_event_weight=True, weight_alpha=3.0, weight_gamma=1.0, weight_cap=10.0
    )

    # 训练（日志使用未加权 z-MSE）
    tr_loss, va_loss = trainer.fit(train_loader, val_loader, epochs=epochs, debug_std=False)

    # —— 诊断：看测试集 z 空间的方差与 MSE ——
    @torch.no_grad()
    def debug_test_stats(loader, model_):
        P, T = [], []
        for xb, _, yb in loader:
            pr, _ = model_(xb)
            P.append(pr.cpu().numpy()); T.append(yb.cpu().numpy())
        P = np.concatenate(P); T = np.concatenate(T)
        print(f"[TEST/Z] pred std={P.std():.3f}, target std={T.std():.3f}, MSE={np.mean((P-T)**2):.4f}")
    debug_test_stats(test_loader, model)

    # 测试集预测（物理单位，多步）
    to_phys = lambda yh, last: data.to_physical(yh, last)
    preds_phys, trues_phys = trainer.predict_phys(test_loader, to_phys)
    assert preds_phys.shape == trues_phys.shape, f"shape mismatch: {preds_phys.shape} vs {trues_phys.shape}"

    # === 信号时延估计（基于去重叠平均后的连续轨迹，亚采样精度） ===
    true_track = stitch_windows(trues_phys)   # [T_total]
    pred_track = stitch_windows(preds_phys)   # [T_total]
    lag, corr = estimate_delay(true_track, pred_track, max_lag=delay_max_lag)
    print(f"[DELAY] 估计滞后: {lag:.3f} 个采样点（相关={corr:.3f}）")
    if sampling_period is not None:
        print(f"[DELAY] 折算为时间: ~{lag * sampling_period:.6f} s")

    # （可选）只看一步(h=1)的滞后
    t_h1 = trues_phys[:, 0]
    p_h1 = preds_phys[:, 0]
    lag1, corr1 = estimate_delay(t_h1, p_h1, max_lag=delay_max_lag)
    print(f"[DELAY-h=1] 估计滞后: {lag1:.3f} 个采样点（相关={corr1:.3f}）")
    if sampling_period is not None:
        print(f"[DELAY-h=1] 折算为时间: ~{lag1 * sampling_period:.6f} s")

    # 基线（物理单位；持久化基线：把 y_t 持久化为 y_{t+1..t+H}）
    baseline_phys = persistence_baseline(data.last_test.cpu().numpy(), horizon)

    print("\n===== Test Metrics (physical units) =====")
    metrics(trues_phys, preds_phys,   name="[Model   ]")
    metrics(trues_phys, baseline_phys,name="[Persist ]")

    # 作图（含 step1_only / avg_track）——（未做新的“对齐图”，仅保留原有图）
    plot_all(tr_loss, va_loss, trues_phys, preds_phys, baseline_phys, outdir="figs_GRU_MRB", interval=100)


if __name__ == "__main__":
    main()
