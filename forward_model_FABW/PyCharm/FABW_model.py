import numpy as np
import matplotlib
matplotlib.use("Agg")  # 防止 PyCharm 后端 show() 崩
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter
import os


class FractionalAsymmetricBoucWenPhysical:
    """
    纯物理改进Bouc-Wen (FAN-BW思想):
    - z(t): 内部滞回状态
    - 分数阶记忆: D^alpha i(t)
    - 非对称修正: delta * tanh(g*z)*|dz/dt|
    - 温度补偿 (线性)
    """

    def __init__(
        self,
        # Bouc-Wen核参数
        A=1.0,
        beta=0.8,
        gamma=0.5,
        n=2.0,

        # 输出力通道增益 (缩小到比较小的量级)
        k1=1.0,        # 电流直接贡献
        k2=0.1,        # di/dt 贡献
        alpha_z=0.5,   # 滞回状态贡献

        # 分数阶
        alpha_frac=0.6,  # 0<alpha<1
        frac_gain=0.1,   # 分数阶导项权重，减小

        # 非对称
        delta_asym=0.12,
        tanh_gain=5.0,

        # 温度补偿
        temp_coeff=0.002
    ):
        self.A = A
        self.beta = beta
        self.gamma = gamma
        self.n = n

        self.k1 = k1
        self.k2 = k2
        self.alpha_z = alpha_z

        self.alpha_frac = alpha_frac
        self.frac_gain = frac_gain

        self.delta_asym = delta_asym
        self.tanh_gain = tanh_gain

        self.temp_coeff = temp_coeff

    def fractional_derivative(self, x, dt):
        """
        稳定版 Grünwald–Letnikov 分数阶导近似:
        D^α x(t_i) ≈ (1/dt^α) * sum_{j=0}^{i} c_j * x[i-j]

        递推计算 c_j，避免 gamma() 引起的inf/NaN:
            c_0 = 1
            c_j = c_{j-1} * ((j-1 - α)/j)
        其中 α = self.alpha_frac (0<α<1)
        """
        alpha = self.alpha_frac
        N = len(x)
        D = np.zeros_like(x)

        # 预先算出足够长的一组系数 c_j
        c = np.zeros(N)
        c[0] = 1.0
        for j in range(1, N):
            c[j] = c[j-1] * ((j-1 - alpha) / j)

        # 对每个时间点 i，做卷积和
        inv_dt_alpha = (dt ** (-alpha))
        for i in range(1, N):
            # x[i], x[i-1], ..., x[0]
            xi = x[i::-1]      # 长度 i+1
            ci = c[:i+1]       # 对应的 c_0...c_i
            D[i] = inv_dt_alpha * np.dot(ci, xi)

        return D

    def simulate(self, i_signal, t, T_env=25.0):
        dt = t[1] - t[0]

        di_dt = np.gradient(i_signal, dt)
        Di_frac = self.fractional_derivative(i_signal, dt)

        N = len(i_signal)
        z = np.zeros(N)
        F_raw = np.zeros(N)

        for k in range(1, N):
            z_prev = z[k-1]
            z_abs_prev = abs(z_prev)

            # 在计算时加入更多的随机扰动，以增加不规则性
            k1_perturbed = self.k1 * (1 + np.random.normal(0, 0.15))  # 加强10-15%的随机噪声
            k2_perturbed = self.k2 * (1 + np.random.normal(0, 0.1))

            # 经典Bouc-Wen内核
            core_term = (
                k1_perturbed * di_dt[k]
                - self.beta * abs(di_dt[k]) * (z_abs_prev ** (self.n - 1)) * z_prev
                - self.gamma * di_dt[k]      * (z_abs_prev ** self.n)
            )

            # 引入扰动来增加更多的非线性效果
            frac_term = self.frac_gain * Di_frac[k] * (1 + 0.2 * np.random.randn())  # 增加不规则性

            dz_dt_raw = core_term + frac_term

            # 更强的非对称修正，使环形状更加不规则
            asym_term = (
                self.delta_asym
                * np.tanh(self.tanh_gain * z_prev)
                * abs(dz_dt_raw)
            )

            dz_dt = dz_dt_raw + asym_term

            # 前向欧拉积分
            z[k] = z_prev + dz_dt * dt

            # 力输出（还没归一化）
            base_force = (
                k1_perturbed * i_signal[k]
                + k2_perturbed * di_dt[k]
                + self.alpha_z * z[k]
            )

            # 温度补偿
            temp_scale = 1.0 - self.temp_coeff * (T_env - 25.0)
            F_raw[k] = base_force * temp_scale

        return {
            "i": i_signal,
            "F_raw": F_raw,
            "z": z,
            "di_dt": di_dt,
            "Di_frac": Di_frac
        }

def build_sine_current(Imin, Imax, freq=1.0, total_time=2.0, dt=0.001):
    """
    生成 i(t)=I_mid+I_amp*sin(2πft)
    """
    I_mid = 0.5*(Imin + Imax)
    I_amp = 0.5*(Imax - Imin)

    t = np.arange(0.0, total_time, dt)
    i_signal = I_mid + I_amp * np.sin(2*np.pi*freq*t)
    return t, i_signal


def extract_steady_state(t, i_arr, F_arr, keep_after_ratio=0.5):
    """
    丢掉前半段(第一圈)，只保留后半段，当作稳态磁滞环
    """
    N = len(t)
    start_idx = int(N * keep_after_ratio)
    return i_arr[start_idx:], F_arr[start_idx:]


def normalize_force_to_range(F_arr, Fmin_target=0.0, Fmax_target=3.24):
    """
    对每个环进行独立归一化，避免不符合预期的范围
    """
    F_min = np.min(F_arr)
    F_max = np.max(F_arr)
    if F_max - F_min < 1e-9:
        return np.zeros_like(F_arr)

    F_norm = (F_arr - F_min) / (F_max - F_min)  # -> [0,1]
    F_scaled = Fmin_target + F_norm * (Fmax_target - Fmin_target)
    return F_scaled


def smooth_force(F_arr, window_length=51, polyorder=3):
    """
    使用 Savitzky-Golay 滤波器平滑力数据，减少噪声
    """
    return savgol_filter(F_arr, window_length=window_length, polyorder=polyorder)


def save_loop_csv(filename, current, force):
    """
    保存两个通道到CSV
    """
    df = pd.DataFrame({
        "Current_A": current,
        "Force_N": force
    })
    df.to_csv(filename, index=False)
    print(f"[Saved] {filename} ({len(df)} samples)")


def main():
    # -----------------------
    # 1. 模型参数 (物理/经验)
    # -----------------------
    model = FractionalAsymmetricBoucWenPhysical(
        A=1.0,
        beta=0.8,
        gamma=0.5,
        n=2.0,
        k1=1.0,        # 减小增益，避免力太大
        k2=0.1,
        alpha_z=0.5,
        alpha_frac=0.6,
        frac_gain=0.1,
        delta_asym=0.12,
        tanh_gain=5.0,
        temp_coeff=0.002
    )

    # -----------------------
    # 2. 三条电流扫幅 (指定范围)
    # -----------------------
    loops = [
        {"name": "Loop_0p0_1p0", "Imin":0.0, "Imax":1.0, "color":"red",
         "label":"Loop 1 (0.0A-1.0A)"},
        {"name": "Loop_0p2_0p8", "Imin":0.2, "Imax":0.8, "color":"orange",
         "label":"Loop 2 (0.2A-0.8A)"},
        {"name": "Loop_0p4_0p6", "Imin":0.4, "Imax":0.6, "color":"green",
         "label":"Loop 3 (0.4A-0.6A)"},
    ]

    freq = 1.0        # Hz
    total_time = 2.0  # s, 两个周期
    dt = 0.001
    T_env = 25.0      # °C

    # -----------------------
    # 3. 绘图
    # -----------------------
    fig, ax = plt.subplots(figsize=(8,4.5))
    ax.set_xlabel("Current (A)")
    ax.set_ylabel("Force (N)")
    ax.set_title("MRF Hysteresis Loops (Physical FAN-BW Model, normalized 0-3.24N)")
    ax.grid(True, alpha=0.3)

    # 竖线(可选)做视觉参考，比如 0.5A
    ax.axvline(x=0.5, linestyle='--', color='k', linewidth=1)

    for lp in loops:
        # (a) 生成该电流范围的激励
        t, i_sig = build_sine_current(
            Imin=lp["Imin"],
            Imax=lp["Imax"],
            freq=freq,
            total_time=total_time,
            dt=dt
        )

        # (b) 物理模型仿真
        sim_res = model.simulate(i_sig, t, T_env=T_env)
        i_full = sim_res["i"]
        F_full = sim_res["F_raw"]

        # (c) 丢掉第一圈，取稳态
        i_ss, F_ss = extract_steady_state(t, i_full, F_full, keep_after_ratio=0.5)

        # (d) 把力缩放到 [0, 3.24N]，每个环独立归一化
        F_scaled = normalize_force_to_range(F_ss, 0.0, 3.24)

        # (e) 平滑处理
        F_smoothed = smooth_force(F_scaled)

        # (f) 保存CSV (稳态段)
        csv_name = lp["name"] + ".csv"
        save_loop_csv(csv_name, i_ss, F_smoothed)

        # (g) 为了画“回线”，我们把稳态段拆成上升+下降两段
        mid = len(i_ss) // 2
        i_up,   F_up   = i_ss[:mid],   F_smoothed[:mid]
        i_down, F_down = i_ss[mid:],   F_smoothed[mid:]

        ax.plot(i_up,   F_up,   '-', linewidth=2, color=lp["color"], label=lp["label"])
        ax.plot(i_down, F_down, '-', linewidth=2, color=lp["color"])

    # 统一坐标
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 3.24)

    ax.legend(loc='upper left')
    plt.tight_layout()

    # 只保存，不 plt.show()，避免 PyCharm interagg 崩
    out_fig = "mrf_hysteresis_physical_smoothed_nested_final.png"
    plt.savefig(out_fig, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {out_fig}")


if __name__ == "__main__":
    os.makedirs(".", exist_ok=True)
    main()
