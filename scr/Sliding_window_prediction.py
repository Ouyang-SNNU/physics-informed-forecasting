"""
UDE Prediction v2 — TCN 架构 + PINNs 微调 + 滑动窗口预测

Based on UDE_Pred_v1.py, with added sliding window prediction mode.

新增功能:
  - 从第 5 周开始（前 5 周数据已用于 PINNs 校准），以滑动窗口逐段预测
  - 每次预测后续连续的 4 个时间步（4 周）
  - 预测完成后窗口向后滑动 1 个时间步（1 周）
  - 重复直至预测窗口完整覆盖数据末尾
  - ODE 按天计算，绘图按周输出，仅绘制 inc 变量
  - 每 4 周触发一次滑动窗口阶段性微调（基于已知周发病率 MSE loss）
  - 微调权重继承上一次微调结果，不重新加载初始权重
  - 每个滑动窗口的预测结果可视化 (test.py 风格, 含 erfinv 置信区间)

原有流程 (完整保留):
  1. 加载 S2_v1 模型 ({Region}_UDE_S2_v1.pt)
  2. 用 PINNs β (前 L_pinns 天) 对模型做少量轮次微调 (fine_tune_on_pinns)
  3. 微调后模型预测全季 β → S-G 后处理平滑
  4. 从 PINNs 初始条件 (S0, E0, Is0, Ia0) 出发做 ODE 积分
  5. 三面板可视化 (beta / incidence / delta-beta)

输出: figures/UDE_Pred_{region}_{yr}_sliding.png
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import joblib
from datetime import date
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from Transmission_network import TransNet, CLIMATE_WINDOW

torch.manual_seed(1234); np.random.seed(1234)
torch.set_default_dtype(torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"CLIMATE_WINDOW = {CLIMATE_WINDOW} days")

# ====================== 预测配置 ======================
PREDICT_CONFIG = {
    # "Abruzzo":                  [2025],
    # "Apulia":                   [2025],
    # "Campania":                 [2025],
    # "Emilia-Romagna":           [2025],
    # "Friuli-Venezia_Giulia":    [2025],
    # "Lazio":                    [2025],
    # "Liguria":                  [2025],
    # "Lombardy":                 [2025],
    # "Marche":                   [2025],
    # "Piedmont":                 [2025],
    # "Sardinia":                 [2025],
    # "Sicily":                   [2025],
    # "Tuscany":                  [2025],
    # "Umbria":                   [2025],
    "Veneto":                   [2025],
}

# ====================== PINNs 微调配置 ======================
FINETUNE_CONFIG = {
    "enabled": True,
    "epochs": 200,            # 微调轮次 (少量)
    "lr": 1e-5,               # 微调学习率 (低, 防止灾难遗忘)
    "print_every": 50,        # 每 N 轮打印 loss
}
# ===========================================================

SMOOTH_CONFIG = {
    "method": "savgol",
    "savgol_window": 7,
    "savgol_polyorder": 3,
    "gaussian_sigma": 1.0,
}

# ====================== 滑动窗口配置 ======================
SLIDING_CONFIG = {
    "window_weeks": 4,        # 每次预测后续 4 周
    "start_week": 5,          # 从第 5 周开始 (前 5 周已用于 PINNs 校准)
    "slide_step": 1,          # 窗口每次向后滑动 1 周
    "max_subplots": 30,       # 可视化中最多展示的子图数量
}
# ===========================================================

# ====================== 滑动窗口阶段性微调配置 ======================
SLIDING_FINETUNE_CONFIG = {
    "enabled": True,
    "interval_weeks": 1,
    "epochs": 50,             # 每次微调轮次
    "lr": 5e-6,               # 微调学习率
    "print_every": 25,        # 每 N 轮打印 loss
}
# ===================================================================

PARAM_COLS = ["temperature_c", "absolute_humidity"]

SIGMA = 0.67
GAMMA_S = 0.20
GAMMA_A = 0.33
DELTA = 0.60
THETA = 0.40

CLIMATE_DIR = "../data/processed_climate_data"
PINNS_DIR = "../data/PINNs_results"
Epi_DIR = "../data/ILI_data"
SAVE_DIR = "../model"
FIG_DIR = "../figures"

# 全局绘图风格（与 plt/plot_PINNs_by_region.py 保持一致）
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'legend.frameon': False,
    'figure.dpi': 600,
})


# ====================== 工具函数 ======================

def get_season_date_range(incidence_file):
    df = pd.read_csv(incidence_file)
    sy, sw = map(int, df['year_week'].iloc[0].split('-'))
    ey, ew = map(int, df['year_week'].iloc[-1].split('-'))
    return (pd.to_datetime(date.fromisocalendar(sy, sw, 1)),
            pd.to_datetime(date.fromisocalendar(ey, ew, 7)))


def get_prediction_plans():
    plans = {}
    for region, years in PREDICT_CONFIG.items():
        valid = []
        for yr in years:
            pnns = os.path.join(PINNS_DIR, f"{region}_{yr}.csv")
            if os.path.exists(pnns):
                valid.append(yr)
            else:
                print(f"  WARNING {region} {yr}: PINNs file missing, skip")
        if valid:
            plans[region] = sorted(valid)
        else:
            print(f"  WARNING {region}: no valid prediction data, skip")
    return plans


def smooth_beta(beta_np, method="savgol", **kwargs):
    L = len(beta_np)
    if method == "none":
        return beta_np, {"method": "none"}
    elif method == "savgol":
        window = kwargs.get("window", 7)
        polyorder = kwargs.get("polyorder", 3)
        window = min(window, L if L % 2 == 1 else L - 1)
        if window < polyorder + 2:
            window = polyorder + 2
            if window % 2 == 0:
                window += 1
        beta_smooth = savgol_filter(beta_np, window_length=window,
                                    polyorder=polyorder)
        info = {"method": "savgol", "window": window, "polyorder": polyorder}
    elif method == "gaussian":
        sigma = kwargs.get("sigma", 1.0)
        beta_smooth = gaussian_filter1d(beta_np, sigma=sigma)
        info = {"method": "gaussian", "sigma": sigma}
    else:
        raise ValueError(f"Unknown smoothing method: {method}")
    return beta_smooth, info


def compute_smoothness_metrics(beta_np):
    d1 = np.diff(beta_np)
    d2 = np.diff(d1)
    return {
        "rms_delta": float(np.sqrt(np.mean(d1 ** 2))),
        "rms_delta2": float(np.sqrt(np.mean(d2 ** 2))),
        "max_jump": float(np.max(np.abs(d1))),
        "mean_abs_jump": float(np.mean(np.abs(d1))),
        "n_days": len(beta_np),
    }


def forward_euler_single_season(beta_seq, init_cond, max_steps=None):
    """
    前向欧拉积分, 从 init_cond 出发, 按 beta_seq 逐日步进.

    参数:
      beta_seq:   (L,)  β 序列
      init_cond:  [S0, E0, Is0, Ia0]
      max_steps:  int or None.  None → 积分全部 L 天;
                  指定 → 仅积分前 max_steps 天 (用于微调时只算已知段)
    """
    dtype = beta_seq.dtype
    dt = 1.0
    L = len(beta_seq)
    steps = L if max_steps is None else min(max_steps, L)
    inc_list = []
    S, E, Is, Ia = [torch.tensor(v, device=device, dtype=dtype)
                    for v in init_cond]
    for t in range(steps):
        beta = beta_seq[t]
        infection = beta * S * (Is + THETA * Ia)
        S = S + dt * (-infection + GAMMA_S * Is + GAMMA_A * Ia)
        E = E + dt * (infection - SIGMA * E)
        Is = Is + dt * (DELTA * SIGMA * E - GAMMA_S * Is)
        Ia = Ia + dt * ((1 - DELTA) * SIGMA * E - GAMMA_A * Ia)
        inc_list.append(DELTA * SIGMA * E)
    return torch.stack(inc_list)


def build_prediction_data(region, year, clim_df, scaler):
    epi_path = os.path.join(Epi_DIR, f"{region}_{year}.csv")
    pnns_path = os.path.join(PINNS_DIR, f"{region}_{year}.csv")
    if not os.path.exists(pnns_path):
        return None
    if not os.path.exists(epi_path):
        print(f"    WARNING epi file missing: {epi_path}")
        return None

    start, end = get_season_date_range(epi_path)
    L = (end - start).days + 1
    W_season = L // 7

    clim_start = start - pd.Timedelta(days=CLIMATE_WINDOW)
    clim_raw = clim_df.loc[clim_start:end][PARAM_COLS].values.astype(np.float32)
    clim_scaled = scaler.transform(clim_raw)

    if len(clim_scaled) < L + CLIMATE_WINDOW:
        print(f"    WARNING climate data too short: need {L+CLIMATE_WINDOW}, got {len(clim_scaled)}")
        return None

    df_pinns = pd.read_csv(pnns_path)
    beta_pinns_raw = df_pinns['beta'].values.astype(np.float32)
    S0, E0, Is0, Ia0 = df_pinns.iloc[0][['S', 'E', 'Is', 'Ia']].values.astype(np.float32)

    L_pinns = min(len(beta_pinns_raw), L)
    beta_pinns = beta_pinns_raw[:L_pinns]
    if L_pinns < L:
        beta_pinns = np.pad(beta_pinns.astype(np.float64), (0, L - L_pinns),
                           constant_values=np.nan).astype(np.float32)

    inc_obs = None
    if os.path.exists(epi_path):
        df_inc = pd.read_csv(epi_path)
        inc_raw = (df_inc['incidence'].values / 1000.0).astype(np.float32)
        if len(inc_raw) < W_season:
            inc_obs = np.pad(inc_raw.astype(np.float64), (0, W_season - len(inc_raw)),
                            constant_values=np.nan).astype(np.float32)
        else:
            inc_obs = inc_raw[:W_season]

    X_clim, X_seas = [], []
    for d in range(L):
        X_clim.append(clim_scaled[d:d + CLIMATE_WINDOW].flatten())
        current_date = start + pd.Timedelta(days=d)
        doy = current_date.timetuple().tm_yday
        X_seas.append([
            np.sin(2 * np.pi * doy / 365.0),
            np.cos(2 * np.pi * doy / 365.0),
            (d // 7) / max(W_season, 1.0),
        ])

    return {
        'X_clim': torch.tensor(np.array(X_clim, dtype=np.float32), device=device),
        'X_seas': torch.tensor(np.array(X_seas, dtype=np.float32), device=device),
        'init_cond': [S0, E0, Is0, Ia0],
        'beta_pinns': torch.tensor(beta_pinns, device=device),
        'inc_obs': inc_obs,
        'L': L, 'W': W_season, 'L_pinns': L_pinns,
        'label': f"{region}_{year}",
    }


# ====================== PINNs 微调 ======================

def fine_tune_on_pinns(model, X_clim, X_seas, beta_pinns_target, L_pinns,
                       epochs=200, lr=1e-5, print_every=50):
    """
    用 PINNs β (前 L_pinns 天) 对模型做少量轮次微调.

    仅在前 L_pinns 天的数据上计算 MSE loss,
    但 TCN 因果卷积会利用全序列上下文保证时序一致性.

    参数:
      model:             已加载 S2 权重的 DirectBetaNet_v1
      X_clim:            (L, 112)  全季气候特征
      X_seas:            (L, 3)    全季季节特征
      beta_pinns_target: (L_pinns,) PINNs β 目标值 (无 NaN)
      L_pinns:           int       PINNs 覆盖天数
      epochs:            int       微调轮次
      lr:                float     微调学习率
      print_every:       int       打印间隔
    """
    if L_pinns < 7:
        print(f"    Fine-tune: SKIP (PINNs covers only {L_pinns} days)")
        return model

    # 提取前 L_pinns 天的特征和目标
    Xc_train = X_clim[:L_pinns]       # (L_pinns, 112)
    Xs_train = X_seas[:L_pinns]       # (L_pinns, 3)
    target = beta_pinns_target[:L_pinns]  # (L_pinns,)

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    loss_init = None
    print(f"    Fine-tune: {epochs} epochs, lr={lr:.1e}, "
          f"on first {L_pinns} days PINNs beta")

    for ep in range(epochs):
        optimizer.zero_grad()

        # 前向：仅取前 L_pinns 天的预测与 PINNs 对比
        beta_pred_full = model(Xc_train, Xs_train)  # (L_pinns,)
        loss = criterion(beta_pred_full, target)

        loss.backward()
        optimizer.step()

        if ep == 0:
            loss_init = loss.item()

        if (ep + 1) % print_every == 0:
            print(f"      [{ep+1:4d}/{epochs}]  beta MSE: {loss.item():.6e}")

    # loss_final = loss.item()
    model.eval()

    # if loss_init is not None:
    #     reduction = (1 - loss_final / loss_init) * 100
    #     print(f"    Fine-tune done: loss {loss_init:.6e} → {loss_final:.6e} "
    #           f"({reduction:.1f}% reduction)")
    # else:
    #     print(f"    Fine-tune done: loss = {loss_final:.6e}")

    return model


# ====================== 滑动窗口阶段性微调 ======================

def fine_tune_on_incidence(model, X_clim, X_seas, init_cond,
                           inc_obs_weekly, current_week,
                           epochs=50, lr=5e-6, print_every=25):
    """
    用已知周发病率对模型做阶段性微调 (滑动窗口阶段).

    Loss = MSE( ODE积分周发病率[:current_week], 观测周发病率[:current_week] ).
    梯度通过 ODE 积分反向传播至 β, 再经 TCN 编码器回传至模型参数.

    微调仅在已知段 (week 0 .. current_week-1) 上进行,
    但 TCN 因果卷积可利用全序列气候上下文保证时序一致性.

    参数:
      model:           当前模型 (继承上一次微调后的权重)
      X_clim:          (L, 112)  全季气候特征
      X_seas:          (L, 3)    全季季节特征
      init_cond:       [S0, E0, Is0, Ia0]
      inc_obs_weekly:  (W,)      观测周发病率 (/1000), 可能含 NaN
      current_week:    int       当前窗口起始周 (已知段 = 0 .. current_week-1)
      epochs:          int       微调轮次
      lr:              float     微调学习率
      print_every:     int       打印间隔
    """
    if current_week < 2:
        print(f"    Sliding FT: SKIP (current_week={current_week} < 2, "
              f"insufficient data)")
        return model

    # 准备已知周发病率 target (仅取有效部分)
    inc_target_np = inc_obs_weekly[:current_week]       # (current_week,)
    valid_mask = ~np.isnan(inc_target_np)
    if valid_mask.sum() < 2:
        print(f"    Sliding FT: SKIP (only {valid_mask.sum()} valid obs weeks)")
        return model

    inc_target = torch.tensor(inc_target_np, device=device, dtype=torch.float32)
    valid_t = torch.tensor(valid_mask, device=device)

    # 已知段天数
    known_days = current_week * 7

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    loss_init = None
    print(f"    Sliding FT (wk {current_week}): {epochs} epochs, lr={lr:.1e}, "
          f"known={current_week}w ({known_days}d)")

    for ep in range(epochs):
        optimizer.zero_grad()

        # (a) 全季 β 预测 (TCN 需要全序列以保持因果上下文)
        beta_pred = model(X_clim, X_seas)              # (L,)

        # (b) ODE 积分: 仅积分已知段 (前 known_days 天)
        inc_daily = forward_euler_single_season(
            beta_pred, init_cond, max_steps=known_days)  # (known_days,)

        # (c) 日 → 周聚合
        inc_weekly = inc_daily.view(current_week, 7).sum(dim=1)  # (current_week,)

        # (d) MSE loss (仅有效周)
        loss = criterion(inc_weekly[valid_t], inc_target[valid_t])

        loss.backward()
        optimizer.step()

        if ep == 0:
            loss_init = loss.item()

        if (ep + 1) % print_every == 0:
            print(f"      [{ep+1:4d}/{epochs}]  inc MSE: {loss.item():.6e}")

    loss_final = loss.item()
    model.eval()

    # if loss_init is not None:
    #     reduction = (1 - loss_final / max(loss_init, 1e-12)) * 100
    #     print(f"    Sliding FT done (wk {current_week}): "
    #           f"loss {loss_init:.6e} → {loss_final:.6e} "
    #           f"({reduction:.1f}% reduction)")
    # else:
    #     print(f"    Sliding FT done (wk {current_week}): loss = {loss_final:.6e}")

    return model


# ====================== 滑动窗口可视化 ======================

def plot_sliding_windows(data, inc_pred_weekly_np,
                         windows, region, yr):
    """
    绘制滑动窗口预测子图 (test.py 风格).

    残差假设:  UDE 预测值与真实值之间的残差 ~ N(0, sigma²).
    sigma 通过 expanding-window RMS 逐窗口估计:
      residual_std[w] = RMS( inc_pred[:w] - inc_obs[:w] )

    使用 erfinv 反推高斯分位数, 绘制 50%/90% 置信区间.

    每个子图展示 (按 zorder):
      1. 全季真实周发病率 浅蓝散点 (仅点, 不连线)  #9ecae1
      2. 历史已知段         橙色散点 (仅点, 不连线)  #ff7f0e
      3. 预测窗口真值       紫色散点               #9467bd
      4. 90% CI 带          橙色半透明 (仅预测窗)  #ff7f0e  alpha=.25
      5. 50% CI 带          绿色半透明 (仅预测窗)  #2ca02c  alpha=.35
      6. UDE 预测线         红色粗线               #d62728
         (从 wk 0 连续画至预测窗口末尾)

    注意: win_inc 与 residual_std 在此函数内用传入的最终
    inc_pred_weekly_np 重新计算, 确保 CI 中心与 UDE 线完全对齐,
    消除因阶段性微调更新 inc_pred_np 导致的快照版本不一致问题.

    参数:
      data:                   build_prediction_data 返回的数据字典
      inc_pred_weekly_np:     (W,) 平滑预测周发病率 (已乘 1000)
      windows:                list[dict], 含 start_week, weeks, inc_weekly,
                             residual_std
      region:                 str 地区名
      yr:                     int 年份
    """
    from scipy.special import erfinv

    W = data['W']
    n_total = len(windows)

    # 高斯分位数 (test.py 第 166–167 行)
    z50 = np.sqrt(2.0) * erfinv(0.50)
    z90 = np.sqrt(2.0) * erfinv(0.90)

    # 选取代表性窗口 (最多 ~max_subplots 个)
    max_plot = SLIDING_CONFIG["max_subplots"]
    step = max(1, n_total // max_plot)
    selected_idx = list(range(0, n_total, step))
    # 始终包含最后一个窗口
    if selected_idx[-1] != n_total - 1:
        selected_idx.append(n_total - 1)
    selected = [windows[i] for i in selected_idx]

    n_win = len(selected)
    cols = min(4, n_win)
    rows = (n_win + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols,
                             figsize=(4 * cols, 2.8 * rows), dpi=300)
    axes_flat = axes.flatten() if n_win > 1 else[axes]

    # 准备真值数据 (周级别)
    inc_obs_np = None
    if data['inc_obs'] is not None:
        inc_obs_np = data['inc_obs'] * 1000.0

    weeks_all = np.arange(W)

    for idx, win in enumerate(selected):
        ax = axes_flat[idx]
        start_week = win['start_week']
        win_weeks = win['weeks']          # e.g. [w, w+1, w+2, w+3]

        # ---- 使用最终 inc_pred_weekly_np 重算 win_inc 与 residual_std ----
        # 原因: 滑动窗口循环中 inc_pred_np 可能被阶段性微调更新,
        # 窗口快照的 win_inc / residual_std 可能与最终预测不一致,
        # 导致 CI 中心偏离 UDE 线。此处统一用最终值保证对齐。
        win_inc = inc_pred_weekly_np[start_week:
                                     start_week + len(win_weeks)]

        if inc_obs_np is not None and start_week > 0:
            hist_pred = inc_pred_weekly_np[:start_week]
            hist_true = inc_obs_np[:start_week]
            valid = ~np.isnan(hist_true)
            if valid.sum() > 1:
                res = hist_pred[valid] - hist_true[valid]
                residual_std = float(np.sqrt(np.mean(res ** 2)))
            else:
                residual_std = 0.0
        else:
            residual_std = 0.0

        # --- (1) 全季真实周发病率  浅蓝散点 (仅散点, 不连线) ---
        if inc_obs_np is not None:
            ax.plot(weeks_all, inc_obs_np, "o", color="#9ecae1",
                    ms=4, alpha=0.6, zorder=1, label="Observations")

        # --- (2) 历史已知段 (week 0 .. w-1)  散点 (不连线) ---
        if inc_obs_np is not None and start_week > 0:
            # ax.plot(np.arange(start_week), inc_obs_np[:start_week],
            #         "o", color="#ff7f0e", ms=5, zorder=4)
            ax.plot(np.arange(start_week), inc_obs_np[:start_week],
                    "o", color="#9ecae1", ms=5, zorder=4)

        # --- (3) 预测窗口真值  紫色散点 ---
        if inc_obs_np is not None:
            ax.plot(win_weeks, inc_obs_np[win_weeks],
                    "o", color="#9467bd", ms=5, zorder=5)

        # --- 预测起点竖虚线 (forecast origin) ---
        ax.axvline(x=start_week-1, color="gray", linestyle="--",
                   linewidth=1.5, zorder=2,
                   label="Start week")

        # --- UDE 预测线: 从 wk 0 连续画到预测窗口末尾 ---
        ude_x = np.arange(start_week + len(win_weeks))
        ude_y = inc_pred_weekly_np[:start_week + len(win_weeks)]

        # --- CI x 轴: 从历史最后一点 (wk w-1) 起, 喇叭口向外扩张 ---
        # 在 wk w-1 处 CI 收束至锚点 (零宽度), 视觉上与历史段自然衔接
        ci_x = np.concatenate([[start_week - 1], win_weeks])

        # 锚点取 UDE 预测值 (CI 围绕 UDE 预测线展开)
        anchor_val = inc_pred_weekly_np[start_week - 1]

        # --- (4) 90% CI 带  喇叭形 (收束于 wk w-1, 张开覆盖预测窗) ---
        lo90 = np.clip(np.concatenate([
            [anchor_val],                        # 锚点处宽度=0
            win_inc - z90 * residual_std,        # 预测窗内张开
        ]), 0, None)
        hi90 = np.concatenate([
            [anchor_val],
            win_inc + z90 * residual_std,
        ])
        ax.fill_between(ci_x, lo90, hi90,
                        color="#ff7f0e", alpha=0.25, lw=0,
                        label="90%")

        # --- (5) 50% CI 带  喇叭形 (收束于 wk w-1, 张开覆盖预测窗) ---
        lo50 = np.clip(np.concatenate([
            [anchor_val],
            win_inc - z50 * residual_std,
        ]), 0, None)
        hi50 = np.concatenate([
            [anchor_val],
            win_inc + z50 * residual_std,
        ])
        ax.fill_between(ci_x, lo50, hi50,
                        color="#2ca02c", alpha=0.35, lw=0,
                        label="50%")

        # --- (6) 预测线  红色粗线 (wk 0 → 预测窗口末尾) ---
        ax.plot(ude_x, ude_y, "-", color="#d62728",
                lw=2, ms=5, label="UDE", zorder=0)

        # --- 标注 ---
        # ax.set_title(f"wk={start_week}", fontsize=9)
        ax.set_xlabel("Time(weeks)", fontsize=10)
        ax.set_ylabel("Incidence(per 1000)", fontsize=10)
        ax.tick_params(labelsize=9)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)

        # 第一个子图显示图例 (test.py 第 183 行)
        if idx == 0:
            ax.legend(fontsize=7, loc="upper right")

    # 隐藏多余子图
    for i in range(n_win, len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR,
                            f"UDE_Pred_{region}_{yr}_sliding.png")
    plt.savefig(fig_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"    Figure (sliding): {fig_path}")


# ====================== 单地区预测 ======================

def predict_region(region, pred_years):
    print(f"\n{'='*55}\n  {region}  Prediction ({pred_years})\n{'='*55}")

    clim_path = os.path.join(CLIMATE_DIR, f"{region}_Climate_Daily.csv")
    clim_df = pd.read_csv(clim_path, parse_dates=['datetime'],
                          index_col='datetime').sort_index()

    scaler_path = os.path.join(SAVE_DIR, f"scaler_{region}.pkl")
    if not os.path.exists(scaler_path):
        print(f"  WARNING scaler missing: {scaler_path}, skip")
        return
    scaler = joblib.load(scaler_path)

    model_path = os.path.join(SAVE_DIR, f"{region}_Stage2.pt")
    if not os.path.exists(model_path):
        print(f"  WARNING Stage2 model missing: {model_path}, skip")
        return

    # --- 加载预训练模型 ---
    model = TransNet().to(device)
    ckpt = torch.load(model_path, map_location=device)
    state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state)
    print(f"  Model: {model_path}")
    print(f"  Arch: TCN head, {CLIMATE_WINDOW}d window")
    print(f"  Post-smooth: {SMOOTH_CONFIG['method']}")

    for yr in pred_years:
        print(f"\n  --- {region} {yr} ---")

        data = build_prediction_data(region, yr, clim_df, scaler)
        if data is None:
            print(f"    WARNING cannot build prediction data, skip")
            continue

        L, W, L_pinns = data['L'], data['W'], data['L_pinns']
        print(f"    {L} days ({W} weeks), PINNs covers first {L_pinns} days")
        print(f"    Init: S0={data['init_cond'][0]:.4f}, E0={data['init_cond'][1]:.6f}, "
              f"Is0={data['init_cond'][2]:.6f}, Ia0={data['init_cond'][3]:.6f}")

        # --- Step 1: PINNs 微调  ---
        if FINETUNE_CONFIG["enabled"] and L_pinns >= 7:
            # 构造 PINNs 目标
            beta_pinns_np = data['beta_pinns'].cpu().numpy()
            beta_pinns_valid = beta_pinns_np[:L_pinns]
            beta_pinns_target = torch.tensor(beta_pinns_valid, device=device,
                                             dtype=torch.float32)

            model = fine_tune_on_pinns(
                model,
                data['X_clim'], data['X_seas'],
                beta_pinns_target, L_pinns,
                epochs=FINETUNE_CONFIG["epochs"],
                lr=FINETUNE_CONFIG["lr"],
                print_every=FINETUNE_CONFIG["print_every"],
            )
        elif L_pinns < 7:
            print(f"    Fine-tune: SKIP (PINNs too short: {L_pinns} days)")
        else:
            print(f"    Fine-tune: DISABLED")

        # --- Step 2: 微调后全季 β 预测 ---
        with torch.no_grad():
            beta_raw = model(data['X_clim'], data['X_seas'])  # (L,)
        beta_raw_np = beta_raw.cpu().numpy()

        # --- Step 3: S-G 后处理平滑 ---
        beta_smooth_np, smooth_info = smooth_beta(
            beta_raw_np,
            method=SMOOTH_CONFIG["method"],
            window=SMOOTH_CONFIG["savgol_window"],
            polyorder=SMOOTH_CONFIG["savgol_polyorder"],
            sigma=SMOOTH_CONFIG["gaussian_sigma"],
        )

        # --- Step 4: 平滑度指标 ---
        metrics_raw = compute_smoothness_metrics(beta_raw_np)
        metrics_smooth = compute_smoothness_metrics(beta_smooth_np)

        # --- Step 5: ODE 积分 ---
        beta_smooth_tensor = torch.tensor(beta_smooth_np, device=device,
                                          dtype=torch.float32)
        with torch.no_grad():
            Inc_daily_smooth = forward_euler_single_season(beta_smooth_tensor,
                                                           data['init_cond'])
            Inc_pred_weekly = Inc_daily_smooth.view(W, 7).sum(dim=1)

            Inc_daily_raw = forward_euler_single_season(beta_raw, data['init_cond'])
            Inc_weekly_raw = Inc_daily_raw.view(W, 7).sum(dim=1)

        inc_pred_np = Inc_pred_weekly.cpu().numpy() * 1000.0
        inc_raw_np = Inc_weekly_raw.cpu().numpy() * 1000.0
        Inc_daily_smooth_np = Inc_daily_smooth.cpu().numpy() * 1000.0
        beta_pinns_np = data['beta_pinns'].cpu().numpy()

        # # --- Print summary ---
        # beta_pinns_valid_all = beta_pinns_np[~np.isnan(beta_pinns_np)]
        # # PINNs 覆盖段的 β 拟合度
        # beta_pred_pinns_seg = beta_raw_np[:L_pinns]
        # beta_pinns_seg = beta_pinns_np[:L_pinns]
        # mse_pinns_seg = np.mean((beta_pred_pinns_seg - beta_pinns_seg) ** 2)
        #
        # print(f"\n    PINNs segment fit (first {L_pinns}d):")
        # print(f"      MSE = {mse_pinns_seg:.6e}")
        # print(f"\n    Beta smoothness:")
        # print(f"      Raw:      RMS(d)={metrics_raw['rms_delta']:.6f}, "
        #       f"max_jump={metrics_raw['max_jump']:.6f}")
        # print(f"      Smoothed: RMS(d)={metrics_smooth['rms_delta']:.6f}, "
        #       f"max_jump={metrics_smooth['max_jump']:.6f}")
        # red = (1 - metrics_smooth['rms_delta'] / max(metrics_raw['rms_delta'], 1e-12)) * 100
        # print(f"      => Reduction: {red:.1f}%")
        # print(f"\n    Beta stats:")
        # print(f"      Raw:      mean={beta_raw_np.mean():.4f}, "
        #       f"range=[{beta_raw_np.min():.4f}, {beta_raw_np.max():.4f}]")
        # print(f"      Smoothed: mean={beta_smooth_np.mean():.4f}, "
        #       f"range=[{beta_smooth_np.min():.4f}, {beta_smooth_np.max():.4f}]")
        # print(f"      PINNs:    mean={beta_pinns_valid_all.mean():.4f}, "
        #       f"range=[{beta_pinns_valid_all.min():.4f}, {beta_pinns_valid_all.max():.4f}] "
        #       f"({len(beta_pinns_valid_all)} days)")
        # print(f"\n    Incidence (smoothed):")
        # print(f"      Total={inc_pred_np.sum():.1f}, peak_week={np.argmax(inc_pred_np)}, "
        #       f"peak={inc_pred_np.max():.1f}")
        # print(f"    Incidence (raw):")
        # print(f"      Total={inc_raw_np.sum():.1f}, peak_week={np.argmax(inc_raw_np)}, "
        #       f"peak={inc_raw_np.max():.1f}")
        # if data['inc_obs'] is not None:
        #     inc_obs_np = data['inc_obs'] * 1000.0
        #     inc_obs_valid = inc_obs_np[~np.isnan(inc_obs_np)]
        #     print(f"    Incidence (observed):")
        #     print(f"      Total={inc_obs_valid.sum():.1f}, "
        #           f"peak_week={np.nanargmax(inc_obs_np)}, "
        #           f"peak={np.nanmax(inc_obs_np):.1f} ({len(inc_obs_valid)} weeks)")


        # ============================================================
        # --- Step 6: 滑动窗口预测 + 阶段性微调 (v2 新增) ---
        # ============================================================
        # 滑动步长 = 1 周, 预测窗口 = 4 周
        # 每 interval_weeks 周触发一次微调 (权重继承, 不重置)
        # 微调后重新预测全季 β → S-G 平滑 → ODE 积分 → 更新 inc_pred_np
        #
        # 残差假设: pred - true ~ N(0, sigma²)
        # sigma 通过 expanding-window RMS 估计:
        #   residual_std[w] = RMS( inc_pred[:w] - inc_obs[:w] )
        win_weeks = SLIDING_CONFIG["window_weeks"]
        start_wk = SLIDING_CONFIG["start_week"]
        ft_interval = SLIDING_FINETUNE_CONFIG["interval_weeks"]
        ft_enabled = SLIDING_FINETUNE_CONFIG["enabled"]

        # 真值数组 (周, 已乘 1000)
        inc_obs_np = None
        if data['inc_obs'] is not None:
            inc_obs_np = data['inc_obs'] * 1000.0

        windows = []
        for w in range(start_wk, W - win_weeks + 1):
            # ========================================================
            # 阶段性微调 (每 interval_weeks 周触发一次)
            # ========================================================
            if (ft_enabled
                    and inc_obs_np is not None
                    and (w - start_wk) % ft_interval == 0):
                print(f"\n    --- Sliding FT trigger: wk {w} "
                      f"(window #{ (w - start_wk) // ft_interval + 1}) ---")

                # 用已知周发病率微调模型 (权重继承)
                model = fine_tune_on_incidence(
                    model,
                    data['X_clim'], data['X_seas'],
                    data['init_cond'],
                    data['inc_obs'],        # 原始 inc_obs (/1000)
                    current_week=w,
                    epochs=SLIDING_FINETUNE_CONFIG["epochs"],
                    lr=SLIDING_FINETUNE_CONFIG["lr"],
                    print_every=SLIDING_FINETUNE_CONFIG["print_every"],
                )

                # 微调后重跑全季预测流水线
                with torch.no_grad():
                    beta_raw = model(data['X_clim'], data['X_seas'])
                beta_raw_np = beta_raw.cpu().numpy()
                beta_smooth_np, _ = smooth_beta(
                    beta_raw_np,
                    method=SMOOTH_CONFIG["method"],
                    window=SMOOTH_CONFIG["savgol_window"],
                    polyorder=SMOOTH_CONFIG["savgol_polyorder"],
                    sigma=SMOOTH_CONFIG["gaussian_sigma"],
                )
                beta_smooth_tensor = torch.tensor(
                    beta_smooth_np, device=device, dtype=torch.float32)
                with torch.no_grad():
                    Inc_daily_smooth = forward_euler_single_season(
                        beta_smooth_tensor, data['init_cond'])
                    Inc_pred_weekly = Inc_daily_smooth.view(W, 7).sum(dim=1)
                inc_pred_np = Inc_pred_weekly.cpu().numpy() * 1000.0
                print(f"      Updated inc_pred: total={inc_pred_np.sum():.1f}, "
                      f"peak_wk={np.argmax(inc_pred_np)}, "
                      f"peak={inc_pred_np.max():.1f}")

            # ========================================================
            # expanding-window 残差 RMS
            # ========================================================
            if inc_obs_np is not None and w > 0:
                hist_pred = inc_pred_np[:w]
                hist_true = inc_obs_np[:w]
                valid = ~np.isnan(hist_true)
                if valid.sum() > 1:
                    res = hist_pred[valid] - hist_true[valid]
                    residual_std = float(np.sqrt(np.mean(res ** 2)))
                else:
                    residual_std = 0.0
            else:
                residual_std = 0.0

            windows.append({
                'start_week': w,
                'weeks': np.arange(w, w + win_weeks),
                'inc_weekly': inc_pred_np[w:w + win_weeks],
                'residual_std': residual_std,
            })

        n_windows = len(windows)
        n_ft_triggers = sum(
            1 for w in range(start_wk, W - win_weeks + 1)
            if ft_enabled and inc_obs_np is not None
            and (w - start_wk) % ft_interval == 0)
        print(f"\n    Sliding windows: {n_windows} total "
              f"(wk {start_wk}–{W - win_weeks}, "
              f"size={win_weeks}w, step=1w)")
        print(f"    Periodic fine-tunes triggered: {n_ft_triggers} "
              f"(every {ft_interval}w, weight-inheriting)")
        print(f"    First window: weeks [{windows[0]['start_week']}"
              f"–{windows[0]['start_week'] + win_weeks - 1}]")
        print(f"    Last  window: weeks [{windows[-1]['start_week']}"
              f"–{windows[-1]['start_week'] + win_weeks - 1}]")
        print(f"    Residual std: first={windows[0]['residual_std']:.3f}, "
              f"last={windows[-1]['residual_std']:.3f}")

        # --- Step 7: 滑动窗口可视化 (v2 新增) ---
        # ODE 按天计算, 绘图按周输出, 仅绘制 inc 变量
        plot_sliding_windows(data, inc_pred_np,
                            windows, region, yr)


# ====================== 主函数 ======================

def main():
    plans = get_prediction_plans()
    ft_status = "ON" if FINETUNE_CONFIG["enabled"] else "OFF"
    sft_status = "ON" if SLIDING_FINETUNE_CONFIG["enabled"] else "OFF"
    win_wks = SLIDING_CONFIG["window_weeks"]
    sft_int = SLIDING_FINETUNE_CONFIG["interval_weeks"]
    print(f"UDE Prediction (TCN + {CLIMATE_WINDOW}d window + "
          f"PINNs ft [{ft_status}] + sliding ft [{sft_status}] "
          f"every {sft_int}w + {win_wks}w sliding windows)")
    print(f"  {len(plans)} region(s)")
    for region, years in sorted(plans.items()):
        print(f"    {region}: {years}")

    for region in sorted(plans):
        predict_region(region, plans[region])


if __name__ == "__main__":
    main()
