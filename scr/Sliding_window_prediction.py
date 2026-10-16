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

# ====================== Training configuration ======================
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

# ====================== PINNs fine tune configuration ======================
FINETUNE_CONFIG = {
    "enabled": True,
    "epochs": 200,
    "lr": 1e-5,
    "print_every": 50,
}
# ===========================================================

SMOOTH_CONFIG = {
    "method": "savgol",
    "savgol_window": 7,
    "savgol_polyorder": 3,
    "gaussian_sigma": 1.0,
}

# ====================== Sliding window configuration ======================
SLIDING_CONFIG = {
    "window_weeks": 4,
    "start_week": 5,
    "slide_step": 1,
    "max_subplots": 30,
}
# ===========================================================

# ====================== Sliding window fine tune configuration ======================
SLIDING_FINETUNE_CONFIG = {
    "enabled": True,
    "interval_weeks": 1,
    "epochs": 50,
    "lr": 5e-6,
    "print_every": 25,        
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

# plt
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

        beta_pred_full = model(Xc_train, Xs_train)  # (L_pinns,)
        loss = criterion(beta_pred_full, target)

        loss.backward()
        optimizer.step()

        if ep == 0:
            loss_init = loss.item()

        if (ep + 1) % print_every == 0:
            print(f"      [{ep+1:4d}/{epochs}]  beta MSE: {loss.item():.6e}")

    model.eval()
    return model


# ====================== Sliding window fine tune ======================

def fine_tune_on_incidence(model, X_clim, X_seas, init_cond,
                           inc_obs_weekly, current_week,
                           epochs=50, lr=5e-6, print_every=25):
    if current_week < 2:
        print(f"    Sliding FT: SKIP (current_week={current_week} < 2, "
              f"insufficient data)")
        return model

    inc_target_np = inc_obs_weekly[:current_week]       # (current_week,)
    valid_mask = ~np.isnan(inc_target_np)
    if valid_mask.sum() < 2:
        print(f"    Sliding FT: SKIP (only {valid_mask.sum()} valid obs weeks)")
        return model

    inc_target = torch.tensor(inc_target_np, device=device, dtype=torch.float32)
    valid_t = torch.tensor(valid_mask, device=device)

    known_days = current_week * 7

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    loss_init = None
    print(f"    Sliding FT (wk {current_week}): {epochs} epochs, lr={lr:.1e}, "
          f"known={current_week}w ({known_days}d)")

    for ep in range(epochs):
        optimizer.zero_grad()

        # (a) beta pred
        beta_pred = model(X_clim, X_seas)              # (L,)

        # (b) ODE
        inc_daily = forward_euler_single_season(
            beta_pred, init_cond, max_steps=known_days)  # (known_days,)

        inc_weekly = inc_daily.view(current_week, 7).sum(dim=1)  # (current_week,)

        # (c) MSE loss
        loss = criterion(inc_weekly[valid_t], inc_target[valid_t])

        loss.backward()
        optimizer.step()

        if ep == 0:
            loss_init = loss.item()

        if (ep + 1) % print_every == 0:
            print(f"      [{ep+1:4d}/{epochs}]  inc MSE: {loss.item():.6e}")

    model.eval()

    return model


# ====================== plt ======================

def plot_sliding_windows(data, inc_pred_weekly_np,
                         windows, region, yr):
    from scipy.special import erfinv

    W = data['W']
    n_total = len(windows)
    z50 = np.sqrt(2.0) * erfinv(0.50)
    z90 = np.sqrt(2.0) * erfinv(0.90)
    max_plot = SLIDING_CONFIG["max_subplots"]
    step = max(1, n_total // max_plot)
    selected_idx = list(range(0, n_total, step))

    if selected_idx[-1] != n_total - 1:
        selected_idx.append(n_total - 1)
    selected = [windows[i] for i in selected_idx]

    n_win = len(selected)
    cols = min(4, n_win)
    rows = (n_win + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols,
                             figsize=(4 * cols, 2.8 * rows), dpi=300)
    axes_flat = axes.flatten() if n_win > 1 else[axes]

    inc_obs_np = None
    if data['inc_obs'] is not None:
        inc_obs_np = data['inc_obs'] * 1000.0

    weeks_all = np.arange(W)

    for idx, win in enumerate(selected):
        ax = axes_flat[idx]
        start_week = win['start_week']
        win_weeks = win['weeks']          # e.g. [w, w+1, w+2, w+3]

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

        # --- (1)  ---
        if inc_obs_np is not None:
            ax.plot(weeks_all, inc_obs_np, "o", color="#9ecae1",
                    ms=4, alpha=0.6, zorder=1, label="Observations")

        # --- (2)  ---
        if inc_obs_np is not None and start_week > 0:
            # ax.plot(np.arange(start_week), inc_obs_np[:start_week],
            #         "o", color="#ff7f0e", ms=5, zorder=4)
            ax.plot(np.arange(start_week), inc_obs_np[:start_week],
                    "o", color="#9ecae1", ms=5, zorder=4)

        # --- (3)  ---
        if inc_obs_np is not None:
            ax.plot(win_weeks, inc_obs_np[win_weeks],
                    "o", color="#9467bd", ms=5, zorder=5)

        # --- (forecast origin) ---
        ax.axvline(x=start_week-1, color="gray", linestyle="--",
                   linewidth=1.5, zorder=2,
                   label="Start week")

        # --- UDE  ---
        ude_x = np.arange(start_week + len(win_weeks))
        ude_y = inc_pred_weekly_np[:start_week + len(win_weeks)]

        # --- CI  ---
        ci_x = np.concatenate([[start_week - 1], win_weeks])


        anchor_val = inc_pred_weekly_np[start_week - 1]

        # --- (4) 90% CI band ---
        lo90 = np.clip(np.concatenate([
            [anchor_val],
            win_inc - z90 * residual_std,
        ]), 0, None)
        hi90 = np.concatenate([
            [anchor_val],
            win_inc + z90 * residual_std,
        ])
        ax.fill_between(ci_x, lo90, hi90,
                        color="#ff7f0e", alpha=0.25, lw=0,
                        label="90%")

        # --- (5) 50% CI band ---
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

        # --- (6)  ---
        ax.plot(ude_x, ude_y, "-", color="#d62728",
                lw=2, ms=5, label="UDE", zorder=0)


        ax.set_xlabel("Time(weeks)", fontsize=10)
        ax.set_ylabel("Incidence(per 1000)", fontsize=10)
        ax.tick_params(labelsize=9)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)

        if idx == 0:
            ax.legend(fontsize=7, loc="upper right")

    for i in range(n_win, len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR,
                            f"UDE_Pred_{region}_{yr}_sliding.png")
    plt.savefig(fig_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"    Figure (sliding): {fig_path}")


# ====================== Single-region pred ======================

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

    # --- load ---
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

        # --- Step 1: PINNs fine tune  ---
        if FINETUNE_CONFIG["enabled"] and L_pinns >= 7:
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

        # --- Step 2 ---
        with torch.no_grad():
            beta_raw = model(data['X_clim'], data['X_seas'])  # (L,)
        beta_raw_np = beta_raw.cpu().numpy()

        # --- Step 3 ---
        beta_smooth_np, smooth_info = smooth_beta(
            beta_raw_np,
            method=SMOOTH_CONFIG["method"],
            window=SMOOTH_CONFIG["savgol_window"],
            polyorder=SMOOTH_CONFIG["savgol_polyorder"],
            sigma=SMOOTH_CONFIG["gaussian_sigma"],
        )

        # --- Step 4 ---
        metrics_raw = compute_smoothness_metrics(beta_raw_np)
        metrics_smooth = compute_smoothness_metrics(beta_smooth_np)

        # --- Step 5: ODE ---
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

        win_weeks = SLIDING_CONFIG["window_weeks"]
        start_wk = SLIDING_CONFIG["start_week"]
        ft_interval = SLIDING_FINETUNE_CONFIG["interval_weeks"]
        ft_enabled = SLIDING_FINETUNE_CONFIG["enabled"]

        # true
        inc_obs_np = None
        if data['inc_obs'] is not None:
            inc_obs_np = data['inc_obs'] * 1000.0

        windows = []
        for w in range(start_wk, W - win_weeks + 1):
            if (ft_enabled
                    and inc_obs_np is not None
                    and (w - start_wk) % ft_interval == 0):
                print(f"\n    --- Sliding FT trigger: wk {w} "
                      f"(window #{ (w - start_wk) // ft_interval + 1}) ---")

                model = fine_tune_on_incidence(
                    model,
                    data['X_clim'], data['X_seas'],
                    data['init_cond'],
                    data['inc_obs'],
                    current_week=w,
                    epochs=SLIDING_FINETUNE_CONFIG["epochs"],
                    lr=SLIDING_FINETUNE_CONFIG["lr"],
                    print_every=SLIDING_FINETUNE_CONFIG["print_every"],
                )

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

        plot_sliding_windows(data, inc_pred_np,
                            windows, region, yr)


# ====================== main ======================

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
