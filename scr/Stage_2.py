"""
UDE Stage 2 v1 — TCN 架构 + 56 天窗口 + ODE Incidence Fit
输出: model/{Region}_UDE_S2_v1.pt + figures/UDE_S2_v1_{Region}.png
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
from Transmission_network import TransNet, CLIMATE_WINDOW

torch.manual_seed(1234); np.random.seed(1234)
torch.set_default_dtype(torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"CLIMATE_WINDOW = {CLIMATE_WINDOW} days")

# ====================== 训练配置 ======================
TRAIN_CONFIG = {
    # "Abruzzo":                [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2023, 2024],
    # "Apulia":                 [2011, 2013, 2014, 2015, 2016, 2017, 2018, 2023, 2024],
    # "Campania":                [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2023, 2024],
    # "Emilia-Romagna":         [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2023, 2024],
    # "Friuli-Venezia_Giulia":  [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2023, 2024],
    # "Lazio":                  [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2023, 2024],
    # "Liguria":                [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2023, 2024],
    # "Lombardy":               [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2023, 2024],
    # "Marche":                 [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2023, 2024],
    # "Piedmont":               [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2023, 2024],
    # "Sardinia":               [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2023, 2024],
    # "Sicily":                 [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2023, 2024],
    # "Tuscany":                 [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2023, 2024],
    # "Umbria":                 [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2023, 2024],
    "Veneto":                 [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2023, 2024],
}
# =========================================================

PARAM_COLS = ["temperature_c", "absolute_humidity"]
EPOCHS = 300
LR = 1e-4
VAL_RATIO = 0.2

# ODE 流行病学参数
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


def get_season_date_range(incidence_file):
    df = pd.read_csv(incidence_file)
    sy, sw = map(int, df['year_week'].iloc[0].split('-'))
    ey, ew = map(int, df['year_week'].iloc[-1].split('-'))
    return (pd.to_datetime(date.fromisocalendar(sy, sw, 1)),
            pd.to_datetime(date.fromisocalendar(ey, ew, 7)))


def get_training_plans():
    plans = {}
    for region, years in TRAIN_CONFIG.items():
        valid = []
        for yr in years:
            pnns = os.path.join(PINNS_DIR, f"{region}_{yr}.csv")
            epi = os.path.join(Epi_DIR, f"{region}_{yr}.csv")
            if os.path.exists(pnns) and os.path.exists(epi):
                valid.append(yr)
        if valid:
            plans[region] = sorted(valid)
        else:
            print(f"  WARNING {region}: no valid data, skip")
    return plans


def alpha_schedule(ep, target=100, k=12, x0=0.6):
    if ep >= target:
        return 1.0
    return 1.0 / (1.0 + np.exp(-k * (ep / target - x0)))


def forward_euler_multiseason(beta_seq, season_daily_lens, init_conds):
    dtype = beta_seq.dtype
    dt = 1.0
    inc_list = []
    start = 0
    for i, L in enumerate(season_daily_lens):
        S, E, Is, Ia = [torch.tensor(v, device=device, dtype=dtype)
                        for v in init_conds[i]]
        for t in range(L):
            beta = beta_seq[start + t]
            infection = beta * S * (Is + THETA * Ia)
            S = S + dt * (-infection + GAMMA_S * Is + GAMMA_A * Ia)
            E = E + dt * (infection - SIGMA * E)
            Is = Is + dt * (DELTA * SIGMA * E - GAMMA_S * Is)
            Ia = Ia + dt * ((1 - DELTA) * SIGMA * E - GAMMA_A * Ia)
            inc_list.append(DELTA * SIGMA * E)
        start += L
    return torch.stack(inc_list)


def build_season_data(region, year, clim_df, scaler):
    epi_path = os.path.join(Epi_DIR, f"{region}_{year}.csv")
    pnns_path = os.path.join(PINNS_DIR, f"{region}_{year}.csv")
    if not (os.path.exists(epi_path) and os.path.exists(pnns_path)):
        return None

    start, end = get_season_date_range(epi_path)
    clim_start = start - pd.Timedelta(days=CLIMATE_WINDOW)
    clim_raw = clim_df.loc[clim_start:end][PARAM_COLS].values.astype(np.float32)
    clim_scaled = scaler.transform(clim_raw)

    min_required = CLIMATE_WINDOW + 1
    if len(clim_scaled) < min_required:
        print(f"  WARNING {region}_{year}: climate data too short "
              f"({len(clim_scaled)} rows, need >= {min_required})")
        return None

    df_pinns = pd.read_csv(pnns_path)
    beta_pinns = df_pinns['beta'].values.astype(np.float32)
    S0, E0, Is0, Ia0 = df_pinns.iloc[0][['S', 'E', 'Is', 'Ia']].values.astype(np.float32)

    df_inc = pd.read_csv(epi_path)
    inc = (df_inc['incidence'].values / 1000.0).astype(np.float32)

    L = min(len(beta_pinns), len(clim_scaled) - CLIMATE_WINDOW)
    if L < 7:
        print(f"  WARNING {region}_{year}: season too short ({L} days)")
        return None

    clim_scaled = clim_scaled[:L + CLIMATE_WINDOW]
    beta_pinns = beta_pinns[:L]
    W_season = L // 7

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
        'beta_pinns': torch.tensor(beta_pinns, device=device),
        'inc': inc,
        'init_cond': [S0, E0, Is0, Ia0],
        'L': L, 'W': W_season,
        'label': f"{region}_{year}",
    }


def train_region(region, train_years):
    print(f"\n{'='*55}\n  {region}  ({len(train_years)} seasons: {train_years})\n{'='*55}")

    clim_path = os.path.join(CLIMATE_DIR, f"{region}_Climate_Daily.csv")
    clim_df = pd.read_csv(clim_path, parse_dates=['datetime'],
                          index_col='datetime').sort_index()

    scaler_path = os.path.join(SAVE_DIR, f"scaler_{region}.pkl")
    if not os.path.exists(scaler_path):
        print(f"  WARNING scaler missing: {scaler_path}, skip")
        return
    scaler = joblib.load(scaler_path)
    print(f"  Scaler: {scaler_path}")

    seasons = []
    for yr in train_years:
        s = build_season_data(region, yr, clim_df, scaler)
        if s:
            seasons.append(s)
    if not seasons:
        print(f"  WARNING no valid seasons")
        return

    n_seasons = len(seasons)
    total_daily = sum(s['L'] for s in seasons)
    total_weekly = sum(s['W'] for s in seasons)
    print(f"  {n_seasons} seasons, {total_daily} days, {total_weekly} weeks")

    inc_true = torch.tensor(np.concatenate([s['inc'] for s in seasons]),
                            device=device)
    all_inc = np.arange(total_weekly)
    np.random.shuffle(all_inc)
    n_val = int(total_weekly * VAL_RATIO)
    val_idx = torch.tensor(all_inc[:n_val], device=device).long()
    train_idx = torch.tensor(all_inc[n_val:], device=device).long()

    s1_path = os.path.join(SAVE_DIR, f"{region}_Stage1.pt")
    if not os.path.exists(s1_path):
        print(f"  WARNING S1 model missing: {s1_path}, skip")
        return

    model = TransNet().to(device)
    s1_ckpt = torch.load(s1_path, map_location=device)
    s1_state = s1_ckpt['model_state_dict'] if 'model_state_dict' in s1_ckpt else s1_ckpt
    model.load_state_dict(s1_state)
    print(f"  Weights init from: {s1_path}")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.9, patience=30, min_lr=1e-5)
    criterion = nn.MSELoss()

    model_path = os.path.join(SAVE_DIR, f"{region}_Stage2.pt")
    best_val = float('inf')
    patience_counter = 0
    patience_limit = 20

    beta_pinns_full = torch.cat([s['beta_pinns'] for s in seasons])

    print(f"\n  Training ({EPOCHS} epochs)...")
    for ep in range(EPOCHS):
        model.train()
        optimizer.zero_grad()

        alpha = 0.0 if ep < 50 else alpha_schedule(ep)

        beta_seqs = []
        for s in seasons:
            beta_pred = model(s['X_clim'], s['X_seas'])
            beta_seqs.append(beta_pred)
        beta_full = torch.cat(beta_seqs)

        season_lens = [s['L'] for s in seasons]
        init_conds = [s['init_cond'] for s in seasons]
        Incidence_daily = forward_euler_multiseason(
            beta_full, season_lens, init_conds)
        Inc_pred_weekly = Incidence_daily.view(total_weekly, 7).sum(dim=1)

        if ep < 50:
            loss = criterion(beta_full, beta_pinns_full)
        else:
            loss_obs = criterion(Inc_pred_weekly[train_idx], inc_true[train_idx])
            loss_distill = criterion(beta_full, beta_pinns_full)
            loss = alpha * loss_obs + (1 - alpha) * loss_distill

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            if ep < 50:
                train_loss = val_loss = loss.item()
            else:
                train_loss = criterion(Inc_pred_weekly[train_idx],
                                       inc_true[train_idx]).item()
                val_loss = criterion(Inc_pred_weekly[val_idx],
                                     inc_true[val_idx]).item()
        scheduler.step(val_loss)

        if ep >= 150:
            if val_loss < best_val:
                best_val = val_loss
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                           model_path)
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"  Early stop @ epoch {ep}")
                break

        if ep % 10 == 0:
            print(f"  [{ep:4d}] alpha={alpha:.3f} lr={optimizer.param_groups[0]['lr']:.1e} "
                  f"train={train_loss:.4e} val={val_loss:.4e}")

    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, model_path)
    print(f"  OK {model_path}")

    # Visualization
    model.eval()
    with torch.no_grad():
        beta_seqs = []
        for s in seasons:
            beta_pred = model(s['X_clim'], s['X_seas'])
            beta_seqs.append(beta_pred)
        beta_full = torch.cat(beta_seqs)

        Incidence_daily = forward_euler_multiseason(
            beta_full, season_lens, init_conds)
        Inc_pred_weekly = Incidence_daily.view(total_weekly, 7).sum(dim=1)

    inc_UDE = Inc_pred_weekly.detach().cpu().numpy() * 1000.0
    inc_obs = inc_true.cpu().numpy() * 1000.0

    ncols = min(5, n_seasons)
    nrows = (n_seasons + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), dpi=600)
    axes = axes.flatten() if n_seasons > 1 else [axes]
    off = 0
    for i in range(n_seasons):
        W = seasons[i]['W']
        ax = axes[i]
        ax.scatter(np.arange(W), inc_obs[off:off+W], color="#1F77B4",
                   s=20, alpha=0.8, label="Observed")
        ax.plot(np.arange(W), inc_UDE[off:off+W], "--", color="#D62728",
                lw=2, label="UDE-S2 v1")
        ax.set_title(seasons[i]['label'], fontsize=9)
        if i == 0:
            ax.legend(fontsize=7)
        off += W
    for j in range(n_seasons, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, f"Stage2_{region}.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"  Figure: {fig_path}")


def main():
    plans = get_training_plans()
    print(f"Stage2 Training ({CLIMATE_WINDOW}d window)")
    print(f"  {len(plans)} region(s)")
    for region, years in sorted(plans.items()):
        print(f"    {region}: {years}")

    for region in sorted(plans):
        train_region(region, plans[region])


if __name__ == "__main__":
    main()
