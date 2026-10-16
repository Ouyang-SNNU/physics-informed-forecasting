"""
UDE Stage 1 v1 — TCN 架构 + 56 天窗口 β 预测训练

相比 v0 (UDE_S1.py) 的改动:
  - CLIMATE_WINDOW: 35 → 56 天
  - 网络: DirectBetaNet → TransNet (TCN head)
  - 模型保存: {Region}_UDE_S1_v1.pt
  - 不足窗口警告: 当季节长度 < CLIMATE_WINDOW 时打印警告

保留 v0 逻辑:
  - 每地区独立训练, 独立 Scaler
  - 直接 MSE(β_pred, β_pinns) 训练
  - 20k epochs, lr=1e-3

输出: model/{Region}_UDE_S1_v1.pt + model/scaler_{Region}.pkl
"""
import os, glob
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler
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
EPOCHS = 20000
LR = 1e-3

CLIMATE_DIR = "../data/processed_climate_data"
PINNS_DIR = "../data/PINNs_results"
Epi_DIR = "../data/ILI_data"
SAVE_DIR = "../model"


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


def build_season(region, year, clim_df, scaler=None, fit_scaler=False):
    epi_path = os.path.join(Epi_DIR, f"{region}_{year}.csv")
    pnns_path = os.path.join(PINNS_DIR, f"{region}_{year}.csv")
    if not (os.path.exists(epi_path) and os.path.exists(pnns_path)):
        return None

    start, end = get_season_date_range(epi_path)
    clim_start = start - pd.Timedelta(days=CLIMATE_WINDOW)
    clim_raw = clim_df.loc[clim_start:end][PARAM_COLS].values.astype(np.float32)

    if fit_scaler:
        return {'clim_raw': clim_raw, 'start': start, 'year': year,
                'pnns_path': pnns_path}

    clim_scaled = scaler.transform(clim_raw)

    df_pinns = pd.read_csv(pnns_path)
    beta_pinns = df_pinns['beta'].values.astype(np.float32)

    min_required = CLIMATE_WINDOW + 1
    if len(clim_scaled) < min_required:
        print(f"  WARNING {region}_{year}: climate data too short "
              f"({len(clim_scaled)} rows, need >= {min_required}), skip")
        return None

    L = min(len(beta_pinns), len(clim_scaled) - CLIMATE_WINDOW)
    if L < 7:
        print(f"  WARNING {region}_{year}: season too short ({L} days), skip")
        return None

    clim_scaled = clim_scaled[:L + CLIMATE_WINDOW]
    beta_pinns = beta_pinns[:L]
    W = L // 7

    X_clim, X_seas = [], []
    for d in range(L):
        X_clim.append(clim_scaled[d:d + CLIMATE_WINDOW].flatten())
        cur = start + pd.Timedelta(days=d)
        doy = cur.timetuple().tm_yday
        X_seas.append([np.sin(2*np.pi*doy/365), np.cos(2*np.pi*doy/365),
                        (d//7)/max(W,1)])

    return {'X_clim': torch.tensor(np.array(X_clim, dtype=np.float32), device=device),
            'X_seas': torch.tensor(np.array(X_seas, dtype=np.float32), device=device),
            'beta_pinns': torch.tensor(beta_pinns, device=device),
            'L': L, 'W': W,
            'label': f"{region}_{year}"}


def train_region(region, train_years):
    print(f"\n{'='*55}\n  {region}  ({len(train_years)} seasons: {train_years})\n{'='*55}")

    clim_path = os.path.join(CLIMATE_DIR, f"{region}_Climate_Daily.csv")
    clim_df = pd.read_csv(clim_path, parse_dates=['datetime'],
                          index_col='datetime').sort_index()

    # === 1. Fit Scaler ===
    all_clim, raw_seasons = [], []
    for yr in train_years:
        s = build_season(region, yr, clim_df, fit_scaler=True)
        if s:
            all_clim.append(s['clim_raw'])
            raw_seasons.append(s)
    if not all_clim:
        print(f"  WARNING no valid seasons")
        return

    scaler = StandardScaler()
    scaler.fit(np.concatenate(all_clim, axis=0))
    scaler_path = os.path.join(SAVE_DIR, f"scaler_{region}.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"  Scaler: {scaler_path}")

    # === 2. Build normalized seasons ===
    seasons = []
    for s in raw_seasons:
        built = build_season(region, s['year'], clim_df, scaler=scaler)
        if built:
            seasons.append(built)
    n_seasons = len(seasons)
    total_days = sum(s['L'] for s in seasons)
    print(f"  {n_seasons} seasons, {total_days} days")

    # === 3. Train  ===
    model = TransNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    model_path = os.path.join(SAVE_DIR, f"{region}_Stage1.pt")

    for ep in range(EPOCHS):
        model.train(); optimizer.zero_grad()
        total_loss = 0

        for s in seasons:
            beta_pred = model(s['X_clim'], s['X_seas'])
            total_loss += torch.mean((beta_pred - s['beta_pinns']) ** 2)

        total_loss.backward()
        optimizer.step()

        if (ep + 1) % 500 == 0:
            print(f"  [{ep+1:5d}/{EPOCHS}]  beta MSE: {total_loss.item()/n_seasons:.6e}")
            if (ep + 1) % 5000 == 0:
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()}, model_path)

    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, model_path)
    print(f"  OK {model_path}")

    # === 4. Visualization ===
    model.eval()
    ncols = min(5, n_seasons); nrows = (n_seasons + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), dpi=200)
    axes = axes.flatten() if n_seasons > 1 else [axes]
    with torch.no_grad():
        for i, s in enumerate(seasons):
            beta_pred = model(s['X_clim'], s['X_seas'])
            b_r = beta_pred.cpu().numpy()
            b_p = s['beta_pinns'].cpu().numpy()
            axes[i].plot(b_p, color="#1F77B4", lw=1.5, label="PINNs")
            axes[i].plot(b_r, "--", color="#D62728", lw=1.5, label="TCN-v1")
            axes[i].set_title(s['label'], fontsize=9)
            if i == 0: axes[i].legend(fontsize=7)
    for j in range(n_seasons, len(axes)): axes[j].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join("../figures", f"Stage1_{region}.png"), dpi=600)
    plt.close()


def main():
    plans = get_training_plans()
    print(f"Stage_1 Training ({CLIMATE_WINDOW}d window)")
    print(f"  {len(plans)} region(s)")
    for region, years in sorted(plans.items()):
        print(f"    {region}: {years}")

    for region in sorted(plans):
        train_region(region, plans[region])


if __name__ == "__main__":
    main()
