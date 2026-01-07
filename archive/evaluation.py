
import os
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


from archive.utils import make_business_calendar
from archive.utils import simulate_future_gbm
from archive.utils import arithmetic_average
from archive.utils import build_library_balanced

from archive.model1_MLP import MLP
from archive.model1_MLP import pde_residual
from archive.model1_MLP import minibatch


np.random.seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = MLP().to(DEVICE)

def price_asian_pinn(F0, K, r, sigma, T_years, tau=0.999, A0=None):
    if A0 is None:
        A0 = F0
    x = torch.tensor([[tau, F0/F0, A0/F0, K/F0, r, sigma, T_years]],
                     dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        Vn = model(x).item()
    return Vn * F0

def pinn_price_batch(df, tau=0.999):
    model.eval()
    with torch.no_grad():
        F0 = df["F0"].values.astype(np.float32)
        K  = df["K"].values.astype(np.float32)
        r  = df["r"].values.astype(np.float32)
        s  = df["sigma"].values.astype(np.float32)
        T  = df["T_years"].values.astype(np.float32)
        A0 = F0.copy()

        X = np.stack([
            np.full_like(F0, tau, dtype=np.float32),
            F0 / F0,
            A0 / F0,
            K  / F0,
            r, s, T
        ], axis=1)
        X = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        Vn = model(X).cpu().numpy().reshape(-1)
        V  = Vn * F0
        return V

def evaluate_pinn_vs_mc(df_val, tau=0.999, sample_for_curve=300):
    price_pinn = pinn_price_batch(df_val, tau=tau)
    price_mc   = df_val["price_mc"].values.astype(np.float32)

    mae  = mean_absolute_error(price_mc, price_pinn)
    mse  = mean_squared_error(price_mc, price_pinn)
    rmse = np.sqrt(mse)
    r2   = r2_score(price_mc, price_pinn)
    print(f"\nVAL – MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")

    # Parity plot
    plt.figure(figsize=(6,6))
    plt.scatter(price_mc, price_pinn, s=12, alpha=0.5)
    lims = [0, max(price_mc.max(), price_pinn.max()) * 1.05]
    plt.plot(lims, lims, linestyle="--")
    plt.xlim(lims); plt.ylim(lims)
    plt.xlabel("MC price")
    plt.ylabel("PINN price")
    plt.title("Parity plot: PINN vs Monte Carlo (balanced validation)")
    plt.grid(True)
    plt.show()

    # Curva prezzo vs K a F0 ~ costante
    f0_med = df_val["F0"].median()
    tol = max(0.5, 0.02 * f0_med)
    sub = df_val[df_val["F0"].between(f0_med - tol, f0_med + tol)].copy()
    if len(sub) > sample_for_curve:
        sub = sub.sample(sample_for_curve, random_state=42)
    sub = sub.sort_values("K")
    sub_pinn = pinn_price_batch(sub, tau=tau)

    plt.figure(figsize=(7,5))
    plt.plot(sub["K"].values, sub_pinn, label="PINN", linewidth=2)
    plt.scatter(sub["K"].values, sub["price_mc"].values, s=12, alpha=0.7, label="MC")
    plt.xlabel("Strike K")
    plt.ylabel("Option price")
    plt.title(f"Price vs K  (F0 ≈ {f0_med:.2f})")
    plt.legend()
    plt.grid(True)
    plt.show()

    return {"MAE":mae, "RMSE":rmse, "R2":r2}
