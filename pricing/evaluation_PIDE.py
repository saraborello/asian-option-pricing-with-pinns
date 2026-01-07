
import math
import time
import numpy as np
import pandas as pd
import yfinance as yf


import torch
import torch.nn as nn


from montecarlo import mc_asian_call_arith
from montecarlo import pinn_price_real

from montecarlo_PIDE import mc_asian_call_arith_jump



import plotly.graph_objects as go
import numpy as np

import time
import math
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


lambda_jump = 2.188
mu_J        = 0.0196
sigma_J     = 0.1817

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32


def evaluate_mae_pinn_vs_mc(
   model,
   S_grid_real,
   K_real,
   r,
   sigma,
   T=1.0,
   n_steps_mc=252,
   n_paths_mc=100_000
):
   """
   MAE = mean_i | V_MC(S_i) - V_PINN(S_i) |
   """
   mc_prices = []
   pinn_prices = []


   print("\nRunning MC reference pricing...")
   for S in S_grid_real:
       mc_price, _ = mc_asian_call_arith_jump(
           S0_real=S,
           K_real=K_real,
           r=r,
           sigma=sigma,
           lambda_jump = lambda_jump, 
           mu_J = mu_J , 
           sigma_J = sigma_J,
           T=T,
           n_steps=n_steps_mc,
           n_paths=n_paths_mc,
           antithetic=True
       )
       mc_prices.append(mc_price)


   print("Evaluating PINN prices...")
   for S in S_grid_real:
       p = pinn_price_real(
           model=model,
           S0_real=S,
           K_real=K_real,
           r=r,
           sigma=sigma,
           t0=0.0
       )
       pinn_prices.append(p)


   mc_prices = np.array(mc_prices)
   pinn_prices = np.array(pinn_prices)


   mae = np.mean(np.abs(mc_prices - pinn_prices))


   return mae, mc_prices, pinn_prices



def parity_plot_plotly(mc_prices, pinn_prices, title="Parity plot: PINN vs Monte Carlo"):
    mc_prices = np.asarray(mc_prices)
    pinn_prices = np.asarray(pinn_prices)

    lim_min = min(mc_prices.min(), pinn_prices.min())
    lim_max = max(mc_prices.max(), pinn_prices.max())

    fig = go.Figure()

    # Scatter: MC vs PINN
    fig.add_trace(
        go.Scatter(
            x=mc_prices,
            y=pinn_prices,
            mode="markers",
            name="Observations",
            marker=dict(size=7, opacity=0.8),
        )
    )

    # Parity line y = x
    fig.add_trace(
        go.Scatter(
            x=[lim_min, lim_max],
            y=[lim_min, lim_max],
            mode="lines",
            name="y = x",
            line=dict(dash="dash", width=1),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Monte Carlo price",
        yaxis_title="PINN price",
        xaxis=dict(range=[lim_min, lim_max], scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[lim_min, lim_max]),
        template="plotly_white",
        width=600,
        height=600,
    )

    fig.show()






def numerical_delta(price_fn, S, h=1.0):
    return (price_fn(S + h) - price_fn(S - h)) / (2.0 * h)



def mc_delta_crn(
    S0,
    K,
    r,
    sigma,
    T,
    n_steps,
    n_paths,
    h,
):
    dt = T / n_steps
    disc = math.exp(-r * T)

    half = n_paths // 2
    Z = np.random.normal(size=(half, n_steps))
    Z = np.vstack([Z, -Z])

    drift = (r - 0.5 * sigma**2) * dt
    vol = sigma * math.sqrt(dt)

    def price(S_init):
        S = np.full(Z.shape[0], S_init)
        acc = np.zeros_like(S)
        for k in range(n_steps):
            S *= np.exp(drift + vol * Z[:, k])
            acc += S
        A = acc / n_steps
        return disc * np.maximum(A - K, 0.0).mean()

    return (price(S0 + h) - price(S0 - h)) / (2.0 * h)


# Monte Carlo reference (price, delta, time)

def compute_mc_reference(
    S_grid,
    K,
    r,
    sigma,
    T,
    n_steps,
    n_paths,
    h_delta,
):
    prices, deltas, times = [], [], []

    for S in S_grid:
        t0 = time.perf_counter()
        p, _ = mc_asian_call_arith_jump(
            S0_real=S,
            K_real=K,
            r=r,
            sigma=sigma,
            lambda_jump = lambda_jump, 
            mu_J = mu_J , 
            sigma_J = sigma_J,
            T=T,
            n_steps=n_steps,
            n_paths=n_paths,
            antithetic=True,
        )
        times.append(time.perf_counter() - t0)
        prices.append(p)

        d = mc_delta_crn(
            S0=S,
            K=K,
            r=r,
            sigma=sigma,
            T=T,
            n_steps=n_steps,
            n_paths=n_paths,
            h=h_delta,
        )
        deltas.append(d)

    return {
        "prices": np.asarray(prices),
        "deltas": np.asarray(deltas),
        "times": np.asarray(times),
    }



# PINN outputs (price, delta, time)

def compute_pinn_outputs(
    model,
    S_grid,
    K,
    r,
    sigma,
    h_delta,
):
    prices, deltas, times = [], [], []

    for S in S_grid:
        t0 = time.perf_counter()
        p = pinn_price_real(model, S, K, r, sigma, 0.0)
        times.append(time.perf_counter() - t0)
        prices.append(p)

        d = numerical_delta(
            lambda x: pinn_price_real(model, x, K, r, sigma, 0.0),
            S,
            h=h_delta,
        )
        deltas.append(d)

    return {
        "prices": np.asarray(prices),
        "deltas": np.asarray(deltas),
        "times": np.asarray(times),
    }



# Plot: price – delta – parity (single row)

def plot_price_delta_parity(
    S,
    mc_prices,
    pinn_prices,
    mc_deltas,
    pinn_deltas,
):
    pmax = max(mc_prices.max(), pinn_prices.max())

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[
            "Asian option price",
            "Asian option delta",
            "Parity plot: PINN vs Monte Carlo",
        ],
    )

    # Price
    fig.add_trace(go.Scatter(x=S, y=mc_prices, name="Monte Carlo"), 1, 1)
    fig.add_trace(go.Scatter(x=S, y=pinn_prices, name="PINN", line=dict(dash="dash")), 1, 1)

    # Delta
    fig.add_trace(go.Scatter(x=S, y=mc_deltas, showlegend=False), 1, 2)
    fig.add_trace(go.Scatter(x=S, y=pinn_deltas, showlegend=False, line=dict(dash="dash")), 1, 2)

    # Parity
    fig.add_trace(go.Scatter(x=mc_prices, y=pinn_prices, mode="markers", showlegend=False), 1, 3)
    fig.add_trace(go.Scatter(x=[0, pmax], y=[0, pmax], mode="lines", line=dict(dash="dash"), showlegend=False), 1, 3)

    fig.update_xaxes(title_text="S", row=1, col=1)
    fig.update_yaxes(title_text="V", row=1, col=1)

    fig.update_xaxes(title_text="S", row=1, col=2)
    fig.update_yaxes(title_text="∂V / ∂S", row=1, col=2)

    fig.update_xaxes(title_text="Monte Carlo price", range=[0, pmax], row=1, col=3)
    fig.update_yaxes(title_text="PINN price", range=[0, pmax], scaleanchor="x", row=1, col=3)

    fig.update_layout(
        template="plotly_white",
        width=1500,
        height=420,
        legend=dict(orientation="h", x=0.5, y=-0.25, xanchor="center"),
    )

    fig.show()



# Master evaluation


def run_full_evaluation_PIDE(
    model,
    S_grid,
    K,
    r,
    sigma,
    T=1.0,
    n_steps=252,
    n_paths=200_000,
    h_delta=1.0,
):
    mc = compute_mc_reference(S_grid, K, r, sigma, T, n_steps, n_paths, h_delta)
    pinn = compute_pinn_outputs(model, S_grid, K, r, sigma, h_delta)

    mae = float(np.mean(np.abs(mc["prices"] - pinn["prices"])))

    print(f"MAE (PINN vs MC): {mae:.6f}")
    print(f"Speed-up (MC / PINN): {(mc['times'] / pinn['times']).mean():.1f}x")

    plot_price_delta_parity(
        S_grid,
        mc["prices"],
        pinn["prices"],
        mc["deltas"],
        pinn["deltas"],
    )

    return {"mae": mae, "mc": mc, "pinn": pinn}
