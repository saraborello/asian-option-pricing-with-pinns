import torch
import numpy as np
import math

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32


def mc_asian_call_arith_jump(
    S0_real, K_real, r, sigma,
    lambda_jump, mu_J, sigma_J,
    T=1.0,
    n_steps=252,
    n_paths=200_000,
    antithetic=True
):
    dt = T / n_steps
    disc = math.exp(-r * T)

    # Brownian part
    if antithetic:
        half = n_paths // 2
        Z = np.random.normal(size=(half, n_steps))
        Z = np.vstack([Z, -Z])
    else:
        Z = np.random.normal(size=(n_paths, n_steps))

    n_sim = Z.shape[0]

    # Poisson jumps
    N_jump = np.random.poisson(lambda_jump * dt, size=(n_sim, n_steps))

    S = np.full((n_sim,), S0_real, dtype=float)
    sumS = np.zeros_like(S)

    # Drift compensation
    kappa = math.exp(mu_J + 0.5 * sigma_J**2) - 1.0
    drift = (r - lambda_jump * kappa - 0.5 * sigma**2) * dt
    vol = sigma * math.sqrt(dt)

    for k in range(n_steps):
        # diffusion
        S *= np.exp(drift + vol * Z[:, k])

        # jumps
        jump_mask = N_jump[:, k] > 0
        if np.any(jump_mask):
            Y = np.random.normal(mu_J, sigma_J, size=jump_mask.sum())
            S[jump_mask] *= np.exp(Y)

        sumS += S

    A = sumS / n_steps
    payoff = np.maximum(A - K_real, 0.0)

    price = disc * payoff.mean()
    stderr = disc * payoff.std(ddof=1) / math.sqrt(len(payoff))

    return price, stderr


# PINN pricing (real scale, K-normalized training)


@torch.no_grad()
def pinn_price_real(model, S0_real, K_real, r, sigma, t0=0.0):
    S_tilde = torch.tensor([[S0_real / K_real]], device=DEVICE, dtype=DTYPE)
    I_tilde = torch.tensor([[0.0]], device=DEVICE, dtype=DTYPE)
    t = torch.tensor([[t0]], device=DEVICE, dtype=DTYPE)
    rr = torch.tensor([[r]], device=DEVICE, dtype=DTYPE)
    ss = torch.tensor([[sigma]], device=DEVICE, dtype=DTYPE)

    X = torch.cat([S_tilde, I_tilde, t, rr, ss], dim=1)
    V_tilde = model(X).item()

    return K_real * V_tilde



# Global evaluation (NO region filtering)

def evaluate_pinn_vs_mc_jump(
    model,
    S_grid,
    K, r, sigma, T,
    lambda_jump, mu_J, sigma_J,
    n_paths_mc=200_000
):
    pinn_prices = []
    mc_prices   = []

    for S in S_grid:
        p_pinn = pinn_price_real(model, S, K, r, sigma)

        p_mc, _ = mc_asian_call_arith_jump(
            S, K, r, sigma,
            lambda_jump, mu_J, sigma_J,
            T=T,
            n_paths=n_paths_mc
        )

        pinn_prices.append(p_pinn)
        mc_prices.append(p_mc)

    pinn_prices = np.array(pinn_prices)
    mc_prices   = np.array(mc_prices)

    mae  = np.mean(np.abs(pinn_prices - mc_prices))
    rmse = np.sqrt(np.mean((pinn_prices - mc_prices)**2))

    return {
        "S_grid": S_grid,
        "PINN": pinn_prices,
        "MC": mc_prices,
        "MAE": mae,
        "RMSE": rmse
    }
