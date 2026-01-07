
import torch
import torch.nn as nn
import numpy as np
import math

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32



# Monte Carlo pricing (real scale) + PINN pricing (rescaled)

def mc_asian_call_arith(
   S0_real, K_real, r, sigma, T=1.0,
   n_steps=252, n_paths=200_000, antithetic=True
):
   dt = T / n_steps
   disc = math.exp(-r * T)


   if antithetic:
       half = n_paths // 2
       Z = np.random.normal(size=(half, n_steps))
       Z = np.vstack([Z, -Z])
   else:
       Z = np.random.normal(size=(n_paths, n_steps))


   S = np.full((Z.shape[0],), S0_real, dtype=float)
   sumS = np.zeros_like(S)


   drift = (r - 0.5 * sigma**2) * dt
   vol = sigma * math.sqrt(dt)


   for k in range(n_steps):
       S *= np.exp(drift + vol * Z[:, k])
       sumS += S


   A = sumS / n_steps
   payoff = np.maximum(A - K_real, 0.0)
   price = disc * payoff.mean()
   stderr = disc * payoff.std(ddof=1) / math.sqrt(len(payoff))
   return price, stderr


@torch.no_grad()
def pinn_price_real(model, S0_real, K_real, r, sigma, t0=0.0):
   """
   Uses scaling identity (paper):
     V_real = K_real * V_pinn(S/K, I/K, t, r, sigma)   with PINN trained at K=1
   At time t=0 for continuous average, I(0)=0.
   """
   S_tilde = torch.tensor([[S0_real / K_real]], device=DEVICE, dtype=DTYPE)
   I_tilde = torch.tensor([[0.0]], device=DEVICE, dtype=DTYPE)
   t = torch.tensor([[t0]], device=DEVICE, dtype=DTYPE)
   rr = torch.tensor([[r]], device=DEVICE, dtype=DTYPE)
   ss = torch.tensor([[sigma]], device=DEVICE, dtype=DTYPE)
   X = torch.cat([S_tilde, I_tilde, t, rr, ss], dim=1)
   V_tilde = model(X).item()
   return K_real * V_tilde


