
import time
import math
import torch
import torch.nn as nn
from archive.datasets import sample_domain
from archive.datasets import sample_TC
from archive.datasets import sample_BC_S0
from archive.datasets import sample_BC_Smax
from archive.datasets import sample_domain
from archive.datasets import sample_BC_Imax

SEED = 42
torch.manual_seed(SEED)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32


class PINN(nn.Module):
   def __init__(self, in_dim=5, width=160, depth=4):
       super().__init__()
       layers = []
       layers.append(nn.Linear(in_dim, width))
       layers.append(nn.GELU())
       for _ in range(depth - 1):
           layers.append(nn.Linear(width, width))
           layers.append(nn.GELU())
       layers.append(nn.Linear(width, 1))
       self.net = nn.Sequential(*layers)


       # small init helps stability
       for m in self.net:
           if isinstance(m, nn.Linear):
               nn.init.xavier_uniform_(m.weight)
               nn.init.zeros_(m.bias)


   def forward(self, x):
       return self.net(x)
   

def pde_residual(model, X):
   """
   X = [S, I, t, r, sigma] all normalized except r,sigma,t.
   """
   X = X.clone().detach().requires_grad_(True)
   V = model(X)  # (N,1)


   grads = torch.autograd.grad(
       V, X, grad_outputs=torch.ones_like(V),
       create_graph=True, retain_graph=True
   )[0]
   V_S = grads[:, 0:1]
   V_I = grads[:, 1:2]
   V_t = grads[:, 2:3]
   r   = X[:, 3:4]
   sig = X[:, 4:5]
   S   = X[:, 0:1]


   # second derivative w.r.t S
   V_SS = torch.autograd.grad(
       V_S, X, grad_outputs=torch.ones_like(V_S),
       create_graph=True, retain_graph=True
   )[0][:, 0:1]


   res = V_t + 0.5 * (sig**2) * (S**2) * V_SS + r * S * V_S + S * V_I - r * V
   return res


import torch
import numpy as np
import math

def pide_residual(model, X):
    """
    PIDE residual for an arithmetic Asian option under jump–diffusion dynamics.

    X = [S, I, t, r, sigma]
    - S, I: normalized
    - t, r, sigma: physical
    """

    # Jump model parameters (Merton jump–diffusion)
    lambda_jump = 2.188
    mu_J        = 0.0196
    sigma_J     = 0.1817

    # Jump compensator κ = E[e^Y - 1]
    kappa = math.exp(mu_J + 0.5 * sigma_J**2) - 1.0

    # Gauss–Hermite quadrature (NumPy → Torch)
    
    n_gh = 8
    y_nodes_np, y_weights_np = np.polynomial.hermite.hermgauss(n_gh)

    # Rescale for Y ~ N(mu_J, sigma_J^2)
    y_nodes_np   = np.sqrt(2.0) * sigma_J * y_nodes_np + mu_J
    y_weights_np = y_weights_np / np.sqrt(np.pi)

    y_nodes   = torch.tensor(y_nodes_np, dtype=X.dtype, device=X.device)
    y_weights = torch.tensor(y_weights_np, dtype=X.dtype, device=X.device)

    # Jump amplitude γ(y) = y
    def gamma_fn(y):
        return y

    
    # Enable autograd
    
    X = X.clone().detach().requires_grad_(True)
    V = model(X)

    
    # First derivatives
    grads = torch.autograd.grad(
        V,
        X,
        grad_outputs=torch.ones_like(V),
        create_graph=True,
        retain_graph=True
    )[0]

    V_S = grads[:, 0:1]
    V_I = grads[:, 1:2]
    V_t = grads[:, 2:3]

    S   = X[:, 0:1]
    I   = X[:, 1:2]
    t   = X[:, 2:3]
    r   = X[:, 3:4]
    sig = X[:, 4:5]

    
    # Second derivative wrt S
    
    V_SS = torch.autograd.grad(
        V_S,
        X,
        grad_outputs=torch.ones_like(V_S),
        create_graph=True,
        retain_graph=True
    )[0][:, 0:1]

    
    # Jump integral (deterministic quadrature)
    
    jump_integral = 0.0
    for y, w in zip(y_nodes, y_weights):
        S_jump = S * torch.exp(gamma_fn(y))
        X_jump = torch.cat([S_jump, I, t, r, sig], dim=1)
        V_jump = model(X_jump)
        jump_integral += w * V_jump
    

    # PIDE residual
    
    res = (
        V_t
        + 0.5 * sig**2 * S**2 * V_SS
        + (r - lambda_jump * kappa) * S * V_S
        + S * V_I
        - (r + lambda_jump) * V
        + lambda_jump * jump_integral
    )

    return res



def train_pinn(
   S_max, I_max, r_max, sigma_max,
   width=160,
   depth=4,
   n_epochs=200_000,     # paper uses 500k; set higher if you want
   lr0=1e-3,
   Np=1000,
   n_bc_axis=100,        # paper: n=100 per axis
   w_pde=1.0,
   print_every=2000
):
   model = PINN(in_dim=5, width=width, depth=depth).to(DEVICE)
   opt = torch.optim.Adam(model.parameters(), lr=lr0)


   # cosine annealing to 0 (single cycle)
   sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=0.0)


   t0 = time.time()
   for ep in range(1, n_epochs + 1):
       model.train()
       opt.zero_grad(set_to_none=True)


       # Collocation points
       Xp = sample_domain(Np, S_max, I_max, r_max, sigma_max)
       res = pide_residual(model, Xp)
       loss_pde = torch.mean(res**2)


       # Data points (TC + BCs): 4*n vectors total as paper (here we do 4 blocks)
       X_tc, Y_tc = sample_TC(n_bc_axis, S_max, I_max, r_max, sigma_max)
       X_s0, Y_s0 = sample_BC_S0(n_bc_axis, S_max, I_max, r_max, sigma_max)
       X_sM, Y_sM = sample_BC_Smax(n_bc_axis, S_max, I_max, r_max, sigma_max)
       X_iM, Y_iM = sample_BC_Imax(n_bc_axis, S_max, I_max, r_max, sigma_max)


       Xd = torch.cat([X_tc, X_s0, X_sM, X_iM], dim=0)
       Yd = torch.cat([Y_tc, Y_s0, Y_sM, Y_iM], dim=0)


       pred_d = model(Xd)
       loss_d = torch.mean((pred_d - Yd)**2)


       loss = w_pde * loss_pde + loss_d
       loss.backward()
       opt.step()
       sched.step()


       if ep % print_every == 0:
           elapsed = time.time() - t0
           print(f"ep={ep:>7d} | loss={loss.item():.3e} | pde={loss_pde.item():.3e} | data={loss_d.item():.3e} | lr={sched.get_last_lr()[0]:.2e} | {elapsed/60:.1f} min")


   return model