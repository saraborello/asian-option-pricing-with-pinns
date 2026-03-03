
# Asian Option Pricing with Physics-Informed Neural Networks

The pricing of Asian options represents a challenging problem in quantitative finance due to their path-dependent payoff structure and the general absence of closed-form analytical solutions. As a consequence, practitioners typically rely on numerical methods. Among these, Monte Carlo simulation is widely adopted and provides accurate estimates, yet it is computationally intensive, particularly in settings requiring repeated valuation, sensitivity analysis, or real-time pricing.

<p align="center">
  <img width="857" height="398" alt="Screenshot 2026-03-03 alle 09 47 24" src="https://github.com/user-attachments/assets/38c350f3-ed10-4408-8771-e4f1cbad80c0" />
</p>

This thesis investigates the application of deep learning techniques to the pricing of Asian options through the framework of Physics-Informed Neural Networks (PINNs). By embedding the governing pricing equations directly into the loss function, PINNs enable the neural network to learn the solution while explicitly enforcing the underlying financial dynamics.

<p align="center">
  <img width="885" height="184" alt="Screenshot 2026-03-03 alle 09 47 55" src="https://github.com/user-attachments/assets/298f01b5-1e94-4a6b-9281-39c510d8be50" />
</p>

Both partial differential equation (PDE) formulations and partial integro-differential equation (PIDE) formulations are examined. The latter naturally arise when incorporating jump components in the underlying asset price dynamics, extending the classical diffusion-based framework.

<p align="center">
  <img width="477" height="385" alt="Screenshot 2026-03-03 alle 09 50 00" src="https://github.com/user-attachments/assets/fe0696f6-77d8-406e-b880-e83635d92bc1" />
</p>

The proposed methodology is validated through an empirical case study in the energy market, focusing on Asian options written on West Texas Intermediate (WTI) crude oil. The results indicate that PINNs provide a flexible and computationally efficient alternative to traditional pricing approaches, particularly for complex derivatives where classical numerical techniques may encounter scalability or performance limitations.

<p align="center">
  <img width="786" height="365" alt="Screenshot 2026-03-03 alle 09 45 24" src="https://github.com/user-attachments/assets/49d39e5c-5dfa-4f5c-85fb-9585a18cb966" />
</p>


