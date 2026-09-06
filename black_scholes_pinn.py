# -*- coding: utf-8 -*-

'''

Physics-Informed Neural Network for solving the Black-Scholes PDE

Black-Scholes PDE:

    dV_dt + 0.5 * sigma**2 * S**2 * d2V_dS2 + r * S * dV_dS - r * V = 0
    
Also extends to the inverse problem of recovering volatility from some synthetic
option prices. A two stage training approach is used, with sigma remaining frozen
while the forward problem is solved, and then sigma is unfrozen and also fit with
the PDE constraints

'''
#importing all modules

import torch

from torch import nn

import numpy as np

from scipy.stats import norm

import matplotlib.pyplot as plt

#setting seeds

torch.manual_seed(0)

np.random.seed(0)   # pylint: disable=no-member

#Defining all terms in the equation

K = 100                       # Strike Price

r = 0.05                      # Risk-free interest rate

true_sigma = 0.2              # True volatility (used to generate option prices)

T = 1                         # Maturity time of the option (in years)

S_max = 150                   # Domain restriction for the stock price

#Hyperparamters for the training process

learning_rate = 0.001

num_epoch1 = 10000          # Stage 1: Forward problem

num_epoch2 = 10000         # STage 2: Inverse problem

#Setting the device to train on

DEVICE = 'cpu'

class PINN(nn.Module):

    '''

    MLP mapping (S, t) to V with an additional learnable volatility parameter

    '''

    def __init__(self, n_hidden = 20, sigma_init = 0.1):

        super().__init__()

        #simple MLP with 2 hidden layers

        self.net = nn.Sequential(

            nn.Linear(2,n_hidden),

            nn.Tanh(),

            nn.Linear(n_hidden,n_hidden),

            nn.Tanh(),

            nn.Linear(n_hidden,n_hidden),

            nn.Tanh(),

            nn.Linear(n_hidden,n_hidden),

            nn.Tanh(),

            nn.Linear(n_hidden,1)
        )

        self.raw_sigma = nn.Parameter(torch.tensor(sigma_init, dtype = torch.float32))

    def sigma(self):

        '''

        Utilises softplus to ensure sigma remains positive

        '''

        return torch.nn.functional.softplus(self.raw_sigma)    # pylint: disable=not-callable

    def forward(self, S, t):

        x=torch.cat([S / S_max, t / T], dim = 1)

        return self.net(x)

def pde_residual(model, S, t):

    '''

    Computes the Black-Scholes residual at given (S, t) points using autodiff

    Returns a tensor of residual values

    '''

    S.requires_grad_(True)

    t.requires_grad_(True)

    V=model(S, t)

    # Calculating all partial derivatives with autodiff

    dV_dS  = torch.autograd.grad(V, S, grad_outputs=torch.ones_like(V),
                                 create_graph=True)[0]

    dV_dt = torch.autograd.grad(V, t, grad_outputs=torch.ones_like(V),
                                 create_graph=True)[0]

    d2V_dS2 = torch.autograd.grad(dV_dS, S, grad_outputs=torch.ones_like(dV_dS),
                                 create_graph=True)[0]

    sigma = model.sigma()

    # Calculating residual

    residual = dV_dt + 0.5 * sigma**2 * S**2 * d2V_dS2 + r * S * dV_dS - r * V

    return residual

def sample_points(n_boundary = 200, n_interior = 4000, n_terminal = 200):

    '''

    Sample random (S,t) points for each loss term: Boundary condition, PDE
    residual, and terminal condition

    This is resampled every training step

    '''

    # Boundary at S = 0

    t_b0 = T * torch.rand(n_boundary, 1, device = DEVICE)

    S_b0 = torch.zeros_like(t_b0)

    V_b0 = torch.zeros_like(t_b0)

    # Boundary at S = S_max

    t_bmax = T * torch.rand(n_boundary, 1, device = DEVICE)

    S_bmax = torch.full_like(t_bmax, S_max)

    V_bmax = S_max - K * torch.exp(-r * (T-t_bmax))

    # Interior

    t_f = T * torch.rand(n_interior, 1, device = DEVICE)

    S_f = S_max * torch.rand(n_interior, 1, device = DEVICE)

    # Terminal at t=T, S is random, payoff = max(S-K, 0)

    S_term = S_max * torch.rand(n_terminal, 1, device = DEVICE)

    t_term = torch.full_like(S_term, T)

    V_term = torch.clamp(S_term - K, min=0.0)

    return(S_f, t_f,
           S_term, t_term, V_term,
           S_b0, t_b0, V_b0,
           S_bmax, t_bmax, V_bmax)

def black_scholes_call(S, t):

    '''

    Generates synthetic market data using the true value of sigma. Used to generate
    synthetic 'observed' market data for the inverse problem, but is also used
    to validate the PINN forward solution

    '''

    tau = T - t

    tau = np.maximum(tau, 1e-8)

    d1 = (np.log(S/K) + (r + 0.5 * true_sigma**2) * tau) / (true_sigma * np.sqrt(tau))

    d2 = d1 - true_sigma * np.sqrt(tau)

    return S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)

n_obs = 50

S_obs = S_max * torch.rand(n_obs, 1, device = DEVICE)

t_obs = T * torch.rand(n_obs, 1, device = DEVICE)

V_obs_np = black_scholes_call(S_obs.numpy(), t_obs.numpy())

V_obs = torch.tensor(V_obs_np, dtype = torch.float32, device = DEVICE)

def run_epoch(model, optimizer, w_pde, w_term, w_bound, w_data, include_data):  # pylint: disable=redefined-outer-name

    '''

    Run one training step: Sample points, compute all loss terms, backpropagate,
    and update parameters. 'include_data' determines whether the data fitting loss
    is included in the loss term (for sigma recovery): False during stage 1
    (forward problem), True during stage 2 (inverse problem)

    '''

    (S_f, t_f,
     S_term, t_term, V_term,
     S_b0, t_b0, V_b0,
     S_bmax, t_bmax, V_bmax) = sample_points()

    optimizer.zero_grad()

    # Boundary loss at zero

    V_b0_pred = model(S_b0, t_b0)

    loss_b0 = torch.mean((V_b0_pred - V_b0)**2)

    # Boundary loss at max

    V_bmax_pred = model(S_bmax, t_bmax)

    loss_bmax = torch.mean((V_bmax_pred - V_bmax)**2)

    loss_bound = loss_b0 + loss_bmax

    # PDE interior loss

    res = pde_residual(model, S_f, t_f)

    loss_pde = torch.mean(res**2)

    # Terminal loss

    V_term_pred = model(S_term, t_term)

    loss_term = torch.mean((V_term_pred - V_term)**2)

    # Data loss

    V_obs_pred = model(S_obs, t_obs)

    loss_data = torch.mean((V_obs_pred - V_obs)**2)

    # Computing total loss

    loss = w_pde * loss_pde + w_bound * loss_bound + w_term * loss_term

    if include_data:
        loss = loss + w_data * loss_data

    loss.backward()

    optimizer.step()

    return (loss_pde.item(), loss_term.item(), loss_bound.item(),
            loss_data.item(), model.sigma().item())

def train(model, w_pde = 2.0, w_term = 1.0, w_bound = 1.0, w_data = 6.0,    # pylint: disable=redefined-outer-name
          epochs_phase1 = num_epoch1, epochs_phase2 = num_epoch2, lr = learning_rate):

    '''

    Two stage training:

        Phase 1: Solve the forward PDE problem with sigma frozen

        Phase 2: Unfreeze sigma and fit the PDE and synthetic market data to
                 recover sigma

    '''

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []

    # Stage 1: solving the forward problem (sigma frozen)

    model.raw_sigma.requires_grad_(False)

    for epoch in range(epochs_phase1):

        loss_pde, loss_term, loss_bound, loss_data, sigma_val = run_epoch(
            model, optimizer, w_pde, w_term, w_bound, w_data, include_data = False)

        history.append((loss_pde, loss_bound, loss_term, loss_data, sigma_val))

        if epoch % 500 == 0:

            print(f'phase 1 epoch {epoch:5d} | pde {loss_pde:.5f} |'
                  f'term {loss_term:.5f} | bound {loss_bound:.5f} | data {loss_data:.5f}')

    # Stage 2: solving the inverse problem (sigma unfrozen)

    model.raw_sigma.requires_grad_(True)

    for epoch in range(epochs_phase2):

        loss_pde, loss_term, loss_bound, loss_data, sigma_val = run_epoch(
            model, optimizer, w_pde, w_term, w_bound, w_data, include_data = True)

        history.append((loss_pde, loss_bound, loss_term, loss_data, sigma_val))

        if epoch % 500 == 0:

            print(f'phase 2 epoch {epoch:5d} | pde {loss_pde:.5f} |'
                  f'term {loss_term:.5f} | bound {loss_bound:.5f} | data {loss_data:.5f} |'
                  f'sigma {sigma_val:.5f}')

    return w_pde, w_bound, w_term, w_data, history

if __name__ == '__main__':

    trained_model = PINN().to(DEVICE)

    w_pde, w_term, w_bound, w_data, training_history = train(trained_model)

    # Evaluate at t = 0 for range of S

    S_test = np.linspace(1, S_max, 200)

    t_test = np.zeros_like(S_test)

    with torch.no_grad():

        S_t = torch.tensor(S_test, dtype = torch.float32, device = DEVICE).view(-1,1)

        t_t = torch.tensor(t_test, dtype = torch.float32, device = DEVICE).view(-1,1)

        V_pinn = trained_model(S_t, t_t).cpu().numpy().flatten()

    V_exact = black_scholes_call(S_test, t_test)

    fig, axes = plt.subplots(1, 2, figsize=(12,5))

    # PINN solution vs analytical solution

    axes[0].plot(S_test, V_exact, label = 'Analytical Black-Scholes', lw = 2)

    axes[0].plot(S_test, V_pinn, '--', label = 'PINN', lw = 2)

    axes[0].set_xlabel('Stock Price S')

    axes[0].set_ylabel('Option Value V')

    axes[0].set_title('PINN vs Analytical Solution')

    axes[0].legend()

    # PINN error when comapring to analytical solution

    axes[1].plot(S_test, np.abs(V_pinn - V_exact))

    axes[1].set_xlabel('Stock Price S')

    axes[1].set_ylabel('Absolute Error')

    axes[1].set_title('PINN Error vs Closed form solution')

    plt.tight_layout()

    plt.savefig('analytical_comparison.png', dpi = 150)

    plt.show()

    # Sigma estimation curve over training

    print(f'True sigma: {true_sigma}, Learned sigma: {trained_model.sigma().item():.5f}')

    sigma_history = [h[4] for h in training_history]

    plt.figure()

    plt.plot(sigma_history, label = 'Learned sigma')

    plt.axhline(true_sigma, color = 'red', linestyle = '--', label = 'True sigma')

    plt.xlabel('Epoch')

    plt.ylabel('Sigma')

    plt.title('Sigma recovery during training')

    plt.legend()

    plt.savefig('sigma_recovery.png', dpi = 150)

    plt.show()

    # Loss curves over training

    pde_history = [h[0] for h in training_history]

    term_history = [h[1] for h in training_history]

    bound_history = [h[2] for h in training_history]

    data_history = [h[3] for h in training_history]

    epochs_scale = range(0, len(training_history), 50)

    plt.figure(figsize = (9, 6))

    plt.plot(epochs_scale, pde_history[::50], label = 'PDE loss', lw = 1, alpha = 0.8)

    plt.plot(epochs_scale, bound_history[::50], label = 'Boundary loss', lw = 1, alpha = 0.8)

    plt.plot(epochs_scale, term_history[::50], label = 'Terminal loss', lw = 1, alpha = 0.8)

    plt.plot(epochs_scale, data_history[::50], label = 'Data loss', lw = 1, alpha = 0.8)

    plt.axvline(x = num_epoch1, color = 'black', linestyle = ':', label = 'Phase 1 to Phase 2')

    plt.yscale('log')

    plt.xlabel('Epoch')

    plt.ylabel('Loss (log scale)')

    plt.title('Loss components during training')

    weight_info = (f'w$_{{pde}}$ = {w_pde}      '
                   f'w$_{{bound}}$ = {w_bound}      '
                   f'w$_{{term}}$ = {w_term}      '
                   f'w$_{{data}}$ = {w_data}')

    plt.figtext(0.5, -0.02, weight_info, ha = 'center', va = 'top', fontsize=12,
             color='black')

    plt.legend()

    plt.tight_layout()

    plt.savefig('loss_curves.png', dpi = 150, bbox_inches='tight')

    plt.show()

    # PDE residual heatmap

    n_S_res = 100

    n_t_res = 100

    S_vals_res = np.linspace(1, S_max - 1, n_S_res)

    t_vals_res = np.linspace(0, T-1e-3, n_t_res)

    S_grid_res, t_grid_res = np.meshgrid(S_vals_res, t_vals_res)

    S_res = torch.tensor(S_grid_res.flatten(), dtype = torch.float32, device = DEVICE).view(-1, 1)

    t_res = torch.tensor(t_grid_res.flatten(), dtype = torch.float32, device = DEVICE).view(-1, 1)

    res_vals = pde_residual(trained_model, S_res, t_res)

    res_grid = res_vals.detach().numpy().reshape(S_grid_res.shape)

    plt.figure(figsize=(9, 6))

    plt.colorbar(plt.contourf(S_grid_res, t_grid_res, np.abs(res_grid),
                              levels = 50, cmap = 'inferno'), label = '|PDE Residual|')

    plt.xlabel('Stock Price S')

    plt.ylabel('Time t')

    plt.title('PDE Residual magnitude across domain')

    plt.tight_layout()

    plt.savefig('residual_heatmap.png', dpi = 150)

    plt.show()
