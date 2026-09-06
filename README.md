# Black-Scholes PINN

A Physics-Informed Neural Network built to solve the Black-Scholes partial differential equation. The PINN has also been extended to an inverse problem that recovers the volatility from synthetic market data.

## Background

The Black-Scholes equation is a second order PDE, and is structurally the same type of equation as the heat equation, just applied to option pricing. This project utilises a PINN to solve it directly. Instead of training on real-world price data, it trains to satisfy the PDE itself, as well as the known boundary and terminal conditions.

The project extends to an inverse problem: recovering the volatility given some synthetic option prices.

Full derivation of the PDE can be found [here](pde-derivation.md)

## Results

- **Forward problem**: A domain-wide absolute error under 0.2 when compared against the analytical solution to the Black-Scholes PDE (option values are in the approximate range 0-60)

- **Inverse problem**: Sigma recovered to within 2% of the true value (0.197 vs 0.2).

## Method

**Forward problem**

The PINN maps (S, t) to V. It is trained by minimising the total loss $L$:

$$L = w_{pde} \cdot (PDE \\ residual)^2 + w_{bound} \cdot (V - boundary \\ value)^2 + w_{term} \cdot (V - payoff)^2$$

The PDE residual is computed by calculating $\frac{dV}{dS}$, $\frac{dV}{dt}$, and $\frac{d^2V}{dS^2}$ using `torch.autograd.grad`, and then substituting these back into the equation.

**Inverse problem**

Volatility is added to the network as another learnable parameter, and a data-fitting loss against synthetic prices is added to the total loss:

$$L = w_{pde} \cdot (PDE \\ residual)^2 + w_{bound} \cdot (V - boundary \\ value)^2 + w_{term} \cdot (V - payoff)^2 + w_{data} \cdot (data \\ residual)^2$$

## Identifiability issue and resolution

Initially the forward and inverse problems were trained together, but this produced a highly incorrect estimate of volatility (0.04 vs 0.2), despite the training loss being small. This was caused by sigma only appearing in the PDE multiplied by the curvature term, meaning that the network was only able constrain the loss of this product, and not the two components individually (the product of a bad value of sigma and a bad value of curvature was still able to satisfy the PDE residual).

**Fix:**

Two stage training was implemented. Stage 1 solves the forward problem using a value of sigma that is frozen at an initial guess, allowing a physically sensible solution shape to be found. Stage 2 unfreezes sigma and adds data-fitting loss, allowing sigma to be trained with an already near-correct V already in place. This brought the value of sigma to approximately 0.196, consequently reducing the error from ~78% to ~2%.

## Limitations

- **Far boundary error**. The far boundary condition (where S = S_max) assumes that V is exactly linear in S at this point, while in the true solution some curvature remains. This is visible as an error ramp for S beyond ~135. This effect persists regardless of training time, and would likely be resolved by implementing a Neumann boundary condition.

## Setup

```bash
pip install torch numpy scipy matplotlib
```

## Run

```bash
python black_scholes_pinn.py
```

## Possible extensions

- American options
- Gradient-based adaptive loss weighting to replace the manually chosen fixed weight currently used
- Comparison against Monte-Carlo and finite-difference methods (accuracy vs compute time tradeoffs)
- Neumann boundary condition to address the aforementioned boundary approximation error







