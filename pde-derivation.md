# Deriving the Black-Scholes PDE

The value of our option, V, is a function of stock price S and time t, meaning that $V = V(S, t)$

This allows us to apply Itô's lemma:

$$dV = \frac{\partial V}{\partial t} dt + \frac{\partial V}{\partial S} (S, t) dS + \frac{1}{2} \frac{\partial ^2 V}{\partial S ^2} (S, t) dS^2$$

The price of a stock is governed by geometric Brownian motion, the expression for which is:

$$dS(t) = \mu S(t) dt + \sigma S(t) dW(t)$$

We can then substitute this expression into Itô's lemma to obtain

$$dV = \left(\frac{\partial V}{\partial t}(S, t) +\mu S \frac{\partial V}{\partial S}(S, t) + \frac{1}{2} \sigma ^2 S^2 \frac{\partial ^2 V}{\partial S ^2} (S, t)\right)dt + \sigma S \frac{\partial V}{\partial S}(S, t)dW$$

We can then consider a portfolio consisting of one long option V and short position in $\Delta$ shares of the underlying stock S (to eliminate risk), which gives a portfolio with total value of $\Pi = V - \Delta S$. Hence the change in the portfolio value over time is given by $d\Pi = dV - \Delta dS$.

We can then substitute this portfolio into the above equation:

$$d(V - \Delta S) = \left(\frac{\partial V}{\partial t}(S, t) +\mu S \frac{\partial V}{\partial S}(S, t) + \frac{1}{2} \sigma ^2 S^2 \frac{\partial ^2 V}{\partial S ^2} (S, t) - \Delta \mu S\right)dt + \sigma S \left( \frac{\partial V}{\partial S}(S, t) - \Delta \right)dW$$

We can then eliminate the final term (the one associated with randomness) by setting $\Delta = \frac{\partial V}{\partial S} (S, t)$. This leads to the following equation:

$$d(V - \frac{\partial V}{\partial S} S) = \left(\frac{\partial V}{\partial t}(S, t) + \frac{1}{2} \sigma ^2 S^2 \frac{\partial ^2 V}{\partial S ^2} (S, t)\right)dt$$

All risk has been removed, meaning we can apply the no-arbitrage argument to deduce that the portfolio must grow at the risk free rate $r$, and hence $\frac{d}{dt}(V - \frac{\partial V}{\partial S} S) = r(V - \frac{\partial V}{\partial S} S)$. Thus we can state that

$$r(V - \frac{\partial V}{\partial S} S) = \frac{\partial V}{\partial t}(S, t) + \frac{1}{2} \sigma ^2 S^2 \frac{\partial ^2 V}{\partial S ^2} (S, t)$$

After rearranging and dropping the functional arguments, we arrive at the final Black-Scholes partial differential equation:

$$\frac{\partial V}{\partial t} + rS \frac{\partial V}{\partial S} +\frac{1}{2} \sigma ^2 S^2 \frac{\partial ^2 V}{\partial S^2} - rV = 0$$

