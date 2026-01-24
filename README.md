# HestonPricer: High-Performance Stochastic Volatility Engine

**A JIT-compiled pricing and calibration library for Exotic Derivatives, bridging the gap between mathematical theory (Shreve/Hull) and production engineering.**

## Key Capabilities

* **Real-Time Calibration**: Solves the inverse problem for Heston parameters ($\kappa, \theta, \xi, \rho, v_0$) using **L-BFGS-B** optimization against live option chains (e.g., NVDA, TSLA).
* **HPC Architecture**: Python loops are replaced with **Numba** kernels (LLVM compilation), achieving a **~16x speedup** over Pure Python and **2.5x speedup** over vectorized NumPy by eliminating memory overhead.
* **Exotic Pricing**: Supports path-dependent payoffs including **Barrier (Knock-Out/Knock-In)** and **Arithmetic Asian** options.
* **Risk Management**: Computes **Delta, Gamma, and Vega** via Finite Difference, correctly capturing "Negative Gamma" risks near barriers.
* **Mathematical Rigor**: Implements **Gil-Pelaez Fourier Inversion** for fast calibration and **Full Truncation Euler** discretization for simulation stability.

---

## Case Study: NVIDIA (NVDA) Down-and-Out Call

**Calibration Date:** Jan 24, 2026
* **Regime:** High Mean Reversion ($\kappa \approx 4.8$), Negative Skew ($\rho \approx -0.46$).
* **Barrier Risk:** The pricing engine detected **Negative Gamma** ($\Gamma \approx -0.06$) near the barrier ($150), highlighting the "Reverse Hedging" risk (selling into a falling market) inherent in barrier products.
* **Volatility Exposure:** The model correctly identified a **Positive Vega** ($+79.1$) for the structure, as the "Long Call" optionality outweighed the "Knock-Out" probability at the current spot level ($187).

---

## Performance Benchmarks

*Hardware: Standard Consumer Laptop (Python 3.12)*
*Simulation: 2,000,000 Paths, 252 Steps (Daily Monitoring)*

| Implementation | Execution Time | Speedup vs Python | Speedup vs NumPy |
| :--- | :--- | :--- | :--- |
| **Pure Python** | ~610 s (Est.) | 1.0x | - |
| **NumPy Vectorized** | 94.63 s | 6.4x | 1.0x |
| **HestonPricer (Numba)** | **38.38 s** | **~15.9x** | **2.5x** |

*Note: Numba JIT compiles the Monte Carlo kernel to machine code, eliminating memory allocation overhead for intermediate path arrays which bottlenecks NumPy at high scale.*

---

## Usage

### 1. Market Calibration (The "Strat" View)
Fetches live option chains, filters for liquidity, and performs a dual-phase calibration (Analytical Fourier → Monte Carlo refinement).
```bash
python examples/1_market_calibration.py
```
*Output: Saves `calibration_[TICKER]_[DATE]_meta.json` and reports IV-RMSE.*

### 2. Exotic Pricing & Risk (The "Structuring" View)
Loads the calibrated parameters to price a **Down-and-Out Call** and an **Arithmetic Asian Call**, including full Greeks.
```bash
python examples/2_exotic_pricing.py
```
*Output: Prices, Delta, Gamma, Vega.*

### 3. Convergence & Benchmarking (The "Quant Dev" View)
Validates the numerical stability of the engine and benchmarks Numba performance.
```bash
python examples/3_convergence_analysis.py
```
*Output: Speedup metrics and Fourier vs. MC error analysis.*

---

## 📚 Mathematical Methodology

**1. Geometric Brownian Motion (Black-Scholes)**
Standard risk-neutral discretization for an asset with risk-free rate $r$ and dividend yield $q$:
$$S_{t+\Delta t} = S_t \exp\left( (r - q - \frac{1}{2}\sigma^2)\Delta t + \sigma \sqrt{\Delta t} Z \right)$$

**2. Heston Stochastic Volatility Model**
Modeled via two correlated Stochastic Differential Equations (SDEs) to capture volatility clustering and skew (leverage effect):

$$dS_t = (r - q) S_t dt + \sqrt{v_t} S_t dW_S$$
$$dv_t = \kappa (\theta - v_t) dt + \xi \sqrt{v_t} dW_v$$
$$\text{Corr}(dW_S, dW_v) = \rho$$

* **$\rho$ (Correlation):** Controls the **Skew**. A negative $\rho$ (e.g., -0.7) means when Spot falls, Volatility spikes (Crash Risk).
* **$\xi$ (Vol of Vol):** Controls the **Smile** (Kurtosis/Fat Tails).
* **$\kappa$ (Mean Reversion):** The speed at which variance returns to the long-run average $\theta$.

**3. Exotic Payoffs**
* **Asian Option**: Payoff depends on the arithmetic mean of the path: $\max(\frac{1}{N}\sum S_{t_i} - K, 0)$.
* **Barrier Option**: Path-dependent activation. The option creates (Knock-In) or destroys (Knock-Out) value if $S_t$ breaches a barrier $B$ at any time $t$.

---

## Installation

```bash
git clone [https://github.com/LCVlieger/heston_pricer](https://github.com/LCVlieger/heston_pricer)
pip install -e .
```

## Testing

The library includes a regression test suite to ensure mathematical accuracy.

```bash
pytest tests/test_pricing.py -v
```

* **Convergence Checks**: Verifies that Monte Carlo estimates converge to the exact Black-Scholes price (European) and Turnbull-Wakeman approximation (Asian).
* **Parity checks**: Validates logical consistency, such as **Put-Call Parity**.

```