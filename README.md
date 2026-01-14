# HestonPricer: High-Performance Stochastic Volatility Engine

**A JIT-compiled pricing and calibration library for Exotic Derivatives, bridging the gap between mathematical theory (Shreve/Hull) and production engineering.**

##  Key Capabilities

* **Real-Time Calibration**: Solves the inverse problem for Heston parameters ($\kappa, \theta, \xi, \rho, v_0$) using **L-BFGS-B** optimization against live **S&P 500** option chains.
* **HPC Architecture**: Python loops are replaced with **Numba** kernels (LLVM compilation), achieving a **~28x speedup** over vectorized NumPy for Monte Carlo simulation.
* **Exotic Pricing**: Supports path-dependent payoffs including **Barrier (Knock-Out/Knock-In)** and **Arithmetic Asian** options.
* **Mathematical Rigor**: Implements **Gil-Pelaez Fourier Inversion** for fast calibration and **Full Truncation Euler** discretization for simulation stability.

---

##  Mathematical Methodology

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

##  Performance Benchmarks

*Hardware: Standard Cloud Instance (Python 3.10)*

| Implementation | Paths | Execution Time | Speedup |
| :--- | :--- | :--- | :--- |
| **Pure Python** | 50k | 2.12 s | 1.0x |
| **NumPy Vectorized** | 50k | 0.09 s | ~23x |
| **HestonPricer (Numba)** | 1M | **2.08 s** | **~28x (vs Vectorized)** |

*Note: Numba JIT compiles the Monte Carlo kernel to machine code, bypassing the Python Global Interpreter Lock (GIL) overhead for the inner loops.*

---

##  Installation & Usage

```bash
git clone [https://github.com/LCVlieger/heston_pricer](https://github.com/LCVlieger/heston_pricer)
pip install -e .
```

### Example: Pricing a Heston Barrier Option

```python
from heston_pricer.market import MarketEnvironment
from heston_pricer.instruments import BarrierOption, BarrierType, OptionType
from heston_pricer.models.process import HestonProcess
from heston_pricer.models.mc_pricer import MonteCarloPricer

# 1. Configure Market (Calibrated to S&P 500)
# r=4.5%, q=1.5% (Dividend Yield)
env = MarketEnvironment(
    S0=6896.00, r=0.045, q=0.015,
    v0=0.04, kappa=0.1, theta=0.04, xi=2.0, rho=-0.78
)

# 2. Initialize Engine
process = HestonProcess(env)
pricer = MonteCarloPricer(process)

# 3. Define Instrument (Down-and-Out Call)
# Strike=6900, Barrier=6000 (Knock-Out)
barrier_opt = BarrierOption(
    K=6900, T=0.5, 
    barrier=6000.0, 
    barrier_type=BarrierType.DOWN_AND_OUT, 
    option_type=OptionType.CALL
)

# 4. Price (High-Performance Monte Carlo)
# Returns price, standard error, and 95% confidence interval
res = pricer.price(barrier_opt, n_paths=200_000)

print(f"Exotic Price: {res.price:.4f} +/- {1.96 * res.std_error:.4f}")
```

## Testing & Validation

The library includes a regression test suite to ensure mathematical accuracy.

```bash
pytest tests/test_pricing.py -v
```

* **Heston Validation:**: Monte Carlo results are cross-validated against the Semi-Analytical Solution (using Fourier integration logic similar to Heston '93)
* **Convergence Checks**: Verifies that Monte Carlo estimates converge to the exact Black-Scholes price (European) and Turnbull-Wakeman approximation (Asian) within statistical tolerance.
* **Parity checks**: Validates logical consistency, such as **Put-Call Parity** ($C - P = S - Ke^{-rT}$)