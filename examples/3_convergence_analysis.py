import time
import numpy as np
import math
from heston_pricer.models.mc_kernels import generate_heston_paths
from heston_pricer.analytics import HestonAnalyticalPricer

# --- 1. COMPETITOR: Pure Python (The Baseline) ---
def heston_pure_python(S0, r, q, v0, kappa, theta, xi, rho, T, n_paths, n_steps):
    """
    Standard 'loopy' Python implementation. 
    Demonstrates why C++/Numba is necessary.
    """
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)
    
    # Pre-compute Correlation matrix decomp
    c1 = rho
    c2 = math.sqrt(1 - rho**2)
    
    final_prices = []
    
    for i in range(n_paths):
        s_t = S0
        v_t = v0
        
        for j in range(n_steps):
            # Slow random generation
            z1 = np.random.normal()
            z2 = np.random.normal()
            zv = c1 * z1 + c2 * z2
            
            # Heston Dynamics
            v_positive = max(v_t, 0.0)
            dv = kappa * (theta - v_positive) * dt + xi * math.sqrt(v_positive) * sqrt_dt * zv
            v_t += dv
            
            vol_t = math.sqrt(v_positive)
            drift = (r - q - 0.5 * v_positive) * dt
            diffusion = vol_t * sqrt_dt * z1
            
            s_t *= math.exp(drift + diffusion)
            
        final_prices.append(s_t)
    return np.array(final_prices)

# --- 2. COMPETITOR: NumPy Vectorized (The Standard) ---
def heston_numpy_vectorized(S0, r, q, v0, kappa, theta, xi, rho, T, n_paths, n_steps):
    """
    Vectorized implementation. Fast, but allocates massive RAM for intermediate arrays.
    """
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    
    # Arrays
    prices = np.zeros((n_paths, n_steps + 1))
    prices[:, 0] = S0
    v_t = np.full(n_paths, v0)
    s_t = np.full(n_paths, S0)
    
    # Pre-generate Randoms (Memory Heavy)
    Z1 = np.random.normal(size=(n_steps, n_paths))
    Z2 = np.random.normal(size=(n_steps, n_paths))
    Zv = rho * Z1 + np.sqrt(1 - rho**2) * Z2
    
    for t in range(n_steps):
        v_pos = np.maximum(v_t, 0.0)
        sq_v = np.sqrt(v_pos)
        
        # Variance Step
        dv = kappa * (theta - v_pos) * dt + xi * sq_v * sqrt_dt * Zv[t]
        v_t += dv
        
        # Price Step
        drift = (r - q - 0.5 * v_pos) * dt
        diff = sq_v * sqrt_dt * Z1[t]
        s_t *= np.exp(drift + diff)
        
    return s_t

# --- 3. THE EXECUTION ---
def main():
    print("=== PERFORMANCE BENCHMARK: Heston Monte Carlo ===")
    
    # Parameters
    S0, r, q, T = 100.0, 0.05, 0.0, 1.0
    v0, kappa, theta, xi, rho = 0.04, 1.0, 0.04, 0.5, -0.7
    
    n_paths = 2_000_000
    n_steps = 252

    # A. Pure Python (Run small batch to avoid waiting forever)
    n_paths_py = 50_000 
    print(f"\n[1] Pure Python ({n_paths_py} paths)...")
    t0 = time.time()
    heston_pure_python(S0, r, q, v0, kappa, theta, xi, rho, T, n_paths_py, n_steps)
    t_py = time.time() - t0
    # Extrapolate to 100k paths
    t_py_equiv = t_py * (n_paths / n_paths_py)
    print(f"    Time: {t_py:.4f}s")
    print(f"    Extrapolated to {n_paths} paths: {t_py_equiv:.2f}s")

    # B. NumPy (Run full batch)
    
    print(f"\n[2] NumPy Vectorized ({n_paths} paths)...")
    t0 = time.time()
    heston_numpy_vectorized(S0, r, q, v0, kappa, theta, xi, rho, T, n_paths, n_steps)
    t_np = time.time() - t0
    print(f"    Time: {t_np:.4f}s")

    # C. Numba (Run full batch)
    print(f"\n[3] Numba JIT (Your Engine) ({n_paths} paths)...")
    # Warmup
    generate_heston_paths(S0, r, q, v0, kappa, theta, xi, rho, T, 10, n_steps)
    
    t0 = time.time()
    generate_heston_paths(S0, r, q, v0, kappa, theta, xi, rho, T, n_paths, n_steps)
    t_numba = time.time() - t0
    print(f"    Time: {t_numba:.4f}s")

    # D. Summary
    print("\n=== SPEEDUP SUMMARY ===")
    print(f"Numba vs Python: {t_py_equiv / t_numba:.1f}x FASTER")
    print(f"Numba vs NumPy:  {t_np / t_numba:.1f}x FASTER")
    print("--------------------------------------------------")

    # --- VALIDATION SECTION ---
    print("\n=== NUMERICAL CONVERGENCE CHECK ===")
    K = 100
    # Analytic Price
    price_ana = HestonAnalyticalPricer.price_european_call(S0, K, T, r, q, kappa, theta, xi, rho, v0)
    
    # MC Price (High Precision)
    paths = generate_heston_paths(S0, r, q, v0, kappa, theta, xi, rho, T, 200_000, 100)
    S_T = paths[:, -1]
    payoff = np.maximum(S_T - K, 0)
    price_mc = np.mean(payoff) * np.exp(-r*T)
    
    err = abs(price_ana - price_mc)
    print(f"Analytical Price: {price_ana:.4f}")
    print(f"Monte Carlo Price: {price_mc:.4f}")
    print(f"Error:             {err:.4f}")
    
    if err < 0.05:
        print(">> PASS: Convergence Confirmed.")
    else:
        print(">> FAIL: Check Random Number Generator.")

if __name__ == "__main__":
    main()