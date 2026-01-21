import time
import numpy as np
import math
from heston_pricer.models.mc_kernels import generate_paths_kernel as numba_kernel

# --- 1. Python Implementation (Baseline) ---
def pure_python_mc(S0, r, sigma, T, n_paths, n_steps):
    """
    Standard Python loops without NumPy vectorization.
    """
    dt = T / n_steps
    drift = (r - 0.5 * sigma**2) * dt
    vol = sigma * math.sqrt(dt)
    
    final_prices = []
    
    for i in range(n_paths):
        s_t = S0
        for j in range(n_steps):
            # Slow random generation per step
            z = np.random.normal()
            s_t *= math.exp(drift + vol * z)
        final_prices.append(s_t)
        
    return final_prices

# --- 2. Vectorized NumPy (The Competitor) ---
def numpy_vectorized_mc(S0, r, sigma, T, n_paths, n_steps):
    dt = T / n_steps
    prices = np.zeros((n_paths, n_steps + 1))
    prices[:, 0] = S0
    
    # Pre-generate randoms (Fast)
    Z = np.random.standard_normal((n_paths, n_steps))
    
    # Vectorized Math (Fast)
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    log_returns = np.cumsum(drift + diffusion * Z, axis=1)
    
    prices[:, 1:] = S0 * np.exp(log_returns)
    return prices

def main():
    S0, r, sigma, T = 100, 0.05, 0.2, 1.0
    q = 0.0
    n_paths = 50_000  # Lower paths to keep Pure Python from freezing
    n_steps = 100

    print(f"--- BENCHMARK: {n_paths} paths, {n_steps} steps ---")

    # 1. Pure Python
    print("\n1. Pure Python (Baseline)...")
    t0 = time.time()
    pure_python_mc(S0, r, sigma, T, n_paths, n_steps)
    t_base = time.time() - t0
    print(f"   Time: {t_base:.4f}s")

    # 2. Vectorized NumPy
    print("\n2. Vectorized NumPy...")
    t0 = time.time()
    numpy_vectorized_mc(S0, r, sigma, T, n_paths, n_steps)
    t_numpy = time.time() - t0
    print(f"   Time: {t_numpy:.4f}s (Speedup vs Base: {t_base/t_numpy:.1f}x)")

    # 3. Numba
    print("\n3. Numba (Optimized)...")
    # Warmup
    numba_kernel(S0, r, q, sigma, T, 10, n_steps)
    
    t0 = time.time()
    numba_kernel(S0, r, q, sigma, T, n_paths, n_steps)
    t_numba = time.time() - t0
    print(f"   Time: {t_numba:.4f}s (Speedup vs Base: {t_base/t_numba:.1f}x)")
    
    print(f"\n>>> Numba vs NumPy Speedup: {t_numpy/t_numba:.2f}x <<<")

if __name__ == "__main__":
    main()