import numpy as np
from numba import jit

@jit(nopython=True, cache=True, fastmath=True)
def generate_paths_kernel(S0: float, r: float, sigma: float, 
                          T: float, n_paths: int, n_steps: int) -> np.ndarray:
    """
    JIT-compiled Geometric Brownian Motion generator. 
    """
    dt = T / n_steps

    # Antithetic Variates for Variance Reduction
    if n_paths % 2 != 0:
        n_paths += 1
    
    half_paths = n_paths // 2

    # 1. Randomness: vectorized np.random draws are fast and compatible with Numba
    Z_half = np.random.standard_normal((half_paths, n_steps))
    Z = np.concatenate((Z_half, -Z_half), axis=0)
    
    # 2. Pre-compute constants
    drift_step = (r - 0.5 * sigma**2) * dt
    diffusion_step = sigma * np.sqrt(dt)
    
    # 3. Output Matrix
    prices = np.zeros((n_paths, n_steps + 1))
    prices[:, 0] = S0
    
    # 4. JIT Loop (replacing cumsum)
    for i in range(n_paths):
        current_log_return = 0.0
        for j in range(n_steps):
            # Update the random walk
            shock = Z[i, j]
            current_log_return += drift_step + diffusion_step * shock
            
            # Calculate price here to save memory
            prices[i, j + 1] = S0 * np.exp(current_log_return)
            
    return prices