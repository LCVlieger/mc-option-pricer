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

@jit(nopython=True, cache=True, fastmath=True)
def generate_heston_paths(S0, r, v0, kappa, theta, xi, rho, T, n_paths, n_steps):
    """
    Heston Model Simulation using Euler Discretization with 'Full Truncation'.
    """
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    
    # Pre-compute Correlation Matrix factors
    c1 = rho
    c2 = np.sqrt(1 - rho**2)
    
    # Output arrays
    prices = np.zeros((n_paths, n_steps + 1))
    prices[:, 0] = S0
    
    # Current state - FORCE FLOAT64 to avoid Numba typing errors
    curr_v = np.full(n_paths, v0, dtype=np.float64)
    curr_s = np.full(n_paths, S0, dtype=np.float64)
    
    for j in range(n_steps):
        # 1. Generate Independent Standard Normals
        Z1 = np.random.standard_normal(n_paths) # For Stock
        Z2 = np.random.standard_normal(n_paths) # For Variance helper
        
        # 2. Correlate the Variance Noise
        Zv = c1 * Z1 + c2 * Z2
        
        # 3. Update Variance (Full Truncation)
        v_positive = np.maximum(curr_v, 0.0)
        dv = kappa * (theta - v_positive) * dt + xi * np.sqrt(v_positive) * sqrt_dt * Zv
        curr_v = curr_v + dv
        
        # 4. Update Stock
        vol_t = np.sqrt(v_positive)
        drift = (r - 0.5 * v_positive) * dt
        diffusion = vol_t * sqrt_dt * Z1
        
        curr_s = curr_s * np.exp(drift + diffusion)
        
        prices[:, j + 1] = curr_s
        
    return prices