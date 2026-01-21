import numpy as np
from numba import jit

# 1. Black-Scholes Kernel
@jit(nopython=True, cache=True, fastmath=True)
def generate_paths_kernel(S0: float, r: float, q: float, sigma: float, 
                          T: float, n_paths: int, n_steps: int) -> np.ndarray:
    dt = T / n_steps

    # ... (antithetic logic stays the same) ...
    half_paths = n_paths // 2
    Z_half = np.random.standard_normal((half_paths, n_steps))
    Z = np.concatenate((Z_half, -Z_half), axis=0)
    
    # UPDATE DRIFT HERE
    drift_step = (r - q - 0.5 * sigma**2) * dt  
    diffusion_step = sigma * np.sqrt(dt)
    
    # ... (loop stays the same) ...
    prices = np.zeros((n_paths, n_steps + 1))
    prices[:, 0] = S0
    for i in range(n_paths):
        current_log_return = 0.0
        for j in range(n_steps):
            shock = Z[i, j]
            current_log_return += drift_step + diffusion_step * shock
            prices[i, j + 1] = S0 * np.exp(current_log_return)
            
    return prices

# 2. Heston Kernel
@jit(nopython=True, cache=True, fastmath=True)
def generate_heston_paths(S0, r, q, v0, kappa, theta, xi, rho, T, n_paths, n_steps):
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    
    c1 = rho
    c2 = np.sqrt(1 - rho**2)
    
    prices = np.zeros((n_paths, n_steps + 1))
    prices[:, 0] = S0
    
    curr_v = np.full(n_paths, v0, dtype=np.float64)
    curr_s = np.full(n_paths, S0, dtype=np.float64)
    
    for j in range(n_steps):
        Z1 = np.random.standard_normal(n_paths)
        Z2 = np.random.standard_normal(n_paths)
        Zv = c1 * Z1 + c2 * Z2
        
        v_positive = np.maximum(curr_v, 0.0)
        dv = kappa * (theta - v_positive) * dt + xi * np.sqrt(v_positive) * sqrt_dt * Zv
        curr_v = curr_v + dv
        
        # UPDATE DRIFT HERE
        vol_t = np.sqrt(v_positive)
        drift = (r - q - 0.5 * v_positive) * dt 
        diffusion = vol_t * sqrt_dt * Z1
        
        curr_s = curr_s * np.exp(drift + diffusion)
        prices[:, j + 1] = curr_s
        
    return prices

# 3. Heston CRN Kernel (for Calibration)
@jit(nopython=True, cache=True, fastmath=True)
def generate_heston_paths_crn(S0, r, q, v0, kappa, theta, xi, rho, T, n_paths, n_steps, noise_matrix):
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    c1 = rho
    c2 = np.sqrt(1 - rho**2)
    
    prices = np.zeros((n_paths, n_steps + 1))
    prices[:, 0] = S0
    
    curr_v = np.full(n_paths, v0, dtype=np.float64)
    curr_s = np.full(n_paths, S0, dtype=np.float64)
    
    for j in range(n_steps):
        Z1 = noise_matrix[0, j]
        Z2 = noise_matrix[1, j]
        Zv = c1 * Z1 + c2 * Z2
        
        v_positive = np.maximum(curr_v, 0.0)
        dv = kappa * (theta - v_positive) * dt + xi * np.sqrt(v_positive) * sqrt_dt * Zv
        curr_v = curr_v + dv
        
        # UPDATE DRIFT HERE
        vol_t = np.sqrt(v_positive)
        drift = (r - q - 0.5 * v_positive) * dt 
        diffusion = vol_t * sqrt_dt * Z1
        
        curr_s = curr_s * np.exp(drift + diffusion)
        prices[:, j + 1] = curr_s
        
    return prices