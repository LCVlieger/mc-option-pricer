import numpy as np
from numba import jit

# --- KERNELS ---

@jit(nopython=True, cache=True, fastmath=True)
def generate_paths_kernel(S0: float, r: float, q: float, sigma: float, 
                          T: float, n_paths: int, n_steps: int) -> np.ndarray:
    """
    Classical GBM Black-Scholes kernel. Includes Antithetic Sampling.
    """
    dt = T / n_steps
    half_paths = n_paths // 2
    
    # Variance reduction: Antithetic variates [Z, -Z]
    Z = np.concatenate((np.random.standard_normal((half_paths, n_steps)), 
                        -np.random.standard_normal((half_paths, n_steps))), axis=0)
    
    drift = (r - q - 0.5 * sigma**2) * dt
    diff = sigma * np.sqrt(dt)
    
    prices = np.zeros((n_paths, n_steps + 1))
    prices[:, 0] = S0
    
    for i in range(n_paths):
        log_ret = 0.0
        for j in range(n_steps):
            log_ret += drift + diff * Z[i, j]
            prices[i, j + 1] = S0 * np.exp(log_ret)
            
    return prices

@jit(nopython=True, cache=True, fastmath=True)
def generate_heston_paths(S0, r, q, v0, kappa, theta, xi, rho, T, n_paths, n_steps):
    """
    Heston Model simulation using a truncated Euler discretization (v_i+ = max(0,v_i)) ('Full truncation').
    Reference: 'The Volatility Surface: A Practitioners Guide, Jim Gatheral, CH2 p.21. "
    """
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    c1, c2 = rho, np.sqrt(1 - rho**2)
    
    prices = np.zeros((n_paths, n_steps + 1))
    prices[:, 0] = S0
    
    curr_v = np.full(n_paths, v0)
    curr_s = np.full(n_paths, S0)
    
    for j in range(n_steps):
        Z1 = np.random.standard_normal(n_paths)
        Z2 = np.random.standard_normal(n_paths)
        Zv = c1 * Z1 + c2 * Z2
        

        v_pos = np.maximum(curr_v, 0.0)
        curr_v += kappa * (theta - v_pos) * dt + xi * np.sqrt(v_pos) * sqrt_dt * Zv
        
        v_pos = np.maximum(curr_v, 0.0)
        curr_s *= np.exp((r - q - 0.5 * v_pos) * dt + np.sqrt(v_pos) * sqrt_dt * Z1)
        prices[:, j + 1] = curr_s
        
    return prices

@jit(nopython=True, cache=True, fastmath=True)
def generate_heston_paths_crn(S0, r, q, v0, kappa, theta, xi, rho, T, n_paths, n_steps, noise_matrix):
    """
    Kernel for calibration using 'Common Random Numbers' (CRN). Pre-generates noise matrix. 
    This is needed for stability during the optimization. 
    """
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    c1, c2 = rho, np.sqrt(1 - rho**2)
    
    prices = np.zeros((n_paths, n_steps + 1))
    prices[:, 0] = S0
    curr_v = np.full(n_paths, v0)
    curr_s = np.full(n_paths, S0)
    
    for j in range(n_steps):
        Z1 = noise_matrix[0, j]
        Z2 = noise_matrix[1, j]
        Zv = c1 * Z1 + c2 * Z2
        
        v_pos = np.maximum(curr_v, 0.0)
        curr_v += kappa * (theta - v_pos) * dt + xi * np.sqrt(v_pos) * sqrt_dt * Zv
        
        v_pos = np.maximum(curr_v, 0.0)
        curr_s *= np.exp((r - q - 0.5 * v_pos) * dt + np.sqrt(v_pos) * sqrt_dt * Z1)
        prices[:, j + 1] = curr_s
        
    return prices