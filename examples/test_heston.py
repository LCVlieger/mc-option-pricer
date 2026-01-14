import time
import numpy as np
from heston_pricer.models.mc_kernels import generate_heston_paths

def main():
    # 1. Parameters
    S0 = 100.0
    r = 0.05
    T = 1.0
    n_paths = 50_000
    n_steps = 100
    
    # Heston Specifics
    v0 = 0.04     # 20% vol
    kappa = 2.0   # Fast mean reversion
    theta = 0.04  # Long run 20% vol
    xi = 0.3      # High Vol of Vol (Should create fat tails)
    rho = -0.7    # Negative correlation (Equity Skew)
    
    print("--- Testing Heston Kernel JIT ---")
    
    # 2. Warmup JIT
    print("Compiling...")
    _ = generate_heston_paths(S0, r, v0, kappa, theta, xi, rho, T, 10, 10)
    print("Compilation Complete.")
    
    # 3. Run Simulation
    t0 = time.time()
    paths = generate_heston_paths(S0, r, v0, kappa, theta, xi, rho, T, n_paths, n_steps)
    dt = time.time() - t0
    
    S_T = paths[:, -1]
    
    print(f"Execution Time: {dt:.4f}s")
    print(f"Mean Terminal Price: {np.mean(S_T):.4f}")
    
    # 4. Check for 'The Skew' (Prices of OTM Put vs OTM Call)
    # Strike 80 (Put) vs Strike 120 (Call). Both 20 away from spot.
    # In Black-Scholes (Symmetric), these should be somewhat similar (adjusted for drift).
    # In Heston (with rho=-0.7), the Put should be notably more expensive relative to BS probability.
    
    payoff_put = np.maximum(80 - S_T, 0)
    payoff_call = np.maximum(S_T - 120, 0)
    
    price_put = np.mean(payoff_put) * np.exp(-r*T)
    price_call = np.mean(payoff_call) * np.exp(-r*T)
    
    print("\n--- Market Skew Check ---")
    print(f"OTM Put (K=80):  {price_put:.4f}")
    print(f"OTM Call (K=120): {price_call:.4f}")
    print("If Put is 'surprisingly' high, Heston is working (Fat Left Tail).")

if __name__ == "__main__":
    main()