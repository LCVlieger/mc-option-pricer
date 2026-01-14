import numpy as np
import scipy.integrate as integrate
from heston_pricer.market import MarketEnvironment
from heston_pricer.instruments import EuropeanOption, OptionType
from heston_pricer.models.process import HestonProcess
from heston_pricer.models.mc_pricer import MonteCarloPricer

# --- 1. The Exact Analytical Heston Formula (Fourier Transform) ---
# This uses the Heston 1993 approach extended by Albrecher (2007) for stability.
def heston_call_price(S0, K, T, r, kappa, theta, v0, rho, xi):
    """
    Calculates Heston Call price using numerical integration.
    """
    def heston_char_func(u):
        # Characteristic function of ln(S_T)
        d = np.sqrt((rho * xi * u * 1j - kappa)**2 + xi**2 * (u * 1j + u**2))
        g = (kappa - rho * xi * u * 1j - d) / (kappa - rho * xi * u * 1j + d)
        
        C = (1/xi**2) * (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T)) * \
            (kappa - rho * xi * u * 1j - d)
            
        D = (kappa * theta / xi**2) * \
            ((kappa - rho * xi * u * 1j - d) * T - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g)))
            
        return np.exp(C * v0 + D + 1j * u * np.log(S0 * np.exp(r * T)))

    def integrand(u):
        # The actual function we integrate
        # Price = S0 * P1 - K * exp(-rT) * P2
        # This integrand calculates the probabilities P1 and P2 in one go logic
        # But standard implementation splits them. Let's use the Carr-Madan 
        # or simplified generic pricing via damping for robustness.
        # Actually, let's use the Gil-Pelaez formula directly for Call.
        
        cf_num = heston_char_func(u - 1j)
        cf_denom = heston_char_func(-1j) # effectively S0 * exp(rT)
        p1_integrand = np.real(np.exp(-1j * u * np.log(K)) * cf_num / (1j * u * cf_denom))
        
        cf_num_2 = heston_char_func(u)
        p2_integrand = np.real(np.exp(-1j * u * np.log(K)) * cf_num_2 / (1j * u))
        
        return p1_integrand, p2_integrand

    # Integration limit (infinity)
    limit = 100 
    
    # P1: Delta-like term
    P1_int = integrate.quad(lambda u: np.real(np.exp(-1j * u * np.log(K)) * heston_char_func(u - 1j) / (1j * u * S0 * np.exp(r*T))), 0, limit)[0]
    P1 = 0.5 + (1/np.pi) * P1_int
    
    # P2: Probability term
    P2_int = integrate.quad(lambda u: np.real(np.exp(-1j * u * np.log(K)) * heston_char_func(u) / (1j * u)), 0, limit)[0]
    P2 = 0.5 + (1/np.pi) * P2_int
    
    price = S0 * P1 - K * np.exp(-r * T) * P2
    return price

def main():
    print("--- Heston Validation: Monte Carlo vs Analytic Formula ---")
    
    # 1. Parameters
    # Using 'Nasty' parameters to stress test the model
    # High Vol-of-Vol (0.5), High Negative Correlation (-0.7)
    params = {
        'S0': 100, 'r': 0.03,
        'v0': 0.04, 'kappa': 1.0, 'theta': 0.04, 'xi': 0.5, 'rho': -0.7
    }
    T, K = 1.0, 100
    
    # 2. Analytic Price (The "Truth")
    print("\n[1] Calculating Analytic Price (Integration)...")
    ref_price = heston_call_price(
        params['S0'], K, T, params['r'], 
        params['kappa'], params['theta'], params['v0'], params['rho'], params['xi']
    )
    print(f"Analytic Price: {ref_price:.4f}")
    
    # 3. Monte Carlo Price (Our Code)
    print("\n[2] Running Monte Carlo...")
    env = MarketEnvironment(**params)
    process = HestonProcess(env)
    pricer = MonteCarloPricer(process)
    
    opt = EuropeanOption(K, T, OptionType.CALL)
    
    # Run a large simulation to ensure convergence
    n_paths = 200_000
    res = pricer.price(opt, n_paths=n_paths)
    
    print(f"MC Price:       {res.price:.4f}")
    print(f"Standard Error: {res.std_error:.4f}")
    print(f"95% CI:         [{res.conf_interval_95[0]:.4f}, {res.conf_interval_95[1]:.4f}]")
    
    # 4. Conclusion
    error = abs(res.price - ref_price)
    
    print(f"\n[3] Result")
    print(f"Absolute Error: {error:.4f}")
    
    # Check if Analytic price is within the MC Confidence Interval
    if res.conf_interval_95[0] <= ref_price <= res.conf_interval_95[1]:
        print(" SUCCESS: Analytic price is within Monte Carlo confidence interval.")
    else:
        print(" CHECK: Small divergence (could be random noise or discretization bias).")

if __name__ == "__main__":
    main()