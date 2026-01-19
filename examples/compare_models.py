from heston_pricer.market import MarketEnvironment
from heston_pricer.instruments import EuropeanOption, OptionType
from heston_pricer.models.process import BlackScholesProcess, HestonProcess
from heston_pricer.models.mc_pricer import MonteCarloPricer

def main():
    # 1. Define Market
    # Case: High Vol of Vol (Heston) vs Constant Vol (BS)
    env = MarketEnvironment(
        S0=100, r=0.05, 
        # BS Param
        sigma=0.2, 
        # Heston Params
        v0=0.04, kappa=1.0, theta=0.04, xi=0.5, rho=-0.7
    )
    
    # 2. Define Option (OTM Put - Sensitive to Skew)
    # A crash protection option.
    put_option = EuropeanOption(K=80, T=1.0, option_type=OptionType.PUT)
    
    # 3. Build Two Processes
    bs_process = BlackScholesProcess(env)
    heston_process = HestonProcess(env)
    
    # 4. Run Simulations
    n_paths = 100_000
    
    print("--- Model Comparison: OTM Put (K=80) ---")
    
    # Run BS
    bs_pricer = MonteCarloPricer(bs_process)
    bs_res = bs_pricer.price(put_option, n_paths=n_paths)
    print(f"Black-Scholes Price: {bs_res.price:.4f}")
    
    # Run Heston
    heston_pricer = MonteCarloPricer(heston_process)
    heston_res = heston_pricer.price(put_option, n_paths=n_paths)
    print(f"Heston Price:        {heston_res.price:.4f}")
    
    # Analysis
    diff = heston_res.price - bs_res.price
    pct_diff = (diff / bs_res.price) * 100
    print(f"\nDifference: {diff:.4f} (+{pct_diff:.1f}%)")
    print("Interpretation: The Heston model prices 'Crash Risk' (Left Tail) higher due to skew.")

if __name__ == "__main__":
    main()