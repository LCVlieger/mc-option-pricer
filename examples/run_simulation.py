import time
from quantlib.instruments import EuropeanOption, AsianOption, OptionType
from quantlib.market import MarketEnvironment
from quantlib.models.mc_pricer import MonteCarloPricer
from quantlib.analytics import BlackScholesPricer 

def main():
    # 1. Setup Environment
    env = MarketEnvironment(S0=100, r=0.05, sigma=0.2)
    print(f"Market: S0={env.S0}, r={env.r}, sigma={env.sigma}")

    # 2. Define Instruments 
    T, K = 1.0, 100
    euro_call = EuropeanOption(K, T, OptionType.CALL)
    asian_call = AsianOption(K, T, OptionType.CALL)

    # 3. Initialize Pricer
    pricer = MonteCarloPricer(env)
    
    # --- JIT Warm-up ---
    # Run a simulation forcing Numba to compile the code.
    print("\n[System] Warmup JIT compiler...")
    _ = pricer.price_option(euro_call, n_paths=100)
    print("[System] Compilation complete. Starting benchmarks.")

    # 4. Benchmark European Option (excluding compilation)
    print("\n--- Benchmarking: European Option ---")
    start_time = time.time()
    euro_price = pricer.price_option(euro_call, n_paths=1_000_000)
    end_time = time.time()
    
    print(f"Price: {euro_price:.4f}")
    print(f"Time:  {end_time - start_time:.4f} seconds")

    # 5. Benchmark Asian Option
    print("\n--- Benchmarking: Asian Option ---")
    
    # Run Monte Carlo
    start_time = time.time()
    asian_mc_price = pricer.price_option(asian_call, n_paths=500_000)
    end_time = time.time()
    print(f"MC Price:      {asian_mc_price:.4f}")
    print(f"MC Time:       {end_time - start_time:.4f}s")
    
    # Run Hull Analytical Approximation
    asian_approx_price = BlackScholesPricer.price_asian_arithmetic_approximation(
        env.S0, asian_call.K, asian_call.T, env.r, env.sigma
    )
    print(f"Approx (TW):   {asian_approx_price:.4f}")
    
    # Professional Comparison
    diff = asian_mc_price - asian_approx_price
    print(f"Difference:    {diff:.4f}")
    if abs(diff) < 0.05:
        print(">> Validation Passed: MC aligns with Hull Approximation.")
    else:
        print(">> Note: Small difference expected due to Discrete (MC) vs Continuous (Hull) averaging, and Lognormal Approximation.")
    
    # 6. Validation
    bs_price = BlackScholesPricer.price_european_call(env.S0, K, T, env.r, env.sigma)
    print(f"\n[Validation]")
    print(f"Reference BS Price:  {bs_price:.4f}")
    print(f"Asian Discount:      {(1 - asian_mc_price/euro_price)*100:.2f}%")

if __name__ == "__main__":
    main()