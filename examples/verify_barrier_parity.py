import numpy as np
from quantlib.market import MarketEnvironment
from quantlib.instruments import EuropeanOption, BarrierOption, BarrierType, OptionType
from quantlib.models.mc_pricer import MonteCarloPricer

def main():
    # 1. Setup
    S0 = 100
    K = 100
    T = 1.0
    r = 0.05
    sigma = 0.2
    barrier_level = 90  # Lower than S0
    
    env = MarketEnvironment(S0, r, sigma)
    pricer = MonteCarloPricer(env)
    
    print(f"--- Barrier Option Parity Check ---")
    print(f"Spot: {S0}, Strike: {K}, Barrier: {barrier_level}")

    # 2. Define Instruments
    # Standard European Call
    euro_call = EuropeanOption(K, T, OptionType.CALL)
    
    # Down-and-Out Call (Dies if price hits 90)
    do_call = BarrierOption(K, T, barrier_level, BarrierType.DOWN_AND_OUT, OptionType.CALL)
    
    # Down-and-In Call (Born if price hits 90)
    di_call = BarrierOption(K, T, barrier_level, BarrierType.DOWN_AND_IN, OptionType.CALL)

    # 3. Price them (Use same random seed implicitly or high paths for convergence)
    # Note: For perfect parity in MC, we should technically use the exact same random paths,
    # but with enough paths (100k), the average should be very close.
    n_paths = 200_000
    
    print(f"Running Simulations (N={n_paths})...")
    
    p_euro = pricer.price_option(euro_call, n_paths).price
    p_out = pricer.price_option(do_call, n_paths).price
    p_in = pricer.price_option(di_call, n_paths).price
    
    # 4. Check Parity
    sum_barrier = p_out + p_in
    diff = abs(p_euro - sum_barrier)
    
    print(f"\nResults:")
    print(f"European Price:    {p_euro:.4f}")
    print(f"Down-and-Out:      {p_out:.4f}")
    print(f"Down-and-In:       {p_in:.4f}")
    print(f"Sum (In + Out):    {sum_barrier:.4f}")
    print(f"Difference:        {diff:.4f}")
    
    if diff < 0.05:
        print("\n SUCCESS: Parity holds!")
    else:
        print("\n FAILURE: Difference too large.")

if __name__ == "__main__":
    main()