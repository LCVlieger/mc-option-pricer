import time
import numpy as np
from heston_pricer.analytics import HestonAnalyticalPricer
from heston_pricer.calibration import MarketOption, HestonCalibratorMC
from heston_pricer.market import MarketEnvironment
from heston_pricer.models.process import HestonProcess
from heston_pricer.models.mc_pricer import MonteCarloPricer
from heston_pricer.instruments import EuropeanOption, OptionType
def main():
    S0, r, q = 100.0, 0.03, 0.0
    
    # 1. The Hidden Truth (Theoretical Parameters)
    true_params = {
        'kappa': 1.5, 'theta': 0.04, 'xi': 0.5, 'rho': -0.7, 'v0': 0.04
    }
    
    # 2. Generate Market Data using ANALYTICAL FORMULA
    # This is the "Real World" test.
    strikes = [80, 90, 100, 110, 120]
    maturities = [0.5, 1.0, 2.0]
    
    print("--- 1. Generating Analytical Market Data (Fourier Transform) ---")
    market_options = []
    
    for T in maturities:
        for K in strikes:
            # The "Truth" comes from the Fourier Transform
            price = HestonAnalyticalPricer.price_european_call(
                S0, K, T, r, q, **true_params
            )
            market_options.append(MarketOption(K, T, price))
            print(f"Maturity={T:<4} Strike={K:<4} Price={price:.4f}")

    # 3. Calibrate MC Engine to match Analytical Prices
    print("\n--- 2. Calibrating MC Engine ---")
    
    # Note: n_steps=100 is decent. 
    # If the error is high, increase n_steps to 200 to reduce Euler bias.
    calibrator = HestonCalibratorMC(S0, r, q, n_paths=10000, n_steps=100)
    
    guess = [0.5, 0.02, 0.1, 0.0, 0.02]
    print(f"Initial Guess: {guess}")
    
    t0 = time.time()
    calibrated = calibrator.calibrate(market_options, init_guess=guess)
    dt = time.time() - t0
    
    # 4. Results
    print(f"\n--- 3. Calibration Results ({dt:.2f}s) ---")
    print(f"{'Param':<10} {'True (Analytic)':<18} {'Calibrated (MC)':<18} {'Diff'}")
    print("-" * 60)
    
    keys = ['kappa', 'theta', 'xi', 'rho', 'v0']
    for key in keys:
        truth = true_params[key]
        est = calibrated[key]
        print(f"{key:<10} {truth:<18.4f} {est:<18.4f} {abs(truth - est):.4f}")

    print("-" * 60)
    print(f"Final SSE: {calibrated['sse']:.6f}")
    
    if calibrated['sse'] < 0.5:
        print("\n>> SUCCESS: MC Engine successfully calibrated to Analytical Prices.")
        print(">> Note: Small differences are expected due to Euler Discretization Bias.")
        print(">> These 'Calibrated' parameters are the EFFECTIVE parameters for your MC engine.")
    else:
        print("\n>> WARNING: Fit is poor. Try increasing n_steps or n_paths.")

if __name__ == "__main__":
    main()