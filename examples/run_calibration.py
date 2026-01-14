import numpy as np
from heston_pricer.analytics import HestonAnalyticalPricer
from heston_pricer.calibration import HestonCalibrator, MarketOption

def main():
    # 1. Simulate the Market (The "Hidden Truth")
    S0, r = 100.0, 0.03
    true_params = {
        'kappa': 1.5, 'theta': 0.04, 'xi': 0.5, 'rho': -0.7, 'v0': 0.04
    }
    
    # Create a Vol Surface (Grid of Strikes & Maturities)
    strikes = [90, 95, 100, 105, 110]
    maturities = [0.5, 1.0, 2.0]
    
    market_data = []
    print("--- 1. Observing Market Data ---")
    for T in maturities:
        for K in strikes:
            # Generate 'Market' Price using the True parameters
            price = HestonAnalyticalPricer.price_european_call(
                S0, K, T, r, **true_params
            )
            market_data.append(MarketOption(K, T, price))
            print(f"Maturity={T:<4} Strike={K:<4} Price={price:.4f}")

    # 2. The Inverse Problem
    # We pretend we don't know the parameters. We start from a generic guess.
    print("\n--- 2. Calibrating Model ---")
    calibrator = HestonCalibrator(S0, r)
    
    # Bad initial guess (to prove the optimizer works)
    guess = [0.5, 0.02, 0.1, 0.0, 0.02] 
    print(f"Initial Guess: {guess}")
    
    calibrated = calibrator.calibrate(market_data, init_guess=guess)
    
    # 3. Validation
    print("\n--- 3. Results ---")
    print(f"{'Param':<10} {'True':<10} {'Calibrated':<10} {'Error'}")
    print("-" * 45)
    
    param_keys = ['kappa', 'theta', 'xi', 'rho', 'v0']
    for key in param_keys:
        truth = true_params[key]
        est = calibrated[key]
        print(f"{key:<10} {truth:<10.4f} {est:<10.4f} {abs(truth - est):.4f}")

    if calibrated['success']:
        print("\n>> Calibration SUCCESS: Parameters recovered.")
    else:
        print("\n>> Calibration FAILED.")

if __name__ == "__main__":
    main()