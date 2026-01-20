import time
import numpy as np
from heston_pricer.analytics import HestonAnalyticalPricer
from heston_pricer.calibration import MarketOption, HestonCalibratorMC
from heston_pricer.market import MarketEnvironment
from heston_pricer.models.process import HestonProcess
from heston_pricer.models.mc_pricer import MonteCarloPricer
from heston_pricer.instruments import EuropeanOption, OptionType
def main():
    S0, r, q  = 100.0, 0.03, 0.0
    
    # 1. Define True Parameters
    true_params_list = [1.5, 0.04, 0.5, -0.7, 0.04] 
    
    # 2. Expanded Strike Surface (to fix the identification issue)
    strikes = [80, 90, 100, 110, 120] 
    maturities = [0.5, 1.0, 2.0]      
    
    # 3. Initialize Fast Calibrator
    # n_steps=100 ensures T=0.5 lands exactly on index 25 (if Max T=2.0)
    calibrator = HestonCalibratorMC(S0, r, q, n_paths=30001, n_steps=400)
    
    option_list = []
    for T in maturities:
        for K in strikes:
            option_list.append(MarketOption(K, T, 0.0))
            
    # 4. Generate Truth (Batch Method)
    print("--- 1. Generating Twin Data (Batch/Shared Noise) ---")
    true_prices = calibrator.get_prices(true_params_list, option_list)
    
    for i, opt in enumerate(option_list):
        opt.market_price = true_prices[i]
        print(f"Maturity={opt.maturity:<4} Strike={opt.strike:<4} Price={opt.market_price:.4f}")

    # 5. Calibration
    print("\n--- 2. Calibrating ---")
    guess = [0.5, 0.02, 0.1, 0.0, 0.02] 
    calibrated = calibrator.calibrate(option_list, init_guess=guess)
    
    # 6. Results
    print("\n--- 3. Results ---")
    print(f"{'Param':<10} {'True':<10} {'Calibrated':<10} {'Error'}")
    print("-" * 45)
    
    keys = ['kappa', 'theta', 'xi', 'rho', 'v0']
    for i, key in enumerate(keys):
        truth = true_params_list[i]
        est = calibrated[key]
        print(f"{key:<10} {truth:<10.4f} {est:<10.4f} {abs(truth - est):.4f}")

if __name__ == "__main__":
    main()