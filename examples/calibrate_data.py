import numpy as np
import time
from heston_pricer.calibration import HestonCalibrator
from heston_pricer.analytics import HestonAnalyticalPricer
from fetch_data import fetch_spx_options 

def main():
    print("=== REAL MARKET CALIBRATION (SPX) ===\n")

    # 1. Fetch Data
    # We use the script you just verified.
    # It returns a list of MarketOption objects and the Spot Price S0
    try:
        market_options, S0 = fetch_spx_options(min_open_interest=500)
    except Exception as e:
        print(f"Data Fetch Failed: {e}")
        return

    if not market_options:
        print("No data found! Aborting.")
        return

    # 2. Market Assumptions
    # In a real bank, these come from the IR Curve and Dividend Futures.
    # For now, we hardcode reasonable 2026 estimates.
    r = 0.045  # Risk-Free Rate (~4.5%)
    q = 0.015  # Dividend Yield (~1.5%)
    
    print(f"\n[Environment]")
    print(f"Spot (S0):   {S0:.2f}")
    print(f"Risk-Free:   {r*100:.2f}%")
    print(f"Div Yield:   {q*100:.2f}%")
    print(f"Data Points: {len(market_options)} options")

    # 3. Initialize Calibrator
    # Note: Ensure you updated HestonCalibrator to accept 'q' in step 1!
    calibrator = HestonCalibrator(S0, r, q)
    
    # Initial Guess: [kappa, theta, xi, rho, v0]
    # We start with "standard" equity parameters.
    initial_guess = [2.0, 0.04, 0.5, -0.7, 0.04]
    
    print(f"\n[Calibration] Starting optimization (L-BFGS-B)...")
    t0 = time.time()
    
    # --- THE HEAVY LIFTING ---
    result = calibrator.calibrate(market_options, init_guess=initial_guess)
    
    dt = time.time() - t0
    print(f"Optimization completed in {dt:.2f} seconds.")

    # 4. Report Results
    if result['success']:
        print("\n>>> SUCCESS: Parameters Recovered <<<")
    else:
        print("\n>>> WARNING: Optimizer did not strictly converge <<<")

    print("-" * 40)
    print(f"{'Parameter':<10} {'Value':<10} {'Interpretation'}")
    print("-" * 40)
    print(f"{'v0':<10} {result['v0']:<10.4f} Current Vol ~{np.sqrt(result['v0'])*100:.1f}%")
    print(f"{'theta':<10} {result['theta']:<10.4f} Long-Run Vol ~{np.sqrt(result['theta'])*100:.1f}%")
    print(f"{'kappa':<10} {result['kappa']:<10.4f} Mean Reversion Speed")
    print(f"{'xi':<10} {result['xi']:<10.4f} Vol of Vol (Tails)")
    print(f"{'rho':<10} {result['rho']:<10.4f} Skew Correlation")
    print("-" * 40)

    # 5. Validation (Goodness of Fit)
    print("\n[Sample Fit Check]")
    print(f"{'Expiry':<8} {'Strike':<8} {'Market':<8} {'Model':<8} {'Error'}")
    
    # Sort by maturity then strike for clean printing
    market_options.sort(key=lambda x: (x.maturity, x.strike))
    
    # Pick every 7th option to show a sample
    sample_indices = range(0, len(market_options), 7)
    
    sse = 0.0
    for i in sample_indices:
        opt = market_options[i]
        
        # Price using the calibrated parameters
        model_price = HestonAnalyticalPricer.price_european_call(
            S0, opt.strike, opt.maturity, r, q,
            result['kappa'], result['theta'], result['xi'], result['rho'], result['v0']
        )
        
        err = model_price - opt.market_price
        print(f"{opt.maturity:<8.3f} {opt.strike:<8.0f} {opt.market_price:<8.2f} {model_price:<8.2f} {err:+.2f}")

if __name__ == "__main__":
    main()