import numpy as np
import time
import json
import pandas as pd
from datetime import datetime
# Ensure these imports match your actual folder structure
from heston_pricer.calibration import HestonCalibrator, HestonCalibratorMC
from heston_pricer.analytics import HestonAnalyticalPricer
from fetch_data import fetch_options 

def save_results_to_disk(ticker, S0, r, q, res_ana, res_mc, initial_guess, options, filename_prefix="calibration"):
    """
    Saves the calibration run to a JSON file (metadata) and CSV file (pricing data).
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{filename_prefix}_{ticker}_{timestamp}"
    
    # 1. Save Metadata & Parameters (JSON)
    metadata = {
        "timestamp": timestamp,
        "ticker": ticker,
        "environment": {
            "S0": S0,
            "r": r,
            "q": q
        },
        "initial_guess": {
            "kappa": initial_guess[0],
            "theta": initial_guess[1],
            "xi": initial_guess[2],
            "rho": initial_guess[3],
            "v0": initial_guess[4]
        },
        "parameters_analytical": res_ana,
        "parameters_mc": res_mc
    }
    
    json_path = f"{base_name}_meta.json"
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"\n[Saved] Parameters saved to: {json_path}")

    # 2. Save Pricing Table (CSV)
    data_rows = []
    
    for opt in options:
        # Re-price using the saved Analytical params
        p_ana = HestonAnalyticalPricer.price_european_call(
            S0, opt.strike, opt.maturity, r, q,
            res_ana['kappa'], res_ana['theta'], res_ana['xi'], res_ana['rho'], res_ana['v0']
        )
        # Re-price using the saved Monte Carlo params
        p_mc_params = HestonAnalyticalPricer.price_european_call(
            S0, opt.strike, opt.maturity, r, q,
            res_mc['kappa'], res_mc['theta'], res_mc['xi'], res_mc['rho'], res_mc['v0']
        )
        
        data_rows.append({
            "Maturity": opt.maturity,
            "Strike": opt.strike,
            "Market_Price": opt.market_price,
            "Model_Analytical": p_ana,
            "Model_MC_Params": p_mc_params,
            "Diff_Ana": p_ana - opt.market_price,
            "Diff_MC": p_mc_params - opt.market_price
        })
        
    df = pd.DataFrame(data_rows)
    csv_path = f"{base_name}_prices.csv"
    df.to_csv(csv_path, index=False)
    print(f"[Saved] Pricing table saved to: {csv_path}")

def main():
    print("=== FULL DUAL CALIBRATION: ANALYTICAL VS MONTE CARLO ===\n")

    # 1. Fetch Data
    try:
        ticker = "TSLA"  # Switched to TSLA as requested in your snippet
        market_options, S0 = fetch_options(ticker, max_per_bucket=5) 
    except Exception as e:
        print(f"Data Fetch Failed: {e}")
        return

    if not market_options:
        print("No data found! Aborting.")
        return

    # Sort options
    market_options.sort(key=lambda x: (x.maturity, x.strike))

    # 2. Market Assumptions
    r = 0.036  # Risk-Free Rate
    q = 0.00    # Dividend Yield (TSLA is 0.0)
    
    print(f"\n[Environment]")
    print(f"Spot (S0):   {S0:.2f}")
    print(f"Risk-Free:   {r*100:.2f}%")
    print(f"Div Yield:   {q*100:.2f}%")
    print(f"Total Options to Calibrate: {len(market_options)}")

    # 3. Initialize Calibrators
    calibrator_ana = HestonCalibrator(S0, r, q)
    calibrator_mc = HestonCalibratorMC(S0, r, q, n_paths=25000, n_steps=200)
    
    # Initial Guess: [kappa, theta, xi, rho, v0]
    # NOTE: Ensure this matches your bounds logic if using TSLA (High Vol)
    initial_guess = [1.5, 0.05, 1.5, -0.4, 0.05]
    
    # ---------------------------------------------------------
    # 4. Run Analytical Calibration
    # ---------------------------------------------------------
    print(f"\n[1/2] Analytical Calibration (Fourier) Starting...")
    t0 = time.time()
    try:
        res_ana = calibrator_ana.calibrate(market_options, init_guess=initial_guess)
        print(f"Analytical finished in {time.time() - t0:.2f} seconds.")
    except Exception as e:
        print(f"Analytical Calibration Crashed: {e}")
        return

    # ---------------------------------------------------------
    # 5. Run Monte Carlo Calibration
    # ---------------------------------------------------------
    print(f"\n[2/2] Monte Carlo Calibration Starting...")
    t1 = time.time()
    try:
        res_mc = calibrator_mc.calibrate(market_options, init_guess=initial_guess)
        print(f"Monte Carlo finished in {time.time() - t1:.2f} seconds.")
    except Exception as e:
        print(f"Monte Carlo Calibration Crashed: {e}")
        res_mc = res_ana 

    # ---------------------------------------------------------
    # 6. Compare Parameters
    # ---------------------------------------------------------
    print("\n" + "="*70)
    print(f"{'PARAMETER COMPARISON':^70}")
    print("="*70)
    print(f"{'Param':<10} | {'Analytical':<15} | {'Monte Carlo':<15} | {'Diff':<10}")
    print("-" * 70)
    
    params = ['v0', 'theta', 'kappa', 'xi', 'rho']
    for p in params:
        val_ana = res_ana.get(p, 0.0)
        val_mc = res_mc.get(p, 0.0)
        print(f"{p:<10} | {val_ana:<15.4f} | {val_mc:<15.4f} | {abs(val_ana - val_mc):<10.4f}")
    print("-" * 70)

    # ---------------------------------------------------------
    # 7. Validation
    # ---------------------------------------------------------
    print("\n[Price Fit Check]")
    print(f"{'Mat':<6} {'Strike':<8} {'Mkt Price':<10} {'ANA Model':<10} {'MC Params':<10} {'Diff'}")
    print("-" * 80)
    
    step = 1 if len(market_options) < 20 else len(market_options) // 10
    for i, opt in enumerate(market_options):
        if i % step == 0:
            price_ana = HestonAnalyticalPricer.price_european_call(
                S0, opt.strike, opt.maturity, r, q,
                res_ana['kappa'], res_ana['theta'], res_ana['xi'], res_ana['rho'], res_ana['v0']
            )
            price_mc_params = HestonAnalyticalPricer.price_european_call(
                S0, opt.strike, opt.maturity, r, q,
                res_mc['kappa'], res_mc['theta'], res_mc['xi'], res_mc['rho'], res_mc['v0']
            )
            print(f"{opt.maturity:<6.2f} {opt.strike:<8.0f} {opt.market_price:<10.2f} "
                  f"{price_ana:<10.2f} {price_mc_params:<10.2f} {(price_ana - price_mc_params):+.2f}")

    # ---------------------------------------------------------
    # 8. SAVE RESULTS (With Initial Guess)
    # ---------------------------------------------------------
    save_results_to_disk(ticker, S0, r, q, res_ana, res_mc, initial_guess, market_options)

if __name__ == "__main__":
    main()