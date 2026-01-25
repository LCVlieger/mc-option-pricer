import time
import json
import os
import shutil
import pandas as pd
from datetime import datetime
from heston_pricer.calibration import HestonCalibrator, HestonCalibratorMC
from heston_pricer.analytics import HestonAnalyticalPricer
from heston_pricer.data import fetch_options  

# Clear Numba cache to ensure fresh kernel compilation
for root, dirs, files in os.walk("src"):
    for d in dirs:
        if d == "__pycache__":
            shutil.rmtree(os.path.join(root, d), ignore_errors=True)

# ==========================================
# 1. SAVE RESULTS & GENERATE REPORT
# ==========================================
def save_results(ticker, S0, res_ana, res_mc, options):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"calibration_{ticker}_{timestamp}"
    
    # 1. Save Metadata (JSON)
    with open(f"{base_name}_meta.json", "w") as f: 
        json.dump({
            "market": {"S0": S0, "r": 0.045, "q": 0.003}, 
            "analytical": res_ana, 
            "monte_carlo_results": res_mc 
        }, f, indent=4)

    # 2. Save Pricing Error Table (CSV)
    print("\n" + "="*80)
    print(f"{'PRICING ERRORS (Validation)':^80}")
    print("="*80)
    print(f"{'Mat':<6} {'Strike':<8} {'Market':<8} | {'Ana':<8} {'Diff':<6} | {'Euler':<8} {'Diff':<6}")
    print("-" * 80)

    rows = []
    def gp(r): return [r.get(k, 0) for k in ['kappa', 'theta', 'xi', 'rho', 'v0']]
    
    for i, opt in enumerate(options):
        p_ana = HestonAnalyticalPricer.price_european_call(S0, opt.strike, opt.maturity, 0.045, 0.003, *gp(res_ana))
        p_mc = HestonAnalyticalPricer.price_european_call(S0, opt.strike, opt.maturity, 0.045, 0.003, *gp(res_mc))
        
        if i % max(1, len(options)//10) == 0:
            print(f"{opt.maturity:<6.2f} {opt.strike:<8.0f} {opt.market_price:<8.2f} | "
                  f"{p_ana:<8.2f} {p_ana-opt.market_price:+.2f}   | "
                  f"{p_mc:<8.2f} {p_mc-opt.market_price:+.2f}")
        
        rows.append({
            "Maturity": opt.maturity, "Strike": opt.strike, "Market": opt.market_price, 
            "Model_Ana": p_ana, "Diff_Ana": p_ana-opt.market_price,
            "Model_Euler": p_mc, "Diff_Euler": p_mc-opt.market_price
        })

    pd.DataFrame(rows).to_csv(f"{base_name}_prices.csv", index=False)
    print("-" * 80)
    print(f"[Saved] Parameters:  {base_name}_meta.json")
    print(f"[Saved] Price Table: {base_name}_prices.csv")

# ==========================================
# 2. MAIN EXECUTION
# ==========================================
def main():
    ticker = "NVDA" 
    print(f"=== HESTON PRODUCTION CALIBRATION: {ticker} ===")
    
    # 1. Fetch Data (Refactored)
    options, S0 = fetch_options(ticker)
    options.sort(key=lambda x: (x.maturity, x.strike))
    
    if not options:
        print("No options found.")
        return

    print(f"Calibrating to {len(options)} instruments.")

    # Calibration Config
    r, q = 0.045, 0.003
    
    # 2. Initialize Engines
    cal_ana = HestonCalibrator(S0, r, q)
    cal_mc = HestonCalibratorMC(S0, r, q, n_paths=40000, n_steps=252) # Removed 'scheme' arg
    
    # Initial Guess [kappa, theta, xi, rho, v0]
    init_guess = [2.0, 0.04, 0.5, -0.7, 0.04]

    # 3. Run Analytical Calibration (Fast Benchmark)
    print("\n[1] Analytical Calibration (Fourier)...")
    t0 = time.time()
    res_ana = cal_ana.calibrate(options, init_guess)
    print(f" -> RMSE: {res_ana['rmse_iv']*100:.2f}% (Time: {time.time()-t0:.2f}s)")

    # 4. Run Monte Carlo Calibration (Production)
    print("\n[2] Monte Carlo Calibration (Euler + Feller Constraint)...")
    t1 = time.time()
    try:
        res_mc = cal_mc.calibrate(options, init_guess)
        print(f" -> RMSE: {res_mc['rmse_iv']*100:.2f}% (Time: {time.time()-t1:.2f}s)")
    except Exception as e:
        print(f" -> Failed: {e}")
        res_mc = res_ana 

    # 5. Convergence Report
    print("\n" + "="*60)
    print(f"{'PARAMETER CONVERGENCE':^60}")
    print("="*60)
    print(f"{'Param':<10} | {'Analytical':<15} | {'MC Euler':<15} | {'Diff':<10}")
    print("-" * 60)
    for p in ['kappa','theta','xi','rho','v0']:
        v1 = res_ana.get(p, 0.0)
        v2 = res_mc.get(p, 0.0)
        print(f"{p:<10} | {v1:<15.4f} | {v2:<15.4f} | {abs(v1-v2):<10.4f}")
    
    save_results(ticker, S0, res_ana, res_mc, options)

if __name__ == "__main__":
    main()