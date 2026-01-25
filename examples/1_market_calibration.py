import time
import json
import os
import shutil
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from typing import List, Tuple, Dict

from heston_pricer.calibration import HestonCalibrator, HestonCalibratorMC, MarketOption
from heston_pricer.analytics import HestonAnalyticalPricer

# Clear Numba cache to ensure fresh kernel compilation
for root, dirs, files in os.walk("src"):
    for d in dirs:
        if d == "__pycache__":
            shutil.rmtree(os.path.join(root, d), ignore_errors=True)

# ==========================================
# 1. ROBUST DATA FETCHING (SMART BUCKETS)
# ==========================================
def fetch_options(ticker_symbol: str, max_per_bucket: int = 6) -> Tuple[List[MarketOption], float]:
    """
    Fetches option chain and selects a balanced sample across Short, Medium, and Long maturities.
    """
    ticker = yf.Ticker(ticker_symbol)
    
    # A. Get Spot Price
    try:
        S0 = ticker.fast_info.get('last_price', None)
        if S0 is None:
            hist = ticker.history(period="1d")
            if hist.empty: raise ValueError("No price data")
            S0 = hist['Close'].iloc[-1]
    except Exception as e:
        print(f"[Error] Failed to fetch spot: {e}")
        return [], 0.0

    # B. Setup Buckets (To ensure we get the full volatility surface)
    # T > 0.10 filters out short-term gamma noise (options expiring in < 1 month)
    buckets = {
        "Short":  {'min': 0.10, 'max': 0.40, 'count': 0},  # ~1-5 months
        "Medium": {'min': 0.40, 'max': 1.00, 'count': 0},  # ~5-12 months
        "Long":   {'min': 1.00, 'max': 2.50, 'count': 0}   # ~1-2.5 years
    }
    
    # Strikes: roughly 10% OTM/ITM
    target_moneyness = [0.90, 0.95, 1.00, 1.05, 1.10]
    market_options = []
    
    print(f"[Data] Scanning chains for {ticker_symbol} (Spot: {S0:.2f})...")
    expirations = ticker.options
    if not expirations: return [], 0.0

    for exp_str in expirations:
        # Stop if we have enough total data
        if len(market_options) > 60: break

        try:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
            T = (exp_date - datetime.now()).days / 365.25
        except: continue
        
        # Check if this expiration fits into an unfilled bucket
        target_bucket = next((name for name, b in buckets.items() 
                              if b['min'] <= T <= b['max'] and b['count'] < max_per_bucket), None)
        if not target_bucket: continue

        # Fetch Chain
        try:
            calls = ticker.option_chain(exp_str).calls
        except: continue
        if calls.empty: continue

        # Filter roughly around ATM to reduce processing
        calls = calls[(calls['strike'] > S0 * 0.75) & (calls['strike'] < S0 * 1.35)]
        
        # Select specific strikes
        selected_indices = set()
        for m in target_moneyness:
            target_strike = S0 * m
            calls['dist'] = (calls['strike'] - target_strike).abs()
            if calls.empty: continue
            
            best_idx = calls['dist'].idxmin()
            # Only pick if close to target moneyness (within 2.5%)
            if calls.loc[best_idx, 'dist'] / S0 < 0.025:
                selected_indices.add(best_idx)

        # Add selected options
        for idx in selected_indices:
            if buckets[target_bucket]['count'] >= max_per_bucket: break
            
            row = calls.loc[idx]
            price = row['lastPrice'] 
            
            # Sanity Checks: Price > 0.05 and No Arbitrage (Price < Intrinsic)
            if price < 0.05: continue
            if price < (max(S0 - row['strike'], 0) - 0.5): continue 

            market_options.append(MarketOption(
                strike=float(row['strike']),
                maturity=float(T),
                market_price=float(price),
                option_type="CALL"
            ))
            buckets[target_bucket]['count'] += 1

    print(f"   -> Found: Short={buckets['Short']['count']}, Med={buckets['Medium']['count']}, Long={buckets['Long']['count']}")
    return market_options, S0

# ==========================================
# 2. SAVE RESULTS & GENERATE REPORT
# ==========================================
def save_results(ticker, S0, res_ana, res_mc, options):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"calibration_{ticker}_{timestamp}"
    
    # 1. Save Metadata (JSON)
    # Stores the calibrated parameters for later use (e.g. Exotic Pricing)
    with open(f"{base_name}_meta.json", "w") as f: 
        json.dump({
            "market": {"S0": S0, "r": 0.045, "q": 0.003}, 
            "analytical": res_ana, 
            "monte_carlo_results": res_mc # Key used by exotic_pricing.py
        }, f, indent=4)

    # 2. Save Pricing Error Table (CSV)
    print("\n" + "="*80)
    print(f"{'PRICING ERRORS (Validation)':^80}")
    print("="*80)
    print(f"{'Mat':<6} {'Strike':<8} {'Market':<8} | {'Ana':<8} {'Diff':<6} | {'Euler':<8} {'Diff':<6}")
    print("-" * 80)

    rows = []
    # Helper to unpack parameters
    def gp(r): return [r.get(k, 0) for k in ['kappa', 'theta', 'xi', 'rho', 'v0']]
    
    for i, opt in enumerate(options):
        # Reprice using Analytical Engine to validate parameter fit
        p_ana = HestonAnalyticalPricer.price_european_call(S0, opt.strike, opt.maturity, 0.045, 0.003, *gp(res_ana))
        p_mc = HestonAnalyticalPricer.price_european_call(S0, opt.strike, opt.maturity, 0.045, 0.003, *gp(res_mc))
        
        # Print sample to console
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
# 3. MAIN EXECUTION
# ==========================================
def main():
    ticker = "NVDA" 
    print(f"=== HESTON PRODUCTION CALIBRATION: {ticker} ===")
    
    options, S0 = fetch_options(ticker)
    options.sort(key=lambda x: (x.maturity, x.strike))
    
    if not options:
        print("No options found.")
        return

    print(f"Calibrating to {len(options)} instruments.")

    # Calibration Config
    r, q = 0.045, 0.003
    
    # 1. Initialize Engines
    cal_ana = HestonCalibrator(S0, r, q)
    
    # --- FIX: Removed 'scheme' arg ---
    cal_mc = HestonCalibratorMC(S0, r, q, n_paths=100000, n_steps=252) 
    
    # 2. Initial Guess [kappa, theta, xi, rho, v0]
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
        res_mc = res_ana # Fallback if MC fails completely

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
    
    # 6. Save
    save_results(ticker, S0, res_ana, res_mc, options)

if __name__ == "__main__":
    main()