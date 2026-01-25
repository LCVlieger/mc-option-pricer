import time
import json
import os
import shutil
import pandas as pd
from datetime import datetime
from heston_pricer.calibration import HestonCalibrator, HestonCalibratorMC
from heston_pricer.analytics import HestonAnalyticalPricer
from heston_pricer.data import fetch_options

""" Calibrate the Heston model to market prices of real-time market options with different maturities and strike prices. 
Compares Monte Carlo pricing and semi-analytical pricing implementations. For the semi-analytical pricing, see: 
'The Volatility Surface: A Practitioners Guide, Jim Gatheral, CH2 p.16-18'.   """

# save results after calibrating. 
def save_results(ticker, S0, res_ana, res_mc, options):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"calibration_{ticker}_{timestamp}"
    
    with open(f"{base_name}_meta.json", "w") as f: 
        json.dump({
            "market": {"S0": S0, "r": 0.045, "q": 0.003}, 
            "analytical": res_ana, 
            "monte_carlo_results": res_mc 
        }, f, indent=4)

    # Save results as dataframe & print. 
    get_params = lambda r: [r.get(k, 0) for k in ['kappa', 'theta', 'xi', 'rho', 'v0']]
    rows = []
    for opt in options:
        p_ana = HestonAnalyticalPricer.price_european_call(S0, opt.strike, opt.maturity, 0.045, 0.003, *get_params(res_ana))
        p_mc = HestonAnalyticalPricer.price_european_call(S0, opt.strike, opt.maturity, 0.045, 0.003, *get_params(res_mc))
        rows.append({
            "T": opt.maturity, "K": opt.strike, "Mkt": opt.market_price, 
            "Ana": round(p_ana, 2), "Err_A": round(p_ana - opt.market_price, 2),
            "MC": round(p_mc, 2), "Err_MC": round(p_mc - opt.market_price, 2)
        })

    df = pd.DataFrame(rows)
    df.to_csv(f"{base_name}_prices.csv", index=False)
    
    print(df.to_string(index=False))
    log(f"Artifacts: {base_name}_meta.json")


# calibrate  to market prices.  
def main():
    clear_numba_cache()
    ticker = "NVDA" 
    
    options, S0 = fetch_options(ticker)
    if not options:
        log(f"No liquidity for {ticker}")
        return

    options.sort(key=lambda x: (x.maturity, x.strike))
    log(f"Target: {ticker} (S0={S0:.2f}) | N={len(options)}")

    r, q = 0.045, 0.003
    cal_ana = HestonCalibrator(S0, r, q)
    cal_mc = HestonCalibratorMC(S0, r, q, n_paths=100_000, n_steps=252)
    init_guess = [2.0, 0.04, 0.5, -0.7, 0.04]

    # Analytical calibration
    t0 = time.time()
    res_ana = cal_ana.calibrate(options, init_guess)
    log(f"Analytical: RMSE={res_ana['rmse_iv']:.2%} ({time.time()-t0:.2f}s)")

    # Monte Carlo calibration (full truncation + feller condition soft constraints)
    t1 = time.time()
    try:
        res_mc = cal_mc.calibrate(options, init_guess)
        log(f"MonteCarlo: RMSE={res_mc['rmse_iv']:.2%} ({time.time()-t1:.2f}s)")
    except Exception as e:
        log(f"MC Fail: {e}")
        res_mc = res_ana 

    # Parameter difference table
    params = ['kappa', 'theta', 'xi', 'rho', 'v0']
    df_params = pd.DataFrame({
        'Ana': [res_ana.get(p, 0.0) for p in params],
        'MC': [res_mc.get(p, 0.0) for p in params]
    }, index=params)
    df_params['Diff'] = (df_params['Ana'] - df_params['MC']).abs()
    
    print(df_params.to_string(float_format="{:.4f}".format))
    save_results(ticker, S0, res_ana, res_mc, options)



#logging and cache clearing 
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def clear_numba_cache():
    for root, dirs, files in os.walk("src"):
        for d in dirs:
            if d == "__pycache__":
                shutil.rmtree(os.path.join(root, d), ignore_errors=True)

if __name__ == "__main__":
    main()