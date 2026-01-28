import time
import json
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
from scipy.ndimage import gaussian_filter

# Local package imports
try:
    from heston_pricer.calibration import HestonCalibrator, HestonCalibratorMC, implied_volatility
    from heston_pricer.analytics import HestonAnalyticalPricer
    from heston_pricer.data import fetch_options
    from heston_pricer.models.mc_pricer import MonteCarloPricer
    from heston_pricer.models.process import HestonProcess
    from heston_pricer.market import MarketEnvironment
    from heston_pricer.instruments import EuropeanOption, OptionType
except ImportError:
    raise ImportError("heston_pricer package not found. Ensure PYTHONPATH is set correctly.")

""" 
1_market_calibration.py
-----------------------
Calibrates Heston parameters to live market data (NVDA/SPX) and generates 
a publication-grade volatility surface visualization.
"""

def save_results(ticker, S0, r, q, res_ana, res_mc, options):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"calibration_{ticker}_{timestamp}"
    
    # Save Metadata
    with open(f"{base_name}_meta.json", "w") as f: 
        json.dump({
            "market": {"S0": S0, "r": r, "q": q}, 
            "analytical": res_ana, 
            "monte_carlo_results": res_mc 
        }, f, indent=4)

    # --- 1. VALIDATION TABLE ---
    env_mc = MarketEnvironment(
        S0=S0, r=r, q=q,
        kappa=res_mc['kappa'], theta=res_mc['theta'], 
        xi=res_mc['xi'], rho=res_mc['rho'], v0=res_mc['v0']
    )
    process_mc = HestonProcess(env_mc)
    pricer_mc = MonteCarloPricer(process_mc)

    get_params_ana = lambda res: [res.get(k, 0) for k in ['kappa', 'theta', 'xi', 'rho', 'v0']]
    rows = []
    
    print(f"\n[Validation] Re-pricing {len(options)} instruments with Monte Carlo engine...")

    for opt in options:
        p_ana = HestonAnalyticalPricer.price_european_call(S0, opt.strike, opt.maturity, r, q, *get_params_ana(res_ana))
        steps = int(max(20, opt.maturity * 252)) 
        instrument = EuropeanOption(opt.strike, opt.maturity, OptionType.CALL)
        mc_result = pricer_mc.price(instrument, n_paths=100_000, n_steps=steps)
        p_mc = mc_result.price
        iv_mkt = implied_volatility(opt.market_price, S0, opt.strike, opt.maturity, r, q)
        iv_mc = implied_volatility(p_mc, S0, opt.strike, opt.maturity, r, q)

        rows.append({
            "T": opt.maturity, "K": opt.strike, "Mkt": opt.market_price, 
            "Ana": round(p_ana, 2), "Err_A": round(p_ana - opt.market_price, 2),
            "MC": round(p_mc, 2), "Err_MC": round(p_mc - opt.market_price, 2),
            "IV_Mkt": iv_mkt, "IV_MC": iv_mc
        })

    df = pd.DataFrame(rows)
    print(df[["T", "K", "Mkt", "Ana", "Err_A", "MC", "Err_MC"]].to_string(index=False))
    df.to_csv(f"{base_name}_prices.csv", index=False) 
    
    # --- 2. VISUALIZATION (Updated with Smoothing & Clipping) ---
    plot_surface(S0, r, q, res_mc, ticker, f"results/{base_name}", market_options=options)
    
    log(f"Artifacts saved to results/{base_name}*")

def plot_surface(S0, r, q, params, ticker, filename, market_options=None):
    """
    Generates a high-fidelity wireframe surface with:
    1. Strict Domain Clipping (T=[0.1, 2.5], M=[0.5, 1.8])
    2. Gaussian Smoothing (Sigma=0.6) to remove numerical noise
    3. Outlier filtering for market data
    """
    kappa, theta, xi, rho, v0 = params['kappa'], params['theta'], params['xi'], params['rho'], params['v0']

    # --- A. CONFIGURATION ---
    LOWER_M, UPPER_M = 0.5, 1.8 
    LOWER_T, UPPER_T = 0.1, 2.5
    GRID_DENSITY = 100 

    M_range = np.linspace(LOWER_M, UPPER_M, GRID_DENSITY)
    T_range = np.linspace(LOWER_T, UPPER_T, GRID_DENSITY) # Note: T is Y-axis
    X, Y = np.meshgrid(M_range, T_range)
    Z = np.zeros_like(X)

    # --- B. CALCULATION ---
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            T_val, M_val = Y[i, j], X[i, j]
            price = HestonAnalyticalPricer.price_european_call(
                S0, S0 * M_val, T_val, r, q, kappa, theta, xi, rho, v0
            )
            try:
                iv = implied_volatility(price, S0, S0 * M_val, T_val, r, q)
                Z[i, j] = iv if 0.01 < iv < 2.5 else np.nan
            except:
                Z[i, j] = np.nan

    # --- C. SMOOTHING ---
    # Fill NaNs to allow smooth filtering
    mask = np.isnan(Z)
    if np.any(mask):
        Z = pd.DataFrame(Z).interpolate(method='linear', axis=1).ffill(axis=1).bfill(axis=1).values
    
    # Apply Gaussian Filter to remove "shark teeth"
    Z_smooth = gaussian_filter(Z, sigma=0.6)

    # --- D. VISUALIZATION ---
    with plt.style.context('dark_background'):
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 1. Surface
        surf = ax.plot_surface(X, Y, Z_smooth, cmap=cm.RdYlBu_r, 
                               rcount=100, ccount=100,  
                               edgecolor='black', linewidth=0.085, alpha=0.8,                       
                               shade=False, antialiased=True, zorder=1)

        # 2. Market Data (Needles)
        if market_options:
            # STRICT FILTER: Only plot points within the visual box
            plot_opts = [
                o for o in market_options 
                if (LOWER_M <= (o.strike/S0) <= UPPER_M) and (LOWER_T <= o.maturity <= UPPER_T)
            ]
            
            valid_needles = 0
            for opt in plot_opts:
                m_mkt, t_mkt = opt.strike / S0, opt.maturity
                try:
                    iv_mkt = implied_volatility(opt.market_price, S0, opt.strike, opt.maturity, r, q)
                    if iv_mkt < 0.01 or iv_mkt > 2.5: continue
                except: continue

                # Match against SMOOTHED model surface
                m_idx = (np.abs(M_range - m_mkt)).argmin()
                t_idx = (np.abs(T_range - t_mkt)).argmin()
                iv_mod = Z_smooth[t_idx, m_idx]

                if np.isnan(iv_mod): continue

                # Outlier Check (>1.8% deviation)
                if abs(iv_mkt - iv_mod) > 0.018: continue
                
                valid_needles += 1
                is_above = iv_mkt >= iv_mod
                dot_zorder = 10 if is_above else 1
                
                # Needle Line
                ax.plot([m_mkt, m_mkt], [t_mkt, t_mkt], [iv_mod, iv_mkt], 
                        color='white', linestyle='-', linewidth=1.1, alpha=0.6, zorder=dot_zorder)
                
                # Pearl Dot
                lbl = 'Market Price-IV' if valid_needles == 1 else ""
                ax.plot([m_mkt], [t_mkt], [iv_mkt], 
                        marker='o', linestyle='None', color="#F0F0F0",         
                        markersize=3.875, alpha=0.905, zorder=dot_zorder, label=lbl)

        # 3. Aesthetics
        ax.dist = 11
        ax.set_xlim(LOWER_M, UPPER_M)
        ax.set_ylim(UPPER_T, LOWER_T) # Inverted T axis
        
        ax.xaxis.set_pane_color((1, 1, 1, 0))
        ax.yaxis.set_pane_color((1, 1, 1, 0))
        ax.zaxis.set_pane_color((1, 1, 1, 0))
        
        ax.grid(True, color='gray', linestyle=':', linewidth=0.5, alpha=0.3)
        ax.view_init(elev=28, azim=-115)

        # Labels
        fig.text(0.535, 0.84, rf"Heston Implied Volatility Surface: {ticker}", 
                 color='white', fontsize=16, fontweight='bold', family='monospace', ha='center')
        subtitle = rf"$\kappa={kappa:.2f}, \theta={theta:.2f}, \xi={xi:.2f}, \rho={rho:.2f}, v_0={v0:.3f}$"
        fig.text(0.535, 0.81, subtitle, color='#AAAAAA', fontsize=10, family='monospace', ha='center')

        ax.set_xlabel('Moneyness ($K/S_0$)', color='white', labelpad=10)
        ax.set_ylabel('Maturity ($T$ Years)', color='white', labelpad=10)
        ax.set_zlabel(r'Implied Volatility (%)', color='white', labelpad=10)

        if market_options and valid_needles > 0:
            ax.legend(loc='upper left', bbox_to_anchor=(0.157, 0.797), frameon=False, labelcolor="#D7D7D7", fontsize=10)

        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=15, pad=0.01)
        cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')
        cbar.outline.set_visible(False)

        output_file = f"{filename}_surface.png"
        plt.savefig(output_file, dpi=1000, facecolor='black', bbox_inches='tight')
        plt.close()

def main():
    clear_numba_cache()
    # Create results dir if not exists
    os.makedirs("results", exist_ok=True)
    
    ticker = "NVDA" #"^SPX" # "NVDA" 
    
    # Fetch Market Data
    options, S0 = fetch_options(ticker)
    if not options:
        log(f"No liquidity for {ticker}")
        return

    options.sort(key=lambda x: (x.maturity, x.strike))
    log(f"Target: {ticker} (S0={S0:.2f}) | N={len(options)}")
    
    avg_mkt_price = np.mean([o.market_price for o in options]) if options else 1.0
    r, q = 0.045, 0.002 #11  # SPX Dividend Yield approx 0.0 or embedded in Futures

    # Setup Calibrators
    cal_ana = HestonCalibrator(S0, r, q)
    cal_mc = HestonCalibratorMC(S0, r, q, n_paths=75_000, n_steps=252)
    init_guess = [2.0, 0.04, 0.8, -0.7, 0.024] 

    # --- 1. Analytical Calibration ---
    t0 = time.time()    
    res_ana = cal_ana.calibrate(options, init_guess)
    
    rmse_p_ana = np.sqrt(res_ana['sse'] / len(options))
    log(f"Analytical: rmse={rmse_p_ana:.4f} ({rmse_p_ana/avg_mkt_price:.2%}) , IV-rmse={res_ana['rmse_iv']:.4f} ({res_ana['rmse_iv']:.2%}) ({time.time()-t0:.2f}s)") 
    
    # --- 2. Monte Carlo Calibration ---
    t1 = time.time()
    try:
        res_mc = cal_mc.calibrate(options, init_guess)
        rmse_p_mc = np.sqrt(res_mc['fun'] / len(options)) 
        log(f"MonteCarlo: rmse={rmse_p_mc:.4f} ({rmse_p_mc/avg_mkt_price:.2%}) , IV-rmse={res_mc['rmse_iv']:.4f} ({res_mc['rmse_iv']:.2%}) ({time.time()-t1:.2f}s)")
    except Exception as e:
        log(f"MC Fail: {e}")
        res_mc = res_ana 

    # --- 3. Parameter Comparison ---
    params = ['kappa', 'theta', 'xi', 'rho', 'v0']
    df_params = pd.DataFrame({
        'Ana': [res_ana.get(p, 0.0) for p in params],
        'MC': [res_mc.get(p, 0.0) for p in params]
    }, index=params)
    print(df_params.to_string(float_format="{:.4f}".format))
    
    # --- 4. Save and Visualize ---
    save_results(ticker, S0, r, q, res_ana, res_mc, options)

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def clear_numba_cache():
    for root, dirs, files in os.walk("src"):
        for d in dirs:
            if d == "__pycache__":
                shutil.rmtree(os.path.join(root, d), ignore_errors=True)

if __name__ == "__main__":
    main()