import json
import glob
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import gaussian_filter 

# Local package imports
try:
    from heston_pricer.calibration import implied_volatility, HestonCalibrator
    from heston_pricer.analytics import HestonAnalyticalPricer
except ImportError:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
    from heston_pricer.calibration import implied_volatility, HestonCalibrator
    from heston_pricer.analytics import HestonAnalyticalPricer

"""
4_visualize_surface.py (Portfolio Final)
----------------------
1. Loads existing calibration.
2. Compares Analytical vs Monte Carlo performance.
3. Generates the final, honest, publication-grade plot.
"""

class ReconstructedOption:
    def __init__(self, strike, maturity, price, option_type="CALL"):
        self.strike = float(strike)
        self.maturity = float(maturity)
        self.market_price = float(price)
        self.option_type = str(option_type)  # <--- Added missing attribute

def load_latest_calibration():
    patterns = ['results/calibration_*_meta.json', 'calibration_*_meta.json']
    files = []
    for p in patterns: files.extend(glob.glob(p))
    
    if not files: raise FileNotFoundError("No calibration meta file found.")
    
    latest_meta = max(files, key=os.path.getctime)
    base_name = latest_meta.replace("_meta.json", "")
    print(f"Loading Artifact: {base_name}...")
    
    with open(latest_meta, 'r') as f: data = json.load(f)
    
    csv_file = f"{base_name}_prices.csv"
    market_options = []
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            # Safe get for 'Type' to support older CSVs
            otype = row['Type'] if 'Type' in row else "CALL"
            market_options.append(ReconstructedOption(row['K'], row['T'], row['Mkt'], otype))

    return data, market_options, base_name

def refine_calibration(S0, r, q, initial_params, market_options):
    """
    Filters outliers based on initial parameters and re-runs calibration.
    """
    kappa = initial_params['kappa'] 
    theta = initial_params['theta']
    xi = initial_params['xi']
    rho = initial_params['rho']
    v0 = initial_params['v0']
    
    p_init_vals = [kappa, theta, xi, rho, v0]
    print(f"\n[Refinement] Checking {len(market_options)} options against loaded model...")
    clean_options = []
    dropped = 0
    
    THRESHOLD = 1.0 
    
    for opt in market_options:
        try:
            # --- FIX: Dispatch correctly based on option type ---
            if opt.option_type == "PUT":
                p_mod = HestonAnalyticalPricer.price_european_put(
                    S0, opt.strike, opt.maturity, r, q, kappa, theta, xi, rho, v0
                )
            else:
                p_mod = HestonAnalyticalPricer.price_european_call(
                    S0, opt.strike, opt.maturity, r, q, kappa, theta, xi, rho, v0
                )
            
            # Pass option_type to implied_volatility
            iv_mod = implied_volatility(p_mod, S0, opt.strike, opt.maturity, r, q, opt.option_type)
            iv_mkt = implied_volatility(opt.market_price, S0, opt.strike, opt.maturity, r, q, opt.option_type)
            
            if np.isnan(iv_mod) or np.isnan(iv_mkt) or iv_mod == 0.0 or iv_mkt == 0.0: 
                dropped += 1
                continue
            
            err = abs(iv_mkt - iv_mod)
            if err <= THRESHOLD:
                clean_options.append(opt)
            else:
                dropped += 1
        except Exception:
            dropped += 1
            
    print(f"-> Dropped {dropped} extreme outliers. Re-calibrating on {len(clean_options)} instruments...")
    
    cal = HestonCalibrator(S0, r, q)
    t0 = time.time()
    res_final = cal.calibrate(clean_options, init_guess=p_init_vals)
    print(f"-> Optimization Complete ({time.time()-t0:.2f}s)")
    
    return res_final, clean_options, dropped

def plot_surface_professional(S0, r, q, params, ticker, filename, market_options, data_full, dropped_count):
    kappa, theta, xi, rho, v0 = params['kappa'], params['theta'], params['xi'], params['rho'], params['v0']

    # --- 1. CONFIGURATION ---
    LOWER_M, UPPER_M = 0.5, 1.8 
    LOWER_T, UPPER_T = 0.1, 2.5
    GRID_DENSITY = 35

    M_range = np.linspace(LOWER_M, UPPER_M, GRID_DENSITY)
    T_range = np.linspace(LOWER_T, UPPER_T, GRID_DENSITY)
    X, Y = np.meshgrid(M_range, T_range)
    Z = np.zeros_like(X)

    # --- 2. CALCULATION ---
    # We plot the Call surface as the canonical representation
    print(f"-> Generating Surface for: kappa={kappa:.2f}, xi={xi:.2f}, v0={v0:.3f}")
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            T_val, M_val = Y[i, j], X[i, j]
            price = HestonAnalyticalPricer.price_european_call(
                S0, S0 * M_val, T_val, r, q, kappa, theta, xi, rho, v0
            )
            try:
                iv = implied_volatility(price, S0, S0 * M_val, T_val, r, q, "CALL")
                Z[i, j] = iv if 0.01 < iv < 2.5 else np.nan
            except:
                Z[i, j] = np.nan

    mask = np.isnan(Z)
    if np.any(mask):
        Z = pd.DataFrame(Z).interpolate(method='linear', axis=1).ffill(axis=1).bfill(axis=1).values
    Z_smooth = gaussian_filter(Z, sigma=0.8)

    # --- 3. PLOTTING ---
    with plt.style.context('dark_background'):
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(X, Y, Z_smooth, cmap=cm.RdYlBu_r, 
                               rcount=100, ccount=100,  
                               edgecolor='black',       
                               linewidth=0.085,         
                               alpha=0.8,               
                               shade=False,             
                               antialiased=True,        
                               zorder=1)

        if market_options:
            plot_opts = [
                o for o in market_options 
                if (LOWER_M <= (o.strike/S0) <= UPPER_M) and (LOWER_T <= o.maturity <= UPPER_T)
            ]
            
            valid_needles = 0
            for opt in plot_opts:
                m_mkt, t_mkt = opt.strike / S0, opt.maturity
                try:
                    # --- FIX: Correct IV calculation for Puts in the plot ---
                    iv_mkt = implied_volatility(opt.market_price, S0, opt.strike, opt.maturity, r, q, opt.option_type)
                    if iv_mkt < 0.01 or iv_mkt > 2.5: continue
                except: continue

                m_idx = (np.abs(M_range - m_mkt)).argmin()
                t_idx = (np.abs(T_range - t_mkt)).argmin()
                iv_mod = Z_smooth[t_idx, m_idx]

                if np.isnan(iv_mod): continue
                
                valid_needles += 1
                is_above = iv_mkt >= iv_mod
                dot_zorder = 10 if is_above else 1

                ax.plot([m_mkt, m_mkt], [t_mkt, t_mkt], [iv_mod, iv_mkt], 
                        color='white', linestyle='-', linewidth=0.5, alpha=0.4, zorder=dot_zorder)
                
                lbl = 'Market Price-IV' if valid_needles == 1 else ""
                ax.plot([m_mkt, m_mkt], [t_mkt, t_mkt], [iv_mkt], 
                        marker='o', linestyle='None',
                        color="#F0F0F0", markersize=4.0, alpha=0.9, 
                        zorder=dot_zorder, label=lbl)

        # --- 4. AESTHETICS ---
        ax.dist = 11
        ax.set_xlim(LOWER_M, UPPER_M)
        ax.set_ylim(UPPER_T, LOWER_T) 
        
        ax.xaxis.set_pane_color((1, 1, 1, 0))
        ax.yaxis.set_pane_color((1, 1, 1, 0))
        ax.zaxis.set_pane_color((1, 1, 1, 0))
        
        ax.grid(True, color='gray', linestyle=':', linewidth=0.5, alpha=0.3)
        ax.view_init(elev=28, azim=-115) 

        fig.text(0.535, 0.84, rf"Heston Implied Volatility Surface: {ticker}", 
                 color='white', fontsize=16, fontweight='bold', family='monospace', ha='center')
        subtitle = rf"$\kappa={kappa:.2f}, \theta={theta:.2f}, \xi={xi:.2f}, \rho={rho:.2f}, v_0={v0:.3f}$"
        fig.text(0.535, 0.81, subtitle, color='#AAAAAA', fontsize=10, family='monospace', ha='center')

        # --- 5. PERFORMANCE METRICS OVERLAY ---
        mc_res = data_full.get('monte_carlo_results', {})
        # Handle cases where MC might have failed/not run
        mc_rmse = mc_res.get('rmse_iv', 0) if isinstance(mc_res, dict) else 0.0

        ax.set_xlabel('Moneyness ($K/S_0$)', color='white', labelpad=10)
        ax.set_ylabel('Maturity ($T$ Years)', color='white', labelpad=10)
        ax.set_zlabel(r'Implied Volatility (%)', color='white', labelpad=10)

        if market_options and valid_needles > 0:
            ax.legend(loc='upper left', bbox_to_anchor=(0.157, 0.797), frameon=False, labelcolor="#D7D7D7", fontsize=10)

        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=15, pad=0.01)
        cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')
        cbar.outline.set_visible(False)

        save_path = f"{filename}_surface_refined.png"
        plt.savefig(save_path, dpi=600, facecolor='black', bbox_inches='tight')
        print(f"-> Saved: {save_path}")
        plt.close()

def main():
    try:
        data, market_options, base_name = load_latest_calibration()
        S0, r, q = data['market']['S0'], data['market']['r'], data['market']['q']
        
        initial_params = data.get('monte_carlo_results', data.get('analytical'))
        ticker = base_name.split("calibration_")[1].split("_")[0] if "calibration_" in base_name else "Asset"
        
        new_params, clean_options, dropped = refine_calibration(S0, r, q, initial_params, market_options)
        
        plot_surface_professional(S0, r, q, new_params, ticker, base_name, clean_options, data, dropped)
        
    except Exception as e:
        print(f"[Error] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()