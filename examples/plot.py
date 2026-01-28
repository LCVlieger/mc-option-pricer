import json
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import gaussian_filter 

# Local package imports
try:
    from heston_pricer.calibration import implied_volatility
    from heston_pricer.analytics import HestonAnalyticalPricer
except ImportError:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
    from heston_pricer.calibration import implied_volatility
    from heston_pricer.analytics import HestonAnalyticalPricer

"""
4_visualize_surface.py (Final Fix)
----------------------
- Strict Clipping: Removes all points outside T=[0.1, 2.5] and M=[0.5, 1.8]
- Restored Professional Style
"""

class ReconstructedOption:
    def __init__(self, strike, maturity, price):
        self.strike = float(strike)
        self.maturity = float(maturity)
        self.market_price = float(price)

def load_latest_calibration():
    patterns = ['results/calibration_*_meta.json', 'calibration_*_meta.json']
    files = []
    for p in patterns: files.extend(glob.glob(p))
    
    if not files: raise FileNotFoundError("No calibration meta file found.")
    
    latest_meta = max(files, key=os.path.getctime)
    base_name = latest_meta.replace("_meta.json", "")
    print(f"Loading: {base_name}...")
    
    with open(latest_meta, 'r') as f: data = json.load(f)
    
    csv_file = f"{base_name}_prices.csv"
    market_options = []
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            market_options.append(ReconstructedOption(row['K'], row['T'], row['Mkt']))

    return data, market_options, base_name

def plot_surface_professional(S0, r, q, params, ticker, filename, market_options=None):
    kappa, theta, xi, rho, v0 = params['kappa'], params['theta'], params['xi'], params['rho'], params['v0']

    # --- 1. CONFIGURATION ---
    LOWER_M, UPPER_M = 0.5, 1.8 
    LOWER_T, UPPER_T = 0.1, 2.5
    GRID_DENSITY = 100 

    M_range = np.linspace(LOWER_M, UPPER_M, GRID_DENSITY)
    T_range = np.linspace(LOWER_T, UPPER_T, GRID_DENSITY)
    X, Y = np.meshgrid(M_range, T_range)
    Z = np.zeros_like(X)

    # --- 2. CALCULATION ---
    print(f"-> Calculating {GRID_DENSITY}x{GRID_DENSITY} surface points...")
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

    # --- 3. SMOOTHING ---
    mask = np.isnan(Z)
    if np.any(mask):
        Z = pd.DataFrame(Z).interpolate(method='linear', axis=1).ffill(axis=1).bfill(axis=1).values
    Z_smooth = gaussian_filter(Z, sigma=0.6)

    # --- 4. PLOTTING ---
    with plt.style.context('dark_background'):
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Surface
        surf = ax.plot_surface(X, Y, Z_smooth, cmap=cm.RdYlBu_r, 
                               rcount=100, ccount=100,  
                               edgecolor='black',       
                               linewidth=0.085,         
                               alpha=0.8,               
                               shade=False,             
                               antialiased=True,        
                               zorder=1)

        # Market Needles (Strict Clipping)
        if market_options:
            # FIX: Filter strictly for BOTH Moneyness AND Maturity to prevent "ghost" points
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

                m_idx = (np.abs(M_range - m_mkt)).argmin()
                t_idx = (np.abs(T_range - t_mkt)).argmin()
                iv_mod = Z_smooth[t_idx, m_idx]

                if np.isnan(iv_mod): continue

                diff = abs(iv_mkt - iv_mod)
                if diff > 0.018: continue
                
                valid_needles += 1
                is_above = iv_mkt >= iv_mod
                dot_zorder = 10 if is_above else 1

                ax.plot([m_mkt, m_mkt], [t_mkt, t_mkt], [iv_mod, iv_mkt], 
                        color='white', linestyle='-', linewidth=1.1, alpha=0.6, zorder=dot_zorder)
                
                lbl = 'Market Price-IV' if valid_needles == 1 else ""
                ax.plot([m_mkt], [t_mkt], [iv_mkt], 
                        marker='o', linestyle='None',
                        color="#F0F0F0", markersize=3.875, alpha=0.905, 
                        zorder=dot_zorder, label=lbl)

        # --- 5. AESTHETICS ---
        ax.dist = 11
        ax.set_xlim(LOWER_M, UPPER_M)
        ax.set_ylim(UPPER_T, LOWER_T) # Inverted T axis as per standard convention often used
        
        # Transparent Panes
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        
        # Grid
        ax.xaxis._axinfo["grid"]['color'] = (1,1,1,0.15)
        ax.yaxis._axinfo["grid"]['color'] = (1,1,1,0.15)
        ax.zaxis._axinfo["grid"]['color'] = (1,1,1,0.15)
        ax.grid(True, color='gray', linestyle=':', linewidth=0.5, alpha=0.3)
        ax.view_init(elev=28, azim=-115) 

        # Labels
        fig.text(0.535, 0.84, rf"Heston Implied Volatility Surface: {ticker}", 
                 color='white', fontsize=16, fontweight='bold', family='monospace', ha='center')

        subtitle = rf"$\kappa={kappa:.2f}, \theta={theta:.2f}, \xi={xi:.2f}, \rho={rho:.2f}, v_0={v0:.3f}$"
        fig.text(0.535, 0.81, subtitle, 
                 color='#AAAAAA', fontsize=10, family='monospace', ha='center')

        ax.set_xlabel('Moneyness ($K/S_0$)', color='white', labelpad=10)
        ax.set_ylabel('Maturity ($T$ Years)', color='white', labelpad=10)
        ax.set_zlabel(r'Implied Volatility (%)', color='white', labelpad=10)

        # Legend
        if market_options and valid_needles > 0:
            leg = ax.legend(
                loc='upper left',
                bbox_to_anchor=(0.157375, 0.7975), 
                frameon=False,
                labelcolor="#D7D7D7",
                fontsize=10,
                handletextpad=0.5
            )

        # Colorbar
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=15, pad=0.01)
        cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')
        cbar.outline.set_visible(False)

        save_path = f"{filename}_surface_professional.png"
        plt.savefig(save_path, dpi=1000, facecolor='black', bbox_inches='tight')
        print(f"-> Saved: {save_path}")
        plt.close()

def main():
    try:
        data, market_options, base_name = load_latest_calibration()
        ticker = base_name.split("calibration_")[1].split("_")[0] if "calibration_" in base_name else "Asset"
        
        plot_surface_professional(
            data['market']['S0'], data['market']['r'], data['market']['q'], 
            data['monte_carlo_results'], ticker, base_name, market_options
        )
    except Exception as e:
        print(f"[Error] {e}")

if __name__ == "__main__":
    main()