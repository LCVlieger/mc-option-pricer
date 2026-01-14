import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import brentq
from scipy.stats import norm
from heston_pricer.analytics import HestonAnalyticalPricer

def impl_vol(price, S0, K, T, r, q):
    """ 
    Inverts Black-Scholes-Merton (with Dividends) to find Implied Vol 
    """
    if price <= 0: return 0.0
    
    def obj(sigma):
        # BS with dividends: Drift is (r - q)
        d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Call = S * exp(-qT) * N(d1) - K * exp(-rT) * N(d2)
        bs_price = S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return bs_price - price
    
    try:
        # Search for vol between 1% and 500%
        return brentq(obj, 0.01, 5.0)
    except:
        return np.nan

def main():
    print("Generating SPX Volatility Surface...")

    # 1. Calibrated Parameters (From your run)
    S0 = 6896.04
    r = 0.045
    q = 0.015
    
    # "The DNA of the Market"
    params = {
        'kappa': 0.1000, 
        'theta': 0.0015, 
        'xi': 2.0000, 
        'rho': -0.7814, 
        'v0': 0.0400
    }
    
    print(f"Parameters: {params}")
    
    # 2. Define the Grid
    # Moneyness: 80% to 120% of Spot
    moneyness = np.linspace(0.8, 1.2, 30) 
    strikes = S0 * moneyness
    
    # Maturities: Short end (2 weeks) to Long end (1 year)
    # We focus on < 1 year because that's where the "Skew" is most violent
    maturities = np.linspace(0.04, 1.0, 30)
    
    X, Y = np.meshgrid(strikes, maturities)
    Z = np.zeros_like(X)
    
    # 3. Compute Surface
    print("Computing surface points (this might take 10s)...")
    for i in range(len(maturities)):
        for j in range(len(strikes)):
            T = maturities[i]
            K = strikes[j]
            
            # Heston Price (using q)
            price = HestonAnalyticalPricer.price_european_call(S0, K, T, r, q, **params)
            
            # Implied Vol (using q)
            iv = impl_vol(price, S0, K, T, r, q)
            Z[i, j] = iv * 100 # Convert to %

    # 4. Plotting
    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Surface
    surf = ax.plot_surface(X, Y, Z, cmap=cm.inferno, linewidth=0, antialiased=True, alpha=0.9)
    
    ax.set_xlabel(f'Strike Price ($K$) \n(Spot={S0:.0f})')
    ax.set_ylabel('Maturity ($T$)')
    ax.set_zlabel('Implied Volatility (%)')
    ax.set_title(f'S&P 500 Volatility Surface (Calibrated)\nrho={params["rho"]:.2f}, xi={params["xi"]:.2f}')
    
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.view_init(elev=30, azim=-120)
    
    filename = "spx_vol_surface.png"
    plt.savefig(filename)
    print(f"Surface saved to {filename}")
    plt.show()

if __name__ == "__main__":
    main()