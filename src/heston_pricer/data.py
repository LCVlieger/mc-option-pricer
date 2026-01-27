import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Tuple
from .calibration import MarketOption 

def fetch_options(ticker_symbol: str, target_size: int = 150) -> Tuple[List[MarketOption], float]:
    ticker = yf.Ticker(ticker_symbol)
    
    # 1. Fetch Spot
    try:
        S0 = ticker.fast_info.get('last_price', None)
        if S0 is None:
            hist = ticker.history(period="1d")
            S0 = hist['Close'].iloc[-1]
    except:
        return [], 0.0

    print(f"--- Fetching Surface for {ticker_symbol} (Spot: {S0:.2f}) ---")
    
    expirations = ticker.options
    if not expirations: return [], 0.0

    today = datetime.now()
    
    # 2. STABILIZED MATURITY SELECTION (Option A)
    # Filter T < 45 days to prevent "Heston Trap" (Xi explosion)
    MIN_T_YEARS = 55 / 365.25  # ~0.123 years
    
    short_dates = []   # 45 days - 6 months
    med_dates = []     # 6-18 months
    long_dates = []    # > 18 months

    for exp_str in expirations:
        try:
            d = datetime.strptime(exp_str, "%Y-%m-%d")
            T = (d - today).days / 365.25
            
            # [CRITICAL FIX] 
            # Drop everything below 45 days. 
            # These options require Jump-Diffusion (Bates), not Heston.
            if T < MIN_T_YEARS: continue 
            
            if T < 0.5: short_dates.append(exp_str)
            elif T < 1.5: med_dates.append(exp_str)
            else: long_dates.append(exp_str)
        except: continue

    selected_dates = []
    
    # Even spacing helper
    def pick_evenly(lst, n):
        if len(lst) <= n: return lst
        indices = np.linspace(0, len(lst)-1, n, dtype=int)
        return [lst[i] for i in indices]

    selected_dates.extend(pick_evenly(short_dates, 3))
    selected_dates.extend(pick_evenly(med_dates, 4))
    selected_dates.extend(long_dates) # Keep all LEAPS
    
    selected_dates = sorted(list(set(selected_dates)))
    print(f"Scanning {len(selected_dates)} maturities (Filtered T < 14d)...")

    # 3. CONSTRAINED MONEYNESS SEARCH
    # Previous: [0.6 ... 1.6] -> New: [0.75 ... 1.45]
    # Reason: 0.6 delta=1 calls have no vol info. 1.6 calls are often noise.
    target_moneyness = [0.75, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.35, 1.45]
    
    market_options = []

    for exp_str in selected_dates:
        try:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
            T = (exp_date - today).days / 365.25
            
            chain = ticker.option_chain(exp_str).calls
            if chain.empty: continue

            # [FILTER] Liquidity & Data Integrity
            # 1. Open Interest > 50 (Consensus check)
            # 2. Bid > 0.05 (Penny noise check) - CRITICAL for Heston stability
            mask = (chain['openInterest'] > 50) & (chain['bid'] > 0.05)
            
            # Relax OI for LEAPS (T > 1.5) as they are naturally thinner
            if T > 1.5:
                mask = (chain['openInterest'] > 0) & (chain['bid'] > 0.05)
                
            chain = chain[mask]
            if chain.empty: continue

            for m in target_moneyness:
                target_k = S0 * m
                
                # Find closest available strike
                chain['dist'] = (chain['strike'] - target_k).abs()
                candidates = chain.nsmallest(1, 'dist')
                
                if candidates.empty: continue
                
                row = candidates.iloc[0]
                
                # [FILTER] Proximity Check
                # If closest strike is > 7.5% away from target, skip (don't force fit)
                if row['dist'] > (S0 * 0.075): continue

                # Pricing Logic
                bid, ask, last = row.get('bid', 0), row.get('ask', 0), row['lastPrice']
                
                # [FILTER] Spread Integrity
                # If Spread > 40% of mid-price, data is too noisy.
                mid = (bid + ask) / 2.0
                spread = ask - bid
                if mid > 0 and (spread / mid) > 0.4: continue

                price = mid if (bid > 0 and ask > 0) else last
                
                # [FILTER] Hard Arbitrage Check
                # Price must be > Intrinsic + Time Value buffer
                intrinsic = max(S0 - row['strike'], 0)
                if price <= intrinsic: continue 

                # Avoid duplicates
                is_dupe = any(o.strike == row['strike'] and o.maturity == T for o in market_options)
                if not is_dupe:
                    market_options.append(MarketOption(
                        strike=float(row['strike']),
                        maturity=float(T),
                        market_price=float(price),
                        option_type="CALL"
                    ))
        except: continue

    # 4. Final Polish
    market_options.sort(key=lambda x: (x.maturity, x.strike))
    
    if len(market_options) > target_size:
        step = len(market_options) // target_size
        market_options = market_options[::step]

    print(f"Selected {len(market_options)} instruments. Range: T=[{market_options[0].maturity:.2f}, {market_options[-1].maturity:.2f}].")
    return market_options, S0