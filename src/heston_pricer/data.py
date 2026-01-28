import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Tuple
from .calibration import MarketOption 

def fetch_options(ticker_symbol: str, target_size: int = 100) -> Tuple[List[MarketOption], float]:
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
    
    # 2. MAXIMIZED MATURITY SELECTION (T <= 3.0)
    MIN_T_YEARS = 165 / 365.25
    MAX_T_YEARS = 2.7
    
    valid_dates = []

    for exp_str in expirations:
        try:
            d = datetime.strptime(exp_str, "%Y-%m-%d")
            T = (d - today).days / 365.25
            
            if MIN_T_YEARS <= T <= MAX_T_YEARS:
                valid_dates.append(exp_str)
        except: continue

    # To get ~100 options, we need as many dates as possible. 
    # Only downsample if we have an excessive amount (e.g. > 24 dates).
    if len(valid_dates) > 24:
        indices = np.linspace(0, len(valid_dates)-1, 24, dtype=int)
        selected_dates = [valid_dates[i] for i in indices]
    else:
        selected_dates = valid_dates

    # Sort dates to ensure chronological processing
    selected_dates = sorted(list(set(selected_dates)))
    print(f"Scanning {len(selected_dates)} maturities (Max T: {MAX_T_YEARS})...")

    # 3. HIGH DENSITY MONEYNESS SEARCH
    # Create a dense grid from 60% to 150% moneyness to ensure high capture rate
    target_moneyness = np.arange(0.6, 1.55, 0.05) 
    
    market_options = []

    for exp_str in selected_dates:
        try:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
            T = (exp_date - today).days / 365.25
            
            raw_chain = ticker.option_chain(exp_str).calls
            if raw_chain.empty: continue

            # --- LOGIC BRANCHING ---
            
            # Strict Logic: OI > 50 ensures liquidity
            mask_strict = (raw_chain['openInterest'] > 50) & (raw_chain['bid'] > 0.05)
            if T > 1.5:
                # Relax OI for LEAPS slightly as volume is naturally lower
                mask_strict = (raw_chain['openInterest'] > 10) & (raw_chain['bid'] > 0.05)
            
            chain = raw_chain[mask_strict].copy()
            use_fallback = False

            # Fallback Logic
            if chain.empty:
                mask_relaxed = (raw_chain['lastPrice'] > 0) | (raw_chain['bid'] > 0)
                chain = raw_chain[mask_relaxed].copy()
                use_fallback = True 
            
            if chain.empty: continue

            for m in target_moneyness:
                target_k = S0 * m
                
                chain['dist'] = (chain['strike'] - target_k).abs()
                candidates = chain.nsmallest(1, 'dist')
                
                if candidates.empty: continue
                
                row = candidates.iloc[0]
                
                # Proximity Check: Strict 7.5%, Fallback 15%
                limit_dist = S0 * 0.15 if use_fallback else S0 * 0.075
                if row['dist'] > limit_dist: continue

                # Pricing Logic
                bid, ask, last = row.get('bid', 0), row.get('ask', 0), row['lastPrice']
                
                mid = (bid + ask) / 2.0
                spread = ask - bid
                
                if not use_fallback:
                    if mid > 0 and (spread / mid) > 0.4: continue

                price = mid if (bid > 0 and ask > 0) else last
                
                # Arbitrage Check
                intrinsic = max(S0 - row['strike'], 0)
                
                if price <= intrinsic:
                    if not use_fallback:
                        continue 
                    else:
                        price = intrinsic + 0.05 

                # Deduplication
                is_dupe = any(o.strike == row['strike'] and o.maturity == T for o in market_options)
                if not is_dupe:
                    market_options.append(MarketOption(
                        strike=float(row['strike']),
                        maturity=float(T),
                        market_price=float(price),
                        option_type="CALL"
                    ))
        except: continue

    # 4. Final Downsampling
    market_options.sort(key=lambda x: (x.maturity, x.strike))
    
    # If we exceeded the target, step nicely to reduce density uniformly
    if len(market_options) > target_size:
        step = len(market_options) / target_size
        indices = [int(i * step) for i in range(target_size)]
        market_options = [market_options[i] for i in indices]

    print(f"Selected {len(market_options)} instruments. Range: T=[{market_options[0].maturity:.2f}, {market_options[-1].maturity:.2f}].")
        
    return market_options, S0