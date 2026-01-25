import yfinance as yf
import pandas as pd
from datetime import datetime
from typing import List, Tuple
from .calibration import MarketOption

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
        if len(market_options) > 60: break

        try:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
            T = (exp_date - datetime.now()).days / 365.25
        except: continue
        
        target_bucket = next((name for name, b in buckets.items() 
                              if b['min'] <= T <= b['max'] and b['count'] < max_per_bucket), None)
        if not target_bucket: continue

        try:
            calls = ticker.option_chain(exp_str).calls
        except: continue
        if calls.empty: continue

        calls = calls[(calls['strike'] > S0 * 0.75) & (calls['strike'] < S0 * 1.35)]
        
        selected_indices = set()
        for m in target_moneyness:
            target_strike = S0 * m
            calls['dist'] = (calls['strike'] - target_strike).abs()
            if calls.empty: continue
            best_idx = calls['dist'].idxmin()
            if calls.loc[best_idx, 'dist'] / S0 < 0.025:
                selected_indices.add(best_idx)

        for idx in selected_indices:
            if buckets[target_bucket]['count'] >= max_per_bucket: break
            row = calls.loc[idx]
            price = row['lastPrice'] 
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