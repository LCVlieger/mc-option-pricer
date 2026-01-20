import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class MarketOption:
    strike: float
    maturity: float
    market_price: float
    option_type: str = "CALL"

def fetch_options(ticker_symbol: str = "KO", min_open_interest: int = 10, max_per_bucket: int = 20) -> Tuple[List[MarketOption], float]:
    """
    Fetches current options for ANY ticker. 
    Default is 'KO' (Coca-Cola) -> Low Vol-of-Vol Proxy.
    """
    ticker = yf.Ticker(ticker_symbol)
    
    # 1. Get Reference Spot (Last Close)
    try:
        hist = ticker.history(period="5d")
        if hist.empty: raise ValueError("No history found for ticker")
        S0 = hist['Close'].iloc[-1]
        print(f"[Data] Reference Spot ({ticker_symbol}): {S0:.2f}")
    except Exception as e:
        print(f"[Error] Failed to fetch spot: {e}")
        return [], 0.0

    # 2. Scan Chains
    market_options = []
    
    # Adjusted buckets slightly for single stocks which might have different maturities
    buckets = {
        "Short":  {'min': 0.02, 'max': 0.25, 'count': 0},
        "Medium": {'min': 0.25, 'max': 0.75, 'count': 0},
        "Long":   {'min': 0.75, 'max': 2.00, 'count': 0}
    }
    
    print(f"[Data] Scanning option chains for {ticker_symbol}...")

    expirations = ticker.options
    if not expirations:
        print("[Error] No expiration dates found.")
        return [], 0.0

    for exp_str in expirations[:30]:
        if len(market_options) > 60: break

        try:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
            T = (exp_date - datetime.now()).days / 365.25
        except: continue
        
        target_bucket = next((name for name, b in buckets.items() 
                              if b['min'] <= T <= b['max'] and b['count'] < max_per_bucket), None)
        if not target_bucket: continue

        try:
            chain = ticker.option_chain(exp_str)
            calls = chain.calls
        except: continue
        
        if calls.empty: continue

        # 3. FILTERING
        mask = (calls['strike'] > S0 * 0.80) & (calls['strike'] < S0 * 1.20)
        candidates = calls[mask]
        
        for _, row in candidates.iterrows():
            if buckets[target_bucket]['count'] >= max_per_bucket: break
            
            # Basic Validation
            last_trade_val = row.get('lastTradeDate', None)
            if last_trade_val is None: continue
            
            try:
                # Handle timezone aware timestamps
                if isinstance(last_trade_val, pd.Timestamp):
                    trade_date = last_trade_val.date()
                else:
                    trade_date = pd.to_datetime(last_trade_val).date()
            except: continue

            # Relax recency for single stocks (often less liquid than SPY)
            days_diff = (datetime.now().date() - trade_date).days
            if days_diff > 7: continue 

            price = row['lastPrice']
            if price < 0.05: continue
            
            intrinsic = max(0, S0 - row['strike'])
            if price < (intrinsic - 2.0): continue

            market_options.append(MarketOption(
                strike=float(row['strike']),
                maturity=float(T),
                market_price=float(price)
            ))
            buckets[target_bucket]['count'] += 1

    return market_options, S0

if __name__ == "__main__":
    # Test with Coca-Cola (Low Vol Proxy)
    opts, spot = fetch_options("KO")
    print(f"Fetched {len(opts)} options for KO. Spot: {spot:.2f}")