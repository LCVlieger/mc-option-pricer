import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Tuple
from heston_pricer.calibration import MarketOption

def fetch_spx_options(min_open_interest: int = 500) -> Tuple[List[MarketOption], float]:
    """
    Fetches S&P 500 (^SPX) option chains, scanning until it finds 
    valid liquid maturities (T > 14 days).
    """
    ticker_symbol = "^SPX"
    print(f"--- 1. Connecting to Yahoo Finance ({ticker_symbol}) ---")
    
    ticker = yf.Ticker(ticker_symbol)
    
    try:
        hist = ticker.history(period="1d")
        S0 = hist['Close'].iloc[-1]
        print(f"    Spot Price (S0): {S0:.2f}")
    except Exception:
        raise ValueError("Could not fetch spot price. Check Internet/Ticker.")

    expirations = ticker.options
    print(f"    Found {len(expirations)} total expiration dates.")
    
    market_options = []
    found_maturities = 0
    REQUIRED_MATURITIES = 3  # We want at least 3 distinct valid expiration dates
    
    print("\n--- 2. Scanning Option Chains ---")
    
    # Scan up to the first 20 expirations to bypass daily/weekly noise
    for exp_date_str in expirations[:25]:
        if found_maturities >= REQUIRED_MATURITIES:
            break
            
        # Calculate T
        exp_date = datetime.strptime(exp_date_str, "%Y-%m-%d")
        T = (exp_date - datetime.now()).days / 365.25
        
        # FILTER 1: Time to Maturity
        # We want > 14 days (cleaner volatility) and < 1.5 years
        if T < 14/365.25:
            # print(f"    Skipping {exp_date_str} (Too short: {T*365:.1f} days)")
            continue
        if T > 1.5:
            continue
            
        print(f"    -> Checking Expiry: {exp_date_str} (T={T:.3f}y)")
        
        try:
            chain = ticker.option_chain(exp_date_str)
            calls = chain.calls
        except Exception:
            continue

        # FILTER 2: Moneyness & Liquidity
        # Moneyness: 0.85 < K/S0 < 1.15
        mask = (
            (calls['strike'] > S0 * 0.85) & 
            (calls['strike'] < S0 * 1.15) & 
            (calls['openInterest'] > min_open_interest)
        )
        
        filtered_calls = calls[mask]
        
        if filtered_calls.empty:
            print(f"       [!] No liquid options found for this date (OI > {min_open_interest}).")
            continue
            
        # Data Extraction
        count = 0
        for _, row in filtered_calls.iterrows():
            bid, ask = row['bid'], row['ask']
            # Use Mid if valid, else Last
            price = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else row['lastPrice']
            
            # Sanity check: Price must be > 0.05
            if price < 0.05: continue
            
            market_options.append(MarketOption(
                strike=row['strike'],
                maturity=T,
                market_price=price,
                option_type="CALL"
            ))
            count += 1
            
        if count > 0:
            print(f"       Loaded {count} options.")
            found_maturities += 1
            
    print(f"\n--- 3. Summary ---")
    print(f"    Total Liquid Options Collected: {len(market_options)}")
    
    if len(market_options) == 0:
        print("\n[!] CRITICAL: Still 0 options. Try lowering 'min_open_interest' to 100.")
    
    return market_options, S0

if __name__ == "__main__":
    fetch_spx_options()