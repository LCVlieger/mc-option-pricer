from dataclasses import dataclass

@dataclass
class MarketEnvironment:
    S0: float          # Spot Price
    r: float           # Risk-free Rate
    q: float = 0.0     # Dividend Yield 
    sigma: float = 0.2 # Volatility (Black-Scholes)
     
    # Heston Parameters
    v0: float = 0.04
    kappa: float = 1.0
    theta: float = 0.04
    xi: float = 0.1
    rho: float = -0.7