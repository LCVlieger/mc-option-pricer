import numpy as np
from dataclasses import dataclass, replace
from typing import Dict
from ..instruments import Option
from .process import StochasticProcess

@dataclass
class PricingResult:
    price: float
    std_error: float
    conf_interval_95: tuple[float, float]

class MonteCarloPricer:
    def __init__(self, process: StochasticProcess):
        self.process = process

    def price(self, option: Option, n_paths: int = 10000, n_steps: int = 100, **kwargs) -> PricingResult:
        # Pass kwargs (like 'noise') down to generate_paths
        paths = self.process.generate_paths(option.T, n_paths, n_steps, **kwargs)
        
        # 2. Compute Payoffs
        payoffs = option.payoff(paths)
        
        # 3. Discount
        discount_factor = np.exp(-self.process.market.r * option.T)
        discounted_payoffs = payoffs * discount_factor
        
        # 4. Statistics
        mean_price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs, ddof=1) / np.sqrt(n_paths)
        
        return PricingResult(
            price=mean_price,
            std_error=std_error,
            conf_interval_95=(mean_price - 1.96 * std_error, mean_price + 1.96 * std_error)
        )

    def compute_greeks(self, option: Option, n_paths: int = 10000, n_steps: int = 252, bump_ratio: float = 0.01, seed: int = 42) -> Dict[str, float]:
        """
        Computes Greeks using EXPLICIT Common Random Numbers (CRN).
        We generate the noise matrix once in Python and reuse it for all bumps.
        This eliminates Monte Carlo variance from the Delta/Gamma/Vega calculation.
        """
        original_market = self.process.market
        original_S0 = original_market.S0
        epsilon_s = original_S0 * bump_ratio
        epsilon_v = 0.001 
        
        # --- CRITICAL FIX: Pre-generate Noise for Stability ---
        # Heston needs 2 drivers (Price Brownian, Vol Brownian)
        # Shape: (2, n_steps, n_paths)
        rng = np.random.default_rng(seed)
        # We assume the model needs 2 streams (Euler/CRN kernel expects this)
        Z_CRN = rng.standard_normal((2, n_steps, n_paths))
        
        # 1. Base Price
        # We pass 'noise=Z_CRN' to force the kernel to use these exact numbers
        res_curr = self.price(option, n_paths, n_steps, noise=Z_CRN)
        
        # 2. Delta & Gamma (Spot Bumps)
        # Bump Up
        self.process.market = replace(original_market, S0 = original_S0 + epsilon_s)
        res_up = self.price(option, n_paths, n_steps, noise=Z_CRN)
        
        # Bump Down
        self.process.market = replace(original_market, S0 = original_S0 - epsilon_s)
        res_down = self.price(option, n_paths, n_steps, noise=Z_CRN)
        
        # 3. Vega (Vol Bump)
        self.process.market = replace(original_market, v0 = original_market.v0 + epsilon_v, S0=original_S0)
        res_vega = self.price(option, n_paths, n_steps, noise=Z_CRN)
        
        # Restore Market
        self.process.market = original_market
        
        # Calc Finite Differences
        delta = (res_up.price - res_down.price) / (2 * epsilon_s)
        gamma = (res_up.price - 2 * res_curr.price + res_down.price) / (epsilon_s ** 2)
        vega = (res_vega.price - res_curr.price) / epsilon_v
        
        return {
            "price": res_curr.price,
            "delta": delta,
            "gamma": gamma,
            "vega_v0": vega
        }