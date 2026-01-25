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
        paths = self.process.generate_paths(option.T, n_paths, n_steps, **kwargs)
        payoffs = option.payoff(paths)
    
        discount_factor = np.exp(-self.process.market.r * option.T)
        discounted_payoffs = payoffs * discount_factor
    
        mean_price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs, ddof=1) / np.sqrt(n_paths)
        
        return PricingResult(
            price=mean_price,
            std_error=std_error,
            conf_interval_95=(mean_price - 1.96 * std_error, mean_price + 1.96 * std_error)
        )

    def compute_greeks(self, option: Option, n_paths: int = 10000, n_steps: int = 252, bump_ratio: float = 0.01, seed: int = 42) -> Dict[str, float]:
        """
        Computes Greeks using finite differences. Is implemented with Common Random Numbers (CRN).
        """
        original_market = self.process.market
        original_S0 = original_market.S0
        epsilon_s = original_S0 * bump_ratio
        epsilon_v = 0.001 
        
        # Pre-generate noise for the CRN. 
        rng = np.random.default_rng(seed)
        Z_CRN = rng.standard_normal((2, n_steps, n_paths))
        
        # the price
        res_curr = self.price(option, n_paths, n_steps, noise=Z_CRN)
        
        # The delta and Gamma.   
        self.process.market = replace(original_market, S0 = original_S0 + epsilon_s)
        res_up = self.price(option, n_paths, n_steps, noise=Z_CRN)
        
        self.process.market = replace(original_market, S0 = original_S0 - epsilon_s)
        res_down = self.price(option, n_paths, n_steps, noise=Z_CRN)
        
        # Vega computation. 
        self.process.market = replace(original_market, v0 = original_market.v0 + epsilon_v, S0=original_S0)
        res_vega = self.price(option, n_paths, n_steps, noise=Z_CRN)
        # restore market back. 
        self.process.market = original_market

        delta = (res_up.price - res_down.price) / (2 * epsilon_s)
        gamma = (res_up.price - 2 * res_curr.price + res_down.price) / (epsilon_s ** 2)
        vega = (res_vega.price - res_curr.price) / epsilon_v
        
        return {
            "price": res_curr.price,
            "delta": delta,
            "gamma": gamma,
            "vega_v0": vega
        }