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
        # The Pricer now owns a Process (which owns the Market)
        self.process = process

    def price(self, option: Option, n_paths: int = 10000, n_steps: int = 100) -> PricingResult:
        # 1. Delegate path generation to the Process
        paths = self.process.generate_paths(option.T, n_paths, n_steps)
        
        # 2. Compute Payoffs
        payoffs = option.payoff(paths)
        
        # 3. Discount
        # Access rate 'r' from the process's market environment
        discount_factor = np.exp(-self.process.market.r * option.T)
        discounted_payoffs = payoffs * discount_factor
        
        # 4. Statistics (Standard Monte Carlo)
        # Note: We removed Antithetic pairing here because the Heston kernel 
        # as currently written produces independent paths.
        mean_price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs, ddof=1) / np.sqrt(n_paths)
        
        # 95% Confidence Interval
        ci_lower = mean_price - 1.96 * std_error
        ci_upper = mean_price + 1.96 * std_error
        
        return PricingResult(
            price=mean_price,
            std_error=std_error,
            conf_interval_95=(ci_lower, ci_upper)
        )

    def compute_greeks(self, option: Option, n_paths: int = 10000, bump_ratio: float = 0.01, seed: int = 42) -> Dict[str, float]:
        """
        Computes Delta and Gamma using Finite Differences.
        Adapted to work with the Swappable Process architecture.
        """
        # We need to access the market *inside* the process
        original_market = self.process.market
        original_S0 = original_market.S0
        epsilon = original_S0 * bump_ratio
        
        # 1. Central Price
        np.random.seed(seed)
        res_curr = self.price(option, n_paths)
        
        # 2. Up Price
        # We create a modified market and 'inject' it into the process
        market_up = replace(original_market, S0 = original_S0 + epsilon)
        self.process.market = market_up
        
        np.random.seed(seed) 
        res_up = self.price(option, n_paths)
        
        # 3. Down Price
        market_down = replace(original_market, S0 = original_S0 - epsilon)
        self.process.market = market_down
        
        np.random.seed(seed)
        res_down = self.price(option, n_paths)
        
        # 4. Cleanup: Restore the original market to the process
        self.process.market = original_market

        # Finite Differences Calculation
        delta = (res_up.price - res_down.price) / (2 * epsilon)
        gamma = (res_up.price - 2 * res_curr.price + res_down.price) / (epsilon ** 2)
        
        return {
            "price": res_curr.price,
            "std_error": res_curr.std_error,
            "delta": delta,
            "gamma": gamma
        }