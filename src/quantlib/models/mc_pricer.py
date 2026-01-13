import numpy as np
from ..instruments import Option
from ..market import MarketEnvironment
from .mc_kernels import generate_paths_kernel

class MonteCarloPricer:
    def __init__(self, market: MarketEnvironment):
        self.market = market

    def price_option(self, option: Option, n_paths: int = 10000, n_steps: int = 100) -> float:
        # 1. Kernel generates paths
        paths = generate_paths_kernel(
            self.market.S0,
            self.market.r,
            self.market.sigma,
            option.T,
            n_paths,
            n_steps
        )
        
        # 2. Instrument computes the payoffs
        payoffs = option.payoff(paths)
        
        # 3. Discounting
        discount_factor = np.exp(-self.market.r * option.T)
        return float(np.mean(payoffs) * discount_factor)