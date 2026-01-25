from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from ..market import MarketEnvironment
from .mc_kernels import generate_paths_kernel, generate_heston_paths, generate_heston_paths_crn

class StochasticProcess(ABC):
    def __init__(self, market: MarketEnvironment):
        self.market = market

    @abstractmethod
    def generate_paths(self, T: float, n_paths: int, n_steps: int) -> np.ndarray:
        pass

class BlackScholesProcess(StochasticProcess):
    def generate_paths(self, T: float, n_paths: int, n_steps: int) -> np.ndarray:
        return generate_paths_kernel(
            self.market.S0, self.market.r, self.market.q, self.market.sigma,
            T, n_paths, n_steps
        )

class HestonProcess(StochasticProcess):
    def generate_paths(self, T: float, n_paths: int, n_steps: int, noise=None) -> np.ndarray:
        # Common params unpacking
        args = (
            self.market.S0, self.market.r, self.market.q,
            self.market.v0, self.market.kappa, self.market.theta,
            self.market.xi, self.market.rho, T, n_paths, n_steps
        )

        if noise is not None:
            # Calibration Mode (CRN)
            return generate_heston_paths_crn(*args, noise)
        
        # Pricing Mode (Standard Euler)
        return generate_heston_paths(*args)