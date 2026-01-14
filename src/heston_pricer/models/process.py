from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from ..market import MarketEnvironment
from .mc_kernels import generate_paths_kernel, generate_heston_paths

class StochasticProcess(ABC):
    """
    Abstract Base Class for any Ito Process (BS, Heston, Local Vol, etc.)
    """
    def __init__(self, market: MarketEnvironment):
        self.market = market

    @abstractmethod
    def generate_paths(self, T: float, n_paths: int, n_steps: int) -> np.ndarray:
        pass

class BlackScholesProcess(StochasticProcess):
    def generate_paths(self, T: float, n_paths: int, n_steps: int) -> np.ndarray:
        # Map class attributes to the Numba kernel arguments
        return generate_paths_kernel(
            self.market.S0,
            self.market.r,
            self.market.sigma, # Uses sigma from market
            T,
            n_paths,
            n_steps
        )

class HestonProcess(StochasticProcess):
    def generate_paths(self, T: float, n_paths: int, n_steps: int) -> np.ndarray:
        # Map class attributes to the Heston kernel
        return generate_heston_paths(
            self.market.S0,
            self.market.r,
            self.market.v0,
            self.market.kappa,
            self.market.theta,
            self.market.xi,
            self.market.rho,
            T,
            n_paths,
            n_steps
        )