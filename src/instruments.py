from abc import ABC, abstractmethod
from enum import Enum
import numpy as np

class OptionType(Enum):
    CALL = 1
    PUT = -1

class Option(ABC):
    """
    Base Class for options. Forces each option (Asian, Barrier, etc.)
    to have a payoff method. 
    """
    def __init__(self, K: float, T: float, option_type: OptionType): 
        self.K = K
        self.T = T
        self.option_type = option_type

    @abstractmethod
    def payoff(self, prices: np.ndarray) -> np.ndarray:
        """
        Compute payoff given a price path.
        prices shape: (N_paths, N_timesteps)
        """
        pass

class EuropeanOption(Option):
    """
    Unified class for European Calls and Puts.
    """
    def payoff(self, prices: np.ndarray) -> np.ndarray:
        # S_T is the price at the final time step
        S_T = prices[:, -1]
        
        # Vectorized payoff using the phi parameter (1 or -1)
        # Call: 1 * (S - K)  -> S - K
        # Put: -1 * (S - K)  -> K - S
        phi = self.option_type.value
        return np.maximum(phi * (S_T - self.K), 0)