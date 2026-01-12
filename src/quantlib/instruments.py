from abc import ABC, abstractmethod
from enum import Enum
import numpy as np

class OptionType(Enum):
    CALL = 1
    PUT = -1

class Option(ABC):
    def __init__(self, K: float, T: float, option_type: OptionType):
        self.K = K
        self.T = T
        self.option_type = option_type

    @abstractmethod
    def payoff(self, prices: np.ndarray) -> np.ndarray:
        pass

class EuropeanOption(Option):
    def payoff(self, prices: np.ndarray) -> np.ndarray:
        S_T = prices[:, -1]
        phi = self.option_type.value
        return np.maximum(phi * (S_T - self.K), 0)
    
class AsianOption(Option):
    """
    Arithmetic Asian Option.
    Payoff depends on the arithmetic mean of the asset price path.
    """
    def payoff(self, prices: np.ndarray) -> np.ndarray:
        # Calculate arithmetic mean across time steps (excluding t=0)
        # prices shape: (N_paths, N_steps + 1)
        average_price = np.mean(prices[:, 1:], axis=1)
        
        phi = self.option_type.value
        # Payoff: max(phi * (Average - K), 0)
        return np.maximum(phi * (average_price - self.K), 0)