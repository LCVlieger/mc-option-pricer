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
    
class BarrierType(Enum):
    DOWN_AND_OUT = 1
    DOWN_AND_IN = 2
    UP_AND_OUT = 3
    UP_AND_IN = 4

class BarrierOption(Option):
    def __init__(self, K: float, T: float, barrier: float, barrier_type: BarrierType, option_type: OptionType):
        super().__init__(K, T, option_type)
        self.barrier = barrier
        self.barrier_type = barrier_type

    def payoff(self, prices: np.ndarray) -> np.ndarray:
        """
        prices shape: (n_paths, n_steps + 1)
        """
        S_T = prices[:, -1]
        
        # 1. Calculate the 'Intrinsic' Payoff (as if it were European)
        phi = self.option_type.value
        intrinsic_payoff = np.maximum(phi * (S_T - self.K), 0)
        
        # 2. Determine Barrier Breaches
        # We need the min or max of the path for each simulation
        path_min = np.min(prices, axis=1)
        path_max = np.max(prices, axis=1)
        
        # 3. Apply Barrier Logic (Vectorized)
        if self.barrier_type == BarrierType.DOWN_AND_OUT:
            # If min goes below barrier, option dies (payoff = 0)
            # We use a boolean mask: active where min > barrier
            active_mask = path_min > self.barrier
            return intrinsic_payoff * active_mask
            
        elif self.barrier_type == BarrierType.DOWN_AND_IN:
            # Only active if min goes below barrier
            active_mask = path_min <= self.barrier
            return intrinsic_payoff * active_mask
            
        elif self.barrier_type == BarrierType.UP_AND_OUT:
            # If max goes above barrier, option dies
            active_mask = path_max < self.barrier
            return intrinsic_payoff * active_mask
            
        elif self.barrier_type == BarrierType.UP_AND_IN:
            # Only active if max goes above barrier
            active_mask = path_max >= self.barrier
            return intrinsic_payoff * active_mask
            
        return intrinsic_payoff