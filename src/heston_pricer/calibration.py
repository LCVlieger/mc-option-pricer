import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import List, Dict
from .analytics import HestonAnalyticalPricer

@dataclass
class MarketOption:
    strike: float
    maturity: float
    market_price: float
    option_type: str = "CALL" 

class HestonCalibrator:
    def __init__(self, S0: float, r: float, q: float = 0.0): # <--- Add q here
        self.S0 = S0
        self.r = r
        self.q = q  # <--- Store it

    def calibrate(self, options: List[MarketOption], init_guess: List[float] = None) -> Dict:
        """
        Calibrates Heston parameters [kappa, theta, xi, rho, v0].
        """
        # Default Guess: [kappa, theta, xi, rho, v0]
        if init_guess is None:
            x0 = [2.0, 0.04, 0.3, -0.5, 0.04]
        else:
            x0 = init_guess

        # Constraints for [kappa, theta, xi, rho, v0]
        bounds = [
            (0.1, 10.0),   # kappa: Mean reversion speed > 0
            (0.001, 0.5),  # theta: Long run variance
            (0.01, 2.0),   # xi: Vol of Vol
            (-0.99, 0.99), # rho: Correlation
            (0.001, 0.5)   # v0: Initial Variance
        ]

        # Objective Function (Sum of Squared Errors)
        def objective(params):
            kappa, theta, xi, rho, v0 = params
            sse = 0.0
            for opt in options:
                model_price = HestonAnalyticalPricer.price_european_call(
                    self.S0, opt.strike, opt.maturity, self.r, self.q, # <--- Pass self.q
                    kappa, theta, xi, rho, v0
                )
                sse += (model_price - opt.market_price) ** 2
            return sse
        
        print(f"Starting Calibration on {len(options)} instruments...")
        
        # L-BFGS-B is excellent for bound-constrained optimization
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, tol=1e-8)

        print(f"Optimization Finished: {result.message}")
        print(f"Final SSE: {result.fun:.6f}")
        
        return {
            "kappa": result.x[0],
            "theta": result.x[1],
            "xi": result.x[2],
            "rho": result.x[3],
            "v0": result.x[4],
            "success": result.success
        }