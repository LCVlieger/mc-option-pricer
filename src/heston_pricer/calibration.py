import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass
import time
from typing import List, Dict
from .analytics import HestonAnalyticalPricer
from .market import MarketEnvironment
from .instruments import OptionType, EuropeanOption
from .models.process import HestonProcess
from .models.mc_pricer import MonteCarloPricer

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

class HestonCalibratorMC:
    def __init__(self, S0: float, r: float, q: float = 0.0, 
                 n_paths: int = 30000, n_steps: int = 100):
        self.base_env = MarketEnvironment(S0, r, q) 
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.z_noise = None 
        self.options_cache = []
        self.time_indices = []
        self.max_T = 0.0

    def _precompute_batch_grid(self, options: List[MarketOption]):
        # 1. Setup Grid
        self.max_T = max(opt.maturity for opt in options)
        dt = self.max_T / self.n_steps
        
        self.time_indices = []
        for opt in options:
            idx = int(round(opt.maturity / dt))
            idx = min(idx, self.n_steps) # Clamp
            self.time_indices.append(idx)

        # 2. Generate Global Noise
        # Using a fixed seed ensures the optimizer sees a smooth surface
        print(f"   [System] Generating Global CRN Noise ({self.n_paths} paths, {self.n_steps} steps)...")
        np.random.seed(42) 
        self.z_noise = np.random.normal(0, 1, (2, self.n_steps, self.n_paths))

    def get_prices(self, params: List[float], options: List[MarketOption]) -> List[float]:
        # Initialize if needed
        if self.z_noise is None:
            self._precompute_batch_grid(options)

        # 1. Update Parameters
        kappa, theta, xi, rho, v0 = params
        self.process = HestonProcess(self.base_env)
        self.process.market.kappa = kappa
        self.process.market.theta = theta
        self.process.market.xi = xi
        self.process.market.rho = rho
        self.process.market.v0 = v0

        # 2. Run ONE Batch Simulation
        paths = self.process.generate_paths(
            T=self.max_T,
            n_paths=self.n_paths,
            n_steps=self.n_steps,
            noise=self.z_noise
        )

        # 3. Slice and Price
        prices = []
        for i, opt in enumerate(options):
            idx = self.time_indices[i]
            S_T = paths[:, idx]
            
            # Vectorized Payoff
            if opt.option_type == "CALL":
                payoff = np.maximum(S_T - opt.strike, 0.0)
            else:
                payoff = np.maximum(opt.strike - S_T, 0.0)
            
            # Discount
            df = np.exp(-self.process.market.r * opt.maturity)
            prices.append(np.mean(payoff) * df)
            
        return prices

    def calibrate(self, options: List[MarketOption], init_guess: List[float] = None) -> Dict:
        self.options_cache = options
        self._precompute_batch_grid(options)

        if init_guess is None:
            x0 = [2.0, 0.04, 0.3, -0.5, 0.04]
        else:
            x0 = init_guess

        # Constraints
        bounds = [(0.1, 10.0), (0.001, 0.5), (0.01, 2.0), (-0.99, 0.99), (0.001, 0.5)]

        print(f"   [System] Starting Fast Batch Calibration (Target: Analytical Prices)...")
        self.process = HestonProcess(self.base_env)
        
        # Use L-BFGS-B for speed and bounds handling
        result = minimize(self.objective, x0, method='L-BFGS-B', bounds=bounds, tol=1e-5)

        return {
            "kappa": result.x[0], "theta": result.x[1], "xi": result.x[2],
            "rho": result.x[3], "v0": result.x[4], "success": result.success, "sse": result.fun
        }

    def objective(self, params):
        model_prices = self.get_prices(params, self.options_cache)
        sse = 0.0
        for i, price in enumerate(model_prices):
            sse += (price - self.options_cache[i].market_price) ** 2
        return sse