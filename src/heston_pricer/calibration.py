import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import List, Dict
from .analytics import HestonAnalyticalPricer
from .market import MarketEnvironment
from .models.process import HestonProcess

@dataclass
class MarketOption:
    strike: float
    maturity: float
    market_price: float
    option_type: str = "CALL" 

class HestonCalibrator:
    def __init__(self, S0: float, r: float, q: float = 0.0):
        self.S0 = S0
        self.r = r
        self.q = q

    def calibrate(self, options: List[MarketOption], init_guess: List[float] = None) -> Dict:
        """
        Calibrates using L-BFGS-B with WEIGHTED SSE to force gradient movement.
        """
        # Default Guess: [kappa, theta, xi, rho, v0]
        if init_guess is None:
            x0 = [2.0, 0.04, 0.5, -0.7, 0.04]
        else:
            x0 = init_guess

        # Bounds
        bounds = [
            (0.5, 7.5),    # kappa: LOWER BOUND 0.5. Forces the model to actually mean-revert.
                            # This prevents the "theta explosion" hack.
            
            (0.001, 2.0),   # theta: UPPER BOUND 2.0 (140% Vol). 
                            # Safe for TSLA, GME, Crypto, etc.
            
            (0.01, 3),    # xi: Vol of Vol. 5.0 is massive (fat tails).
            
            (-0.999, 0),# rho: Allow full range. 
                            # Equities are usually < 0, but FX/Commodities can be > 0.
            
            (0.001, 2.0)    # v0: Initial variance. Same as theta.
        ]

        # --- 1. THE FIX: Weighted Objective Function ---
        def objective(params):
            kappa, theta, xi, rho, v0 = params
            
            # Penalties for impossible math (Soft Constraints)
            if 2 * kappa * theta < xi**2: 
                # Feller violation penalty (optional, but helps stability)
                penalty = 1e6 * (abs(2 * kappa * theta < xi**2)**2)
            else:
                penalty = 0.0

            sse = 0.0
            
            for opt in options:
                model_price = HestonAnalyticalPricer.price_european_call(
                    self.S0, opt.strike, opt.maturity, self.r, self.q,
                    kappa, theta, xi, rho, v0
                )
                
                # CRITICAL: Weight by 1/MarketPrice
                # A $0.10 error on a $1.00 option is now a HUGE error (10%),
                # forcing the optimizer to fit the skew (xi/rho).
                weight = 1.0 / np.sqrt(opt.market_price + 1e-5) # Avoid div/0
                
                # Relative Squared Error
                error = (model_price - opt.market_price) * weight
                sse += error ** 2
            
            return sse + penalty

        # --- 2. Debug Callback ---
        def callback(xk):
            # This prints every iteration so you KNOW it's moving
            print(f"   Iter: k={xk[0]:.2f}, th={xk[1]:.3f}, xi={xk[2]:.2f}, rho={xk[3]:.2f}, v0={xk[4]:.3f}")

        print(f"Starting L-BFGS-B Calibration on {len(options)} instruments...")
        
        # We increase 'eps' (step size) so it takes bigger finite-difference steps
        # This helps it 'see' the gradient on flat surfaces.
        result = minimize(
            objective, 
            x0, 
            method='L-BFGS-B', 
            bounds=bounds, 
            callback=callback,
            tol=1e-6,
            options={'ftol': 1e-9, 'eps': 1e-4, 'maxiter': 100}
        )

        print(f"Optimization Finished: {result.message}")
        print(f"Final Weighted SSE: {result.fun:.6f}")
        
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
        self.max_T = max(opt.maturity for opt in options)
        dt = self.max_T / self.n_steps
        
        self.time_indices = []
        for opt in options:
            idx = int(round(opt.maturity / dt))
            idx = min(idx, self.n_steps)
            self.time_indices.append(idx)

        # Generate noise ONCE for deterministic gradients
        if self.z_noise is None:
            print(f"   [System] Generating Global CRN Noise ({self.n_paths} paths)...")
            np.random.seed(42) 
            self.z_noise = np.random.normal(0, 1, (2, self.n_steps, self.n_paths))

    def get_prices(self, params: List[float]) -> List[float]:
        kappa, theta, xi, rho, v0 = params
        self.process = HestonProcess(self.base_env)
        self.process.market.kappa = kappa
        self.process.market.theta = theta
        self.process.market.xi = xi
        self.process.market.rho = rho
        self.process.market.v0 = v0

        paths = self.process.generate_paths(
            T=self.max_T,
            n_paths=self.n_paths,
            n_steps=self.n_steps,
            noise=self.z_noise
        )

        prices = []
        for i, opt in enumerate(self.options_cache):
            idx = self.time_indices[i]
            S_T = paths[:, idx]
            
            if opt.option_type == "CALL":
                payoff = np.maximum(S_T - opt.strike, 0.0)
            else:
                payoff = np.maximum(opt.strike - S_T, 0.0)
            
            df = np.exp(-self.process.market.r * opt.maturity)
            prices.append(np.mean(payoff) * df)
        return prices

    def objective(self, params):
        # 1. Calculate Feller Penalty FIRST to save compute if violation is massive
        #    (Though we still compute prices to get the gradient direction)
        kappa, theta, xi, rho, v0 = params
        
        # Feller Condition: 2 * kappa * theta >= xi^2
        # Violation: xi^2 - 2 * kappa * theta > 0
        feller_resid = (xi**2) - (2 * kappa * theta)
        
        if feller_resid > 0:
            # Soft Constraint: Quadratic penalty creates a smooth gradient back to safety
            # Scaling by 1000.0 makes it significant but differentiable
            penalty = 1000.0 * (feller_resid ** 2)
        else:
            penalty = 0.0

        # 2. Calculate Pricing Error
        model_prices = self.get_prices(params)
        sse = 0.0
        for i, price in enumerate(model_prices):
            mkt_price = self.options_cache[i].market_price
            # WEIGHTED SSE
            weight = 1.0 / np.sqrt(mkt_price + 1e-5)
            sse += ((price - mkt_price) * weight) ** 2
            
        return sse + penalty

    def calibrate(self, options: List[MarketOption], init_guess: List[float] = None) -> Dict:
        self.options_cache = options
        self._precompute_batch_grid(options)

        if init_guess is None:
            x0 = [2.0, 0.04, 0.5, -0.7, 0.04]
        else:
            x0 = init_guess

        bounds = [
            (0.5, 7.5),    # kappa
            (0.001, 2.0),   # theta
            (0.01, 3),    # xi
            (-0.999, 0),    # rho
            (0.001, 2.0)    # v0
        ]

        def callback(xk):
             print(f"   [MC] Iter: k={xk[0]:.2f}, th={xk[1]:.3f}, xi={xk[2]:.2f}, rho={xk[3]:.2f}, v0={xk[4]:.3f}")

        print(f"   [System] Starting MC Calibration (L-BFGS-B)...")
        
        result = minimize(
            self.objective, 
            x0, 
            method='L-BFGS-B', 
            bounds=bounds, 
            callback=callback,
            tol=1e-5,
            options={'ftol': 1e-5, 'eps': 1e-5} 
        )

        return {
            "kappa": result.x[0], "theta": result.x[1], "xi": result.x[2],
            "rho": result.x[3], "v0": result.x[4], "success": result.success, "sse": result.fun
        }