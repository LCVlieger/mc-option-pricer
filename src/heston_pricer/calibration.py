import numpy as np
from scipy.optimize import minimize, brentq
from scipy.stats import norm
from dataclasses import dataclass
from typing import List, Dict, Tuple
from .analytics import HestonAnalyticalPricer
from .market import MarketEnvironment
from .models.process import HestonProcess

@dataclass
class MarketOption:
    strike: float
    maturity: float
    market_price: float
    option_type: str = "CALL" 

# --- HELPER: Implied Volatility ---
def implied_volatility(price: float, S: float, K: float, T: float, r: float, q: float) -> float:
    if price <= 0: return 0.0
    intrinsic = max(S * np.exp(-q*T) - K * np.exp(-r*T), 0)
    if price < intrinsic: return 0.0

    def bs_price(sigma):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        val = (S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
        return val - price
    
    try:
        return brentq(bs_price, 0.001, 5.0)
    except:
        return 0.0

class HestonCalibrator:
    def __init__(self, S0: float, r: float, q: float = 0.0):
        self.S0 = S0
        self.r = r
        self.q = q

    def calibrate(self, options: List[MarketOption], init_guess: List[float] = None) -> Dict:
        if init_guess is None: x0 = [2.0, 0.04, 0.5, -0.7, 0.04]
        else: x0 = init_guess

        # High precision bounds for Analytical
        bounds = [(0.5, 10.0), (0.001, 2.0), (0.01, 5.0), (-0.999, 0.0), (0.001, 2.0)]

        def objective(params):
            kappa, theta, xi, rho, v0 = params
            # Soft Constraint: Feller Condition (2*k*th > xi^2)
            if 2 * kappa * theta < xi**2: 
                penalty = 1e6 * (abs(2 * kappa * theta - xi**2)**2)
            else:
                penalty = 0.0

            sse = 0.0
            for opt in options:
                model_price = HestonAnalyticalPricer.price_european_call(
                    self.S0, opt.strike, opt.maturity, self.r, self.q,
                    kappa, theta, xi, rho, v0
                )
                weight = 1.0 / np.sqrt(opt.market_price + 1e-5)
                sse += ((model_price - opt.market_price) * weight) ** 2
            
            return sse + penalty

        def callback(xk):
            # Added flush=True to force real-time printing
            print(f"   [Ana] Iter: k={xk[0]:.2f}, th={xk[1]:.3f}, xi={xk[2]:.2f}, rho={xk[3]:.2f}, v0={xk[4]:.3f}", flush=True)

        print(f"Starting Analytical Calibration on {len(options)} instruments...")
        
        result = minimize(
            objective, 
            x0, 
            method='L-BFGS-B', 
            bounds=bounds, 
            callback=callback,
            tol=1e-6,
            options={'ftol': 1e-9, 'eps': 1e-5, 'maxiter': 100}
        )

        # RMSE Calculation
        kappa, theta, xi, rho, v0 = result.x
        sse_iv, count = 0.0, 0
        for opt in options:
            model_price = HestonAnalyticalPricer.price_european_call(
                self.S0, opt.strike, opt.maturity, self.r, self.q, kappa, theta, xi, rho, v0
            )
            iv_mkt = implied_volatility(opt.market_price, self.S0, opt.strike, opt.maturity, self.r, self.q)
            iv_model = implied_volatility(model_price, self.S0, opt.strike, opt.maturity, self.r, self.q)
            if iv_mkt > 0 and iv_model > 0:
                sse_iv += (iv_model - iv_mkt) ** 2
                count += 1
        
        return {
            "kappa": kappa, "theta": theta, "xi": xi, "rho": rho, "v0": v0,
            "success": result.success, "sse": result.fun, "rmse_iv": np.sqrt(sse_iv / count) if count > 0 else 0.0
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
        self.time_indices = [min(int(round(opt.maturity / dt)), self.n_steps) for opt in options]

        if self.z_noise is None:
            print(f"   [System] Generating Global CRN Noise ({self.n_paths} paths)...")
            np.random.seed(42) 
            self.z_noise = np.random.normal(0, 1, (2, self.n_steps, self.n_paths))

    def get_prices(self, params: List[float]) -> List[float]:
        kappa, theta, xi, rho, v0 = params
        
        # --- FIXED: Removed 'scheme' argument here ---
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
            payoff = np.maximum(S_T - opt.strike, 0.0) if opt.option_type == "CALL" else np.maximum(opt.strike - S_T, 0.0)
            prices.append(np.mean(payoff) * np.exp(-self.process.market.r * opt.maturity))
        return prices

    def objective(self, params):
        kappa, theta, xi, rho, v0 = params
        # Feller Constraint
        if (xi**2) - (2 * kappa * theta) > 0: 
            penalty = 1000.0 * ((xi**2) - (2 * kappa * theta)) ** 2
        else:
            penalty = 0.0

        model_prices = self.get_prices(params)
        sse = 0.0
        for i, price in enumerate(model_prices):
            weight = 1.0 / np.sqrt(self.options_cache[i].market_price + 1e-5)
            sse += ((price - self.options_cache[i].market_price) * weight) ** 2
        return sse + penalty

    def calibrate(self, options: List[MarketOption], init_guess: List[float] = None) -> Dict:
        self.options_cache = options
        self._precompute_batch_grid(options)
        if init_guess is None: x0 = [2.0, 0.04, 0.5, -0.7, 0.04]
        else: x0 = init_guess

        bounds = [(0.5, 10.0), (0.001, 2.0), (0.01, 5.0), (-0.999, 0.0), (0.001, 2.0)]
        
        def callback(xk):
             # Flush output to ensure real-time logging during optimization
             print(f"   [MC] Iter: k={xk[0]:.2f}, th={xk[1]:.3f}, xi={xk[2]:.2f}, rho={xk[3]:.2f}, v0={xk[4]:.3f}", flush=True)

        print(f"   [System] Starting MC Calibration (L-BFGS-B)...")
        result = minimize(self.objective, x0, method='L-BFGS-B', bounds=bounds, callback=callback, tol=1e-5)

        final_mc_prices = self.get_prices(result.x)
        sse_iv, count = 0.0, 0
        for i, price in enumerate(final_mc_prices):
            opt = options[i]
            iv_mkt = implied_volatility(opt.market_price, self.base_env.S0, opt.strike, opt.maturity, self.base_env.r, self.base_env.q)
            iv_model = implied_volatility(price, self.base_env.S0, opt.strike, opt.maturity, self.base_env.r, self.base_env.q)
            if iv_mkt > 0 and iv_model > 0:
                sse_iv += (iv_model - iv_mkt) ** 2
                count += 1
        
        return {
            "kappa": result.x[0], "theta": result.x[1], "xi": result.x[2],
            "rho": result.x[3], "v0": result.x[4], 
            "success": result.success, "rmse_iv": np.sqrt(sse_iv / count) if count > 0 else 0.0
        }