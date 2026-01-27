import numpy as np
from scipy.optimize import minimize, brentq
from scipy.stats import norm
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

# The analytical calibrator for the Heston model.  
class HestonCalibrator:
    def __init__(self, S0: float, r: float, q: float = 0.0):
        self.S0 = S0
        self.r = r
        self.q = q

    def calibrate(self, options: List[MarketOption], init_guess: List[float] = None) -> Dict:
        x0 = init_guess if init_guess else [3.0, 0.05, 0.3, -0.7, 0.04]
        bounds =  [(0.5, 10.0), (0.001, 2.0), (0.01, 5.0), (-0.999, 0.0), (0.001, 2.0)]#[(0.5, 3.5), (0.05, 0.5), (0.01, 0.9), (-0.95, -0.5), (0.005, 0.5)] # 

        def objective(params):
            kappa, theta, xi, rho, v0 = params
            # Soft constraint for feller condition. This soft constraint is added because
            # we use a full truncation scheme which can get biased when the variance process hits zero. 
            penalty = 0.0
            if 2 * kappa * theta < xi**2: 
                penalty = 0e0 * (abs(2 * kappa * theta - xi**2)**2)

            sse = 0.0
            for opt in options:
                # Calibration assumes EU call options 
                # Can be extended later to puts and other liquid options. 
                model_price = HestonAnalyticalPricer.price_european_call(
                    self.S0, opt.strike, opt.maturity, self.r, self.q,
                    kappa, theta, xi, rho, v0
                )
                weight = 1.0 / np.sqrt(opt.market_price + 1e-5)
                sse += ((model_price - opt.market_price) * weight) ** 2
            
            return sse + penalty

        # print the optimization steps during optimization. 
        def callback(xk):
            print(f"   [Analytical] k={xk[0]:.2f}, theta={xk[1]:.3f}, xi={xk[2]:.2f}, rho={xk[3]:.2f}, v0={xk[4]:.3f}", flush=True)

        # We use the L-BFGS-B optimizer because it is gradient-based and can impose
        # bounds on the parameters (box constraints). 
        # See: (https://docs.scipy.org/doc/scipy-1.16.2/reference/generated/scipy.optimize.minimize.html)
        result = minimize(
            objective, x0, method='L-BFGS-B', bounds=bounds,
            callback=callback,
            tol=1-7, options={'ftol': 1e-7, 'eps': 1e-7, 'maxiter': 100}
        )
        # Prices the options using the optimized parameters & calculates the "implied volatility sse"
        # which is given by (1/#options)*sum__{i \in options} (iv_market[i] - iv_model[i])^2. 
        # this is a good measure of fit accounting for heteroskedastic option prices 
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
            "success": result.success, "sse": result.fun, 
            "rmse_iv": np.sqrt(sse_iv / count) if count > 0 else 0.0
        }

# The Monte Carlo calibrator
class HestonCalibratorMC:
    def __init__(self, S0: float, r: float, q: float = 0.0, n_paths: int = 30000, n_steps: int = 100):
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
            T=self.max_T, n_paths=self.n_paths, n_steps=self.n_steps, noise=self.z_noise
        )

        prices = []
        for i, opt in enumerate(self.options_cache):
            idx = self.time_indices[i]
            # Calibration assumes EU call options (same as in the analytical calibration). 
            # Can be extended to puts and other liquid options. 
            payoff = np.maximum(paths[:, idx] - opt.strike, 0.0) 
            prices.append(np.mean(payoff) * np.exp(-self.process.market.r * opt.maturity))
        return prices

    def objective(self, params):
        kappa, theta, xi, rho, v0 = params
        if (xi**2) - (2 * kappa * theta) > 0: 
            penalty = 0e0 * ((xi**2) - (2 * kappa * theta)) ** 2
        else:
            penalty = 0.0

        model_prices = self.get_prices(params)
        sse = 0.0
        for i, price in enumerate(model_prices):
            weight = 1.0 / np.sqrt(self.options_cache[i].market_price + 1e-5)
            sse += ((price - self.options_cache[i].market_price) * weight) ** 2
        return sse + penalty

    def calibrate(self, options: List[MarketOption], init_guess: List[float] = None) -> Dict:
        # precompute paths and print optimization steps
        self.options_cache = options
        self._precompute_batch_grid(options)
        x0 = init_guess if init_guess else [3.0, 0.05, 0.3, -0.7, 0.04]
        bounds = [(0.5, 10.0), (0.001, 2.0), (0.01, 5.0), (-0.999, 0.0), (0.001, 2.0)]#[(0.5, 3.5), (0.05, 0.5), (0.01, 0.9), (-0.95, -0.5), (0.005, 0.5)]
        def callback(xk):
             print(f"   [MonteCarlo] k={xk[0]:.2f}, theta={xk[1]:.3f}, xi={xk[2]:.2f}, rho={xk[3]:.2f}, v0={xk[4]:.3f}", flush=True)
        
        #again, a L-BFGS-B optimizer to handle gradients and parameter bounds / box constraints. 
        #Use a slightly lower tolerance here because of the Monte Carlo noise.    
        result = minimize(
            self.objective, x0, method='L-BFGS-B', bounds=bounds, 
            callback=callback, tol=1e-6, options={'ftol': 1e-7, 'eps': 1e-7, 'maxiter': 200}
        )

        final_mc_prices = self.get_prices(result.x)
        sse_iv, count = 0.0, 0
        for i, price in enumerate(final_mc_prices):
            opt = options[i]
            iv_mkt = implied_volatility(opt.market_price, self.base_env.S0, opt.strike, opt.maturity, self.base_env.r, self.base_env.q)
            iv_model = implied_volatility(price, self.base_env.S0, opt.strike, opt.maturity, self.base_env.r, self.base_env.q)
            if iv_mkt > 0 and iv_model > 0:
                sse_iv += (iv_model - iv_mkt) ** 2
                count += 1
        
        # Ensure return structure contains all necessary metrics
        return {
            "kappa": result.x[0], "theta": result.x[1], "xi": result.x[2],
            "rho": result.x[3], "v0": result.x[4], 
            "success": result.success, 
            "fun": result.fun, # SSE
            "rmse_iv": np.sqrt(sse_iv / count) if count > 0 else 0.0
        }
    
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