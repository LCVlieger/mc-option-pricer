import numpy as np
from scipy.stats import norm
import scipy.integrate as integrate

class BlackScholesPricer:
    @staticmethod
    def price_european_call(S0, K, T, r, sigma):
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    @staticmethod
    def price_asian_arithmetic_approximation(S0, K, T, r, sigma):
        """
        Turnbull-Wakeman (1991) approximation for arithmetic Asian options (ref: Hull CH26.13, p.626).
        Validates Monte Carlo priced asian options under GBM paths. 
        Matches the first two moments of the arithmetic average to a Lognormal distribution. 
        """
        # 1. Moments of the arithmetic average 
        if abs(r) < 1e-6:
            M1 = S0
            M2 = S0**2 * (2 * np.exp(sigma**2 * T) - 1)
        else:
            M1 = (np.exp(r * T) - 1) / (r * T) * S0
            
            term1 = 2 * np.exp((2 * r + sigma**2) * T) / ((r + sigma**2) * (2 * r + sigma**2) * T**2)
            term2 = (2 / (r * T**2)) * (
                1 / (2 * r + sigma**2) - 
                np.exp(r * T) / (r + sigma**2)
            )
            M2 = S0**2 * (term1 + term2)

        # 2. Match lognormal
        # v_eff^2 = ln(E[A^2] / E[A]^2)
        if M2 <= M1**2: return 0.0
            
        # 3. Pricing 
        sigma_eff = np.sqrt(np.log(M2 / M1**2) / T)
        d1 = (np.log(M1 / K) + 0.5 * sigma_eff**2 * T) / (sigma_eff * np.sqrt(T))
        d2 = d1 - sigma_eff * np.sqrt(T)
        
        return np.exp(-r * T) * (M1 * norm.cdf(d1) - K * norm.cdf(d2))
    
class HestonAnalyticalPricer:
    """
    Semi-analytic Heston European Pricing. ('The Volatility Surface: A Practitioners Guide, Jim Gatheral, CH2 p.16-18'). 
    """
    @staticmethod
    def price_european_call(S0, K, T, r, q, kappa, theta, xi, rho, v0):
        # Heston Characteristic Function (Heston, 1993)
        def heston_char_func(u):
            d = np.sqrt((rho * xi * u * 1j - kappa)**2 + xi**2 * (u * 1j + u**2))
            g = (kappa - rho * xi * u * 1j - d) / (kappa - rho * xi * u * 1j + d)
            
            C = (1/xi**2) * (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T)) * \
                (kappa - rho * xi * u * 1j - d)
                
            D = (kappa * theta / xi**2) * \
                ((kappa - rho * xi * u * 1j - d) * T - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g)))
            
            # dividend adjustment 
            drift_term = 1j * u * np.log(S0 * np.exp((r - q) * T))
            
            return np.exp(C * v0 + D + drift_term)

        # Integration 
        limit = 950
        
        def integrand_p1(u):
            num = np.exp(-1j * u * np.log(K)) * heston_char_func(u - 1j)
            denom = 1j * u * S0 * np.exp((r - q) * T) 
            return np.real(num / denom)
            
        def integrand_p2(u):
            num = np.exp(-1j * u * np.log(K)) * heston_char_func(u)
            denom = 1j * u
            return np.real(num / denom)
            
        P1 = 0.5 + (1/np.pi) * integrate.quad(integrand_p1, 0, 950, limit=limit)[0]
        P2 = 0.5 + (1/np.pi) * integrate.quad(integrand_p2, 0, 950, limit=limit)[0]
        
        return S0 * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2

    @staticmethod
    def price_european_put(S0, K, T, r, q, kappa, theta, xi, rho, v0):
        """
        Prices a European Put using Put-Call Parity.
        P = C - S*exp(-qT) + K*exp(-rT)
        """
        call_price = HestonAnalyticalPricer.price_european_call(S0, K, T, r, q, kappa, theta, xi, rho, v0)
        return call_price - S0 * np.exp(-q * T) + K * np.exp(-r * T)