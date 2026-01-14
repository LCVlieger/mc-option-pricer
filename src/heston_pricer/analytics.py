import numpy as np
from scipy.stats import norm
import scipy.integrate as integrate

class BlackScholesPricer:
    @staticmethod
    def price_european_call(S0, K, T, r, sigma):
        """
        Analytical price for European Call.
        """
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return price
    
    @staticmethod
    def price_asian_arithmetic_approximation(S0, K, T, r, sigma):
        """
        Calculates the approximate price of an Arithmetic Asian Call using 
        Moment Matching (Turnbull-Wakeman / Levy). Reference: 
        Hull, Options Futures & Derivatives 11th edition, '26.13: Asian Options, p.626'.
        """
        # 1. Calculate the first two moments of the continuous average
        # M1 = First Moment: Expected Average
        # M2 = Second Moment: Expected Squared Average
        
        # Avoid division by zero for small r
        if abs(r) < 1e-6:
            M1 = S0
            M2 = S0**2 * (2 * np.exp(sigma**2 * T) - 1) # Simplified for r=0
        else:
            M1 = (np.exp(r * T) - 1) / (r * T) * S0
            
            # M2 calculation
            term1 = 2 * np.exp((2 * r + sigma**2) * T) / ((r + sigma**2) * (2 * r + sigma**2) * T**2)
            term2 = (2 / (r * T**2)) * (
                1 / (2 * r + sigma**2) - 
                np.exp(r * T) / (r + sigma**2)
            )
            M2 = S0**2 * (term1 + term2)

        # 2. Map moments to Log-Normal parameters
        # We assume the Average ~ Lognormal(m, v^2)
        # Variance of the log-average: v^2 = ln(M2 / M1^2)
        if M2 <= M1**2: # Safety check
            return 0.0
            
        sigma_eff = np.sqrt(np.log(M2 / M1**2) / T)
        r_eff = np.log(M1 / S0) / T # Effective drift to match mean

        # 3. Use Black's Model (generalized Black-Scholes)
        # We effectively price an option on a future with price F0 = M1
        # Discount factor is still exp(-rT) because payoff is at T
        
        d1 = (np.log(M1 / K) + 0.5 * sigma_eff**2 * T) / (sigma_eff * np.sqrt(T))
        d2 = d1 - sigma_eff * np.sqrt(T)
        
        # Price = exp(-rT) * [F0 * N(d1) - K * N(d2)]
        # Since M1 is the expected average (F0), we use:
        price = np.exp(-r * T) * (M1 * norm.cdf(d1) - K * norm.cdf(d2))
        
        return price
    
class HestonAnalyticalPricer:
    """
    Semi-Analytical Heston Price using Fourier Integration (Gil-Pelaez).
    Now supports Dividend Yield (q).
    """
    @staticmethod
    def price_european_call(S0, K, T, r, q, kappa, theta, xi, rho, v0): # <--- Added q
        # 1. Characteristic Function (Updated for q)
        def heston_char_func(u):
            d = np.sqrt((rho * xi * u * 1j - kappa)**2 + xi**2 * (u * 1j + u**2))
            g = (kappa - rho * xi * u * 1j - d) / (kappa - rho * xi * u * 1j + d)
            
            C = (1/xi**2) * (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T)) * \
                (kappa - rho * xi * u * 1j - d)
                
            D = (kappa * theta / xi**2) * \
                ((kappa - rho * xi * u * 1j - d) * T - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g)))
            
            # <--- UPDATED: Drift is now (r - q)
            # This represents the log-forward price: ln( S0 * exp((r-q)T) )
            drift_term = 1j * u * np.log(S0 * np.exp((r - q) * T))
            
            return np.exp(C * v0 + D + drift_term)

        # 2. Integration (Gil-Pelaez)
        limit = 200 
        
        def integrand_p1(u):
            # <--- UPDATED: Denominator uses (r - q)
            # This normalizes by the Forward Price
            num = np.exp(-1j * u * np.log(K)) * heston_char_func(u - 1j)
            denom = 1j * u * S0 * np.exp((r - q) * T) 
            return np.real(num / denom)
            
        def integrand_p2(u):
            num = np.exp(-1j * u * np.log(K)) * heston_char_func(u)
            denom = 1j * u
            return np.real(num / denom)
            
        P1 = 0.5 + (1/np.pi) * integrate.quad(integrand_p1, 0, limit)[0]
        P2 = 0.5 + (1/np.pi) * integrate.quad(integrand_p2, 0, limit)[0]
        
        # <--- UPDATED: Final Pricing Formula includes exp(-qT)
        # Call = S0 * e^(-qT) * P1 - K * e^(-rT) * P2
        return S0 * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2