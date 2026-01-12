import pytest
import numpy as np
from quantlib.market import MarketEnvironment
from quantlib.instruments import EuropeanOption, AsianOption, OptionType
from quantlib.models.mc_pricer import MonteCarloPricer
from quantlib.analytics import BlackScholesPricer

@pytest.fixture
def default_market():
    """Standard market environment for all tests."""
    return MarketEnvironment(S0=100, r=0.05, sigma=0.2)

def test_european_call_convergence(default_market):
    """
    Green Test 1: Does Monte Carlo converge to the exact Black-Scholes price?
    Tolerance: < 0.05 (Strict for European).
    """
    # 1. Setup
    T, K = 1.0, 100
    option = EuropeanOption(K=K, T=T, option_type=OptionType.CALL)
    pricer = MonteCarloPricer(default_market)
    
    # 2. Execution (High paths for precision)
    # The first run might be slower due to JIT, but accuracy is unaffected.
    mc_price = pricer.price_option(option, n_paths=500_000)
    bs_price = BlackScholesPricer.price_european_call(
        default_market.S0, K, T, default_market.r, default_market.sigma
    )
    
    # 3. Validation
    error = abs(mc_price - bs_price)
    print(f"\nEuropean Error: {error:.4f}")
    assert error < 0.05, f"MC Price {mc_price} deviating from Black-Scholes {bs_price}"

def test_asian_call_approximation(default_market):
    """
    Green Test 2: Does Monte Carlo align with the Turnbull-Wakeman Approximation?
    Tolerance: < 0.20 (Wider tolerance due to approximation bias).
    """
    # 1. Setup
    T, K = 1.0, 100
    option = AsianOption(K=K, T=T, option_type=OptionType.CALL)
    pricer = MonteCarloPricer(default_market)
    
    # 2. Execution
    mc_price = pricer.price_option(option, n_paths=500_000)
    tw_price = BlackScholesPricer.price_asian_arithmetic_approximation(
        default_market.S0, K, T, default_market.r, default_market.sigma
    )
    
    # 3. Validation
    error = abs(mc_price - tw_price)
    print(f"Asian Error: {error:.4f}")
    assert error < 0.20, f"Asian MC {mc_price} diverged from TW Approx {tw_price}"

def test_put_call_parity(default_market):
    """
    Green Test 3: Financial Consistency Check.
    Call - Put = S - K * exp(-rT)
    """
    T, K = 1.0, 100
    call = EuropeanOption(K, T, OptionType.CALL)
    put = EuropeanOption(K, T, OptionType.PUT)
    pricer = MonteCarloPricer(default_market)
    
    c_price = pricer.price_option(call, n_paths=100_000)
    p_price = pricer.price_option(put, n_paths=100_000)
    
    # Theoretical Parity
    discounted_k = K * np.exp(-default_market.r * T)
    lhs = c_price - p_price
    rhs = default_market.S0 - discounted_k
    
    diff = abs(lhs - rhs)
    assert diff < 0.15, f"Put-Call Parity violated by {diff:.4f}"