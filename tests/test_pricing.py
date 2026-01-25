import pytest
import numpy as np
from heston_pricer.market import MarketEnvironment
from heston_pricer.instruments import EuropeanOption, AsianOption, OptionType
from heston_pricer.models.process import BlackScholesProcess
from heston_pricer.models.mc_pricer import MonteCarloPricer
from heston_pricer.analytics import BlackScholesPricer

@pytest.fixture
def default_market():
    return MarketEnvironment(S0=100, r=0.05, q=0.0, sigma=0.2)

def test_european_call_convergence(default_market):
    """
    Verify Monte Carlo black scholes converges to analytical Black-Scholes price.
    """
    T, K = 1.0, 100
    option = EuropeanOption(K=K, T=T, option_type=OptionType.CALL)
    
    process = BlackScholesProcess(default_market)
    pricer = MonteCarloPricer(process)
    
    result = pricer.price(option, n_paths=200_000, n_steps=100)
    
    bs_price = BlackScholesPricer.price_european_call(
        default_market.S0, K, T, default_market.r, default_market.sigma
    )
    
    assert abs(result.price - bs_price) < 0.05

def test_asian_call_approximation(default_market):
    """
    Verify Monte Carlo asian option price aligns with Turnbull-Wakeman approximation.
    """
    T, K = 1.0, 100
    option = AsianOption(K=K, T=T, option_type=OptionType.CALL)
    
    process = BlackScholesProcess(default_market)
    pricer = MonteCarloPricer(process)
    
    result = pricer.price(option, n_paths=200_000, n_steps=100)
    
    tw_price = BlackScholesPricer.price_asian_arithmetic_approximation(
        default_market.S0, K, T, default_market.r, default_market.sigma
    )
    
    # Approximations are not exact, so we allow wider tolerance
    assert abs(result.price - tw_price) < 0.20

def test_put_call_parity(default_market):
    """
    Checks Put-Call Parity: C - P = S - K * exp(-rT)
    """
    T, K = 1.0, 100
    call = EuropeanOption(K, T, OptionType.CALL)
    put = EuropeanOption(K, T, OptionType.PUT)
    
    process = BlackScholesProcess(default_market)
    pricer = MonteCarloPricer(process)
    
    # High path count for stability
    c_price = pricer.price(call, n_paths=100_000, n_steps=100).price
    p_price = pricer.price(put, n_paths=100_000, n_steps=100).price
    
    discounted_k = K * np.exp(-default_market.r * T)
    lhs = c_price - p_price
    rhs = default_market.S0 - discounted_k
    
    assert abs(lhs - rhs) < 0.15