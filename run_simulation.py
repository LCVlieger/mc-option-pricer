from src.instruments import EuropeanOption, OptionType
from src.mc_pricer import MonteCarloPricer
from src.analytics import BlackScholesPricer

def main():
    # Parameters
    S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2

    # 1. Analytical Price (The correct value)
    bs_price = BlackScholesPricer.price_european_call(S0, K, T, r, sigma)
    print(f"Black-Scholes Analytical price: {bs_price:.4f}")

    # 2. Monte Carlo Price (The simulation)
    call_option = EuropeanOption(K, T, OptionType.CALL)
    mc_engine = MonteCarloPricer(r, sigma)
    mc_price = mc_engine.price_option(call_option, S0, n_paths=100000)
    print(f"Monte Carlo Simulation price:     {mc_price:.4f}")

    # 3. The Difference (The error)
    diff = abs(bs_price - mc_price)
    print(f"Differences: {diff:.4f}")

if __name__ == "__main__":
    main()