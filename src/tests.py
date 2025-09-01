
from mc_pricer import MonteCarloPricer
from black_scholes import bs_price, bs_delta
import math

def test_price_close():
    S0, K, r, sigma, T = 100.0, 100.0, 0.01, 0.2, 1.0
    pr = MonteCarloPricer(S0,K,r,sigma,T, antithetic=True, control_variate=True, seed=7)
    res = pr.price(500_000, option="call")
    target = bs_price(S0,K,r,sigma,T, option="call")
    assert abs(res.price - target) < 0.02

def test_delta_close():
    S0, K, r, sigma, T = 100.0, 100.0, 0.01, 0.2, 1.0
    pr = MonteCarloPricer(S0,K,r,sigma,T, antithetic=True, control_variate=False, seed=7)
    delta_mc = pr.delta_pathwise(500_000, option="call")
    delta_bs = bs_delta(S0,K,r,sigma,T, option="call")
    assert abs(delta_mc - delta_bs) < 0.01

if __name__ == "__main__":
    test_price_close()
    test_delta_close()
    print("All tests passed.")
