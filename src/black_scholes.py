
import math

def _phi(x: float) -> float:
    """Standard normal CDF using erf (no SciPy dependency)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _phi_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def d1_d2(S0, K, r, sigma, T):
    if sigma <= 0 or T <= 0:
        raise ValueError("sigma and T must be positive for Black–Scholes formulas.")
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2

def bs_price(S0, K, r, sigma, T, option="call"):
    d1, d2 = d1_d2(S0, K, r, sigma, T)
    if option == "call":
        return S0 * _phi(d1) - K * math.exp(-r * T) * _phi(d2)
    elif option == "put":
        return K * math.exp(-r * T) * _phi(-d2) - S0 * _phi(-d1)
    else:
        raise ValueError("option must be 'call' or 'put'")

def bs_delta(S0, K, r, sigma, T, option="call"):
    d1, _ = d1_d2(S0, K, r, sigma, T)
    if option == "call":
        return _phi(d1)
    elif option == "put":
        return _phi(d1) - 1.0
    else:
        raise ValueError("option must be 'call' or 'put'")

def bs_gamma(S0, K, r, sigma, T):
    d1, _ = d1_d2(S0, K, r, sigma, T)
    return _phi_pdf(d1) / (S0 * sigma * math.sqrt(T))

def bs_vega(S0, K, r, sigma, T):
    d1, _ = d1_d2(S0, K, r, sigma, T)
    return S0 * _phi_pdf(d1) * math.sqrt(T)
