
import time
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

from black_scholes import bs_price, bs_delta, bs_gamma, bs_vega
from variance_reduction import antithetic_pairs

Array = np.ndarray

@dataclass
class MCResult:
    price: float
    stderr: float
    ci95_low: float
    ci95_high: float
    elapsed_sec: float
    n_paths: int
    used_antithetic: bool
    used_control_variate: bool

class MonteCarloPricer:
    """
    Vectorized, seedable Monte Carlo pricer for European calls/puts under GBM.
    Supports antithetic variates and a control variate using discounted S_T.
    Greeks: pathwise Delta, CRN-bump Gamma & Vega.
    """
    def __init__(self, S0: float, K: float, r: float, sigma: float, T: float,
                 antithetic: bool = False, control_variate: bool = False, seed: Optional[int] = None):
        self.S0 = float(S0)
        self.K = float(K)
        self.r = float(r)
        self.sigma = float(sigma)
        self.T = float(T)
        self.antithetic = antithetic
        self.control_variate = control_variate
        self.rng = np.random.default_rng(seed)

        if self.S0 <= 0 or self.K <= 0 or self.sigma <= 0 or self.T <= 0:
            raise ValueError("S0, K, sigma, T must be positive.")

    def _simulate_terminal(self, n_paths: int, Z: Optional[Array] = None) -> Tuple[Array, Array]:
        """
        Simulate terminal prices S_T and also return the standard normal draws used (for CRN).
        Uses antithetic variates if enabled.
        """
        if Z is None:
            n = n_paths if not self.antithetic else (n_paths + 1) // 2  # base draws
            Z = self.rng.standard_normal(n, dtype=np.float64)
        if self.antithetic:
            Z = antithetic_pairs(Z)
            if len(Z) > n_paths:
                Z = Z[:n_paths]
        else:
            if len(Z) != n_paths:
                # If caller passed custom Z, respect its length; otherwise draw n_paths.
                if Z is None or len(Z) != n_paths:
                    Z = self.rng.standard_normal(n_paths, dtype=np.float64)

        drift = (self.r - 0.5 * self.sigma * self.sigma) * self.T
        diffusion = self.sigma * math.sqrt(self.T) * Z
        ST = self.S0 * np.exp(drift + diffusion)
        return ST, Z

    def _payoff(self, ST: Array, option: str) -> Array:
        if option == "call":
            return np.maximum(ST - self.K, 0.0)
        elif option == "put":
            return np.maximum(self.K - ST, 0.0)
        else:
            raise ValueError("option must be 'call' or 'put'")

    def _discount(self, x: Array) -> Array:
        return np.exp(-self.r * self.T) * x

    def price(self, n_paths: int, option: str = "call") -> MCResult:
        t0 = time.perf_counter()

        ST, Z = self._simulate_terminal(n_paths)
        payoff = self._payoff(ST, option)
        disc_payoff = self._discount(payoff)

        if self.control_variate:
            # Use X = exp(-rT) * S_T as control with known expectation E[X] = S0.
            X = self._discount(ST)
            X_mean = X.mean()
            Y = disc_payoff
            # beta = Cov(Y, X) / Var(X)
            cov = np.cov(Y, X, ddof=1)
            beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0.0
            Y_cv = Y - beta * (X - self.S0)
            est = Y_cv.mean()
            sd = Y_cv.std(ddof=1)
        else:
            est = disc_payoff.mean()
            sd = disc_payoff.std(ddof=1)

        se = sd / math.sqrt(len(disc_payoff))
        ci95 = 1.96 * se
        t1 = time.perf_counter()

        return MCResult(
            price=float(est),
            stderr=float(se),
            ci95_low=float(est - ci95),
            ci95_high=float(est + ci95),
            elapsed_sec=float(t1 - t0),
            n_paths=len(disc_payoff),
            used_antithetic=self.antithetic,
            used_control_variate=self.control_variate,
        )

    # -------- Greeks --------
    def delta_pathwise(self, n_paths: int, option: str = "call") -> float:
        ST, Z = self._simulate_terminal(n_paths)
        indicator_call = (ST > self.K).astype(np.float64)
        indicator_put = (ST < self.K).astype(np.float64)
        if option == "call":
            dpayoff_dS0 = indicator_call * (ST / self.S0)
        elif option == "put":
            dpayoff_dS0 = -indicator_put * (ST / self.S0)
        else:
            raise ValueError("option must be 'call' or 'put'")
        return float(self._discount(dpayoff_dS0).mean())

    def gamma_bump_crn(self, n_paths: int, option: str = "call", eps: float = 1e-4) -> float:
        # Common random numbers for stable finite difference
        base_n = n_paths if not self.antithetic else (n_paths + 1) // 2
        Z = self.rng.standard_normal(base_n, dtype=np.float64)
        if self.antithetic:
            Z = np.concatenate([Z, -Z])[:n_paths]

        def price_with_S0(S0_new: float) -> float:
            drift = (self.r - 0.5 * self.sigma * self.sigma) * self.T
            diffusion = self.sigma * math.sqrt(self.T) * Z
            ST = S0_new * np.exp(drift + diffusion)
            payoff = self._payoff(ST, option)
            return float(self._discount(payoff).mean())

        S0p = self.S0 * (1.0 + eps)
        S0m = self.S0 * (1.0 - eps)
        C_p = price_with_S0(S0p)
        C_m = price_with_S0(S0m)
        C_0 = price_with_S0(self.S0)
        gamma = (C_p - 2.0 * C_0 + C_m) / (self.S0 * eps)**2
        return float(gamma)

    def vega_bump_crn(self, n_paths: int, option: str = "call", eps: float = 1e-4) -> float:
        base_n = n_paths if not self.antithetic else (n_paths + 1) // 2
        Z = self.rng.standard_normal(base_n, dtype=np.float64)
        if self.antithetic:
            Z = np.concatenate([Z, -Z])[:n_paths]

        def price_with_sigma(sig_new: float) -> float:
            drift = (self.r - 0.5 * sig_new * sig_new) * self.T
            diffusion = sig_new * math.sqrt(self.T) * Z
            ST = self.S0 * np.exp(drift + diffusion)
            payoff = self._payoff(ST, option)
            return float(self._discount(payoff).mean())

        sigp = self.sigma * (1.0 + eps)
        sigm = self.sigma * (1.0 - eps)
        C_p = price_with_sigma(sigp)
        C_m = price_with_sigma(sigm)
        vega = (C_p - C_m) / (2.0 * self.sigma * eps)
        return float(vega)

def _pretty_ci(res: MCResult) -> str:
    return f"{res.price:.6f} ± {1.96*res.stderr:.6f} (95% CI: [{res.ci95_low:.6f}, {res.ci95_high:.6f}])"

def run_cli():
    import argparse

    p = argparse.ArgumentParser(description="Monte Carlo pricer for European options under GBM (Black–Scholes).")
    p.add_argument("--S0", type=float, required=True)
    p.add_argument("--K", type=float, required=True)
    p.add_argument("--r", type=float, required=True)
    p.add_argument("--sigma", type=float, required=True)
    p.add_argument("--T", type=float, required=True)
    p.add_argument("--n_paths", type=int, required=True)
    p.add_argument("--option", type=str, default="call", choices=["call", "put"])
    p.add_argument("--antithetic", action="store_true", help="Enable antithetic variates.")
    p.add_argument("--control-variate", action="store_true", dest="control_variate", help="Enable S_T control variate.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--greeks", action="store_true", help="Compute Delta (pathwise), Gamma & Vega (CRN bumps).")

    args = p.parse_args()

    pricer = MonteCarloPricer(
        S0=args.S0, K=args.K, r=args.r, sigma=args.sigma, T=args.T,
        antithetic=args.antithetic, control_variate=args.control_variate, seed=args.seed
    )
    res = pricer.price(args.n_paths, option=args.option)

    # Closed-form
    bs = bs_price(args.S0, args.K, args.r, args.sigma, args.T, option=args.option)

    print("Inputs:")
    print(f"  S0={args.S0}, K={args.K}, r={args.r}, sigma={args.sigma}, T={args.T}, n_paths={res.n_paths}")
    print(f"  option={args.option}, antithetic={res.used_antithetic}, control_variate={res.used_control_variate}")
    print("\nResults:")
    print(f"  MC price:   {_pretty_ci(res)}")
    print(f"  BS price:   {bs:.6f}")
    print(f"  abs error:  {abs(res.price - bs):.6f}")
    print(f"  runtime:    {res.elapsed_sec:.3f} sec")

    if args.greeks:
        delta_mc = pricer.delta_pathwise(args.n_paths, option=args.option)
        gamma_mc = pricer.gamma_bump_crn(args.n_paths, option=args.option)
        vega_mc = pricer.vega_bump_crn(args.n_paths, option=args.option)

        delta_bs = bs_delta(args.S0, args.K, args.r, args.sigma, args.T, option=args.option)
        gamma_bs = bs_gamma(args.S0, args.K, args.r, args.sigma, args.T)
        vega_bs = bs_vega(args.S0, args.K, args.r, args.sigma, args.T)

        print("\nGreeks (MC vs BS):")
        print(f"  Delta: {delta_mc:.6f}  |  BS: {delta_bs:.6f}  |  diff: {delta_mc - delta_bs:+.6f}")
        print(f"  Gamma: {gamma_mc:.6f}  |  BS: {gamma_bs:.6f}  |  diff: {gamma_mc - gamma_bs:+.6f}")
        print(f"  Vega:  {vega_mc:.6f}  |  BS: {vega_bs:.6f}   |  diff: {vega_mc - vega_bs:+.6f}")

if __name__ == "__main__":
    run_cli()
