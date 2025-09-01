# Monte Carlo Option Pricing

This project implements a **Monte Carlo simulation engine** in Python to price European options and compute Greeks, with validation against the closed-form Black–Scholes model. Designed to explore accuracy, runtime, and variance reduction methods used in quantitative finance.

---

## Overview

**Goal**  
- Price European call and put options using Monte Carlo methods.  
- Validate results against Black–Scholes analytical solutions.  
- Compute sensitivities (Greeks: Delta, Gamma, Vega).  
- Explore efficiency gains with variance reduction.  

**Data / Inputs**  
- Spot price (S0), strike price (K), risk-free rate (r), volatility (σ), time to maturity (T).  
- Configurable number of simulation paths (10k – 1M).  

**Methods**  
- Monte Carlo with configurable seeds.  
- Variance Reduction:  
  - Antithetic variates.  
  - Control variates.  
- Confidence intervals to quantify simulation uncertainty.  
- Runtime benchmarks.  

---
