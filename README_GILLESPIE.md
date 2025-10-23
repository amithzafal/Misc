# Gillespie Stochastic Simulation - Conversion Guide

## Overview

This repository contains the conversion of a deterministic ODE model into a stochastic Gillespie simulation for a gene regulatory network with two species: M and C.

## Original Deterministic Model (ODEs)

The original model is defined by:

```
dM/dt = mu_M + (alpha/Gamma1)*Hc(C) + (delta/Gamma1)*H_act(M) - M
dC/dt = mu_C + (beta/Gamma1)*H_rep(M) - (Gamma2/Gamma1)*C
```

where:
- `H_act(M)`: Hill function for M self-activation
- `H_rep(M)`: Hill function for M repressing C
- `Hc(C)`: Hill function for C activating M

## Gillespie Conversion

### Reaction Network

The ODEs are decomposed into 7 elementary reactions:

| Reaction | Description | Stoichiometry | Propensity Formula |
|----------|-------------|---------------|-------------------|
| **R1** | Basal production of M | ∅ → M | `a1 = mu_M` |
| **R2** | C-activated production of M | ∅ → M | `a2 = (alpha/Gamma1) * Hc(C)` |
| **R3** | Self-activated production of M | ∅ → M | `a3 = (delta/Gamma1) * H_act(M)` |
| **R4** | Degradation of M | M → ∅ | `a4 = 1.0 * M` |
| **R5** | Basal production of C | ∅ → C | `a5 = mu_C` |
| **R6** | M-repressed production of C | ∅ → C | `a6 = (beta/Gamma1) * H_rep(M)` |
| **R7** | Degradation of C | C → ∅ | `a7 = (Gamma2/Gamma1) * C` |

### Propensity Explanation

Each propensity `ai` represents the **rate** at which reaction `i` occurs given the current state `(M, C)`.

1. **Production terms**: The propensities for production reactions (R1-R3, R5-R6) are derived directly from the positive terms in the ODEs.

2. **Degradation terms**:
   - For M: The ODE has `-M`, giving a first-order degradation with rate constant 1.0, so `a4 = M`
   - For C: The ODE has `-(Gamma2/Gamma1)*C`, so `a7 = (Gamma2/Gamma1)*C`

3. **Hill functions**: The Hill functions are used directly in the propensities as an approximation. In reality, cooperative binding would require explicit modeling of binding states, but this approximation is commonly used in computational biology.

### Gillespie Algorithm Steps

For each iteration:

1. **Compute propensities**: Calculate all `ai` based on current `(M, C)`
2. **Calculate total propensity**: `a0 = sum(ai)`
3. **Draw time to next reaction**: `tau = (1/a0) * log(1/r1)` where `r1 ~ Uniform(0,1)`
4. **Select which reaction occurs**: Choose reaction `i` with probability `ai/a0`
5. **Update state**: Apply stoichiometric changes for selected reaction
6. **Update time**: `t = t + tau`
7. **Repeat** until `t >= T_max`

## Files in this Repository

1. **`gillespie_simulation.py`**: Standalone Gillespie stochastic simulation
   - Contains detailed comments on each reaction and propensity
   - Outputs time traces of M and C

2. **`compare_deterministic_vs_stochastic.py`**: Side-by-side comparison
   - Runs both deterministic ODE and stochastic Gillespie
   - Plots multiple stochastic trajectories against deterministic solution
   - Shows the relationship between the two approaches

## Usage

```bash
# Run Gillespie simulation alone
python gillespie_simulation.py

# Run comparison of both methods
python compare_deterministic_vs_stochastic.py
```

## Expected Behavior

- **Deterministic**: Smooth curves, single trajectory
- **Stochastic**:
  - Fluctuates around deterministic solution
  - Multiple trajectories show different random realizations
  - Fluctuations more pronounced at low molecule counts
  - Converges to deterministic behavior at high molecule counts

## Parameters

All parameters are identical to the original deterministic model:

```python
alpha  = 173    # Production rate parameter
beta   = 50     # Production rate parameter
delta  = 180    # Self-activation rate parameter
Gamma1 = 3.0    # Time scale parameter
Gamma2 = 0.23   # Degradation rate for C
K_M    = 80.0   # Michaelis constant for M self-activation
K_C    = 80.0   # Michaelis constant for M repressing C
K_CM   = 80.0   # Michaelis constant for C activating M
n      = 6      # Hill coefficient
m      = 6      # Hill coefficient
mu_M   = 0.0    # Basal production of M
mu_C   = 0.0    # Basal production of C
```

Initial conditions:
- M(0) = 100
- C(0) = 10

## Mathematical Correspondence

The Gillespie algorithm is mathematically equivalent to the Chemical Master Equation (CME), while the ODE model represents the deterministic mean-field limit.

The relationship:
```
E[dX/dt] ≈ deterministic dX/dt (for large molecule counts)
```

where `E[·]` denotes expectation over stochastic realizations.

## Notes

- The code uses `np.random.seed(42)` for reproducibility
- Set different seeds to see different stochastic realizations
- For very low molecule counts, stochastic effects become more important
- For high molecule counts, the stochastic trajectory will closely follow the deterministic solution
