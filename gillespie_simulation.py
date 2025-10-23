import numpy as np
import matplotlib.pyplot as plt


# ---- parameters ----
alpha     = 173
beta      = 50
delta     = 180
Gamma1    = 3.0
Gamma2    = 0.23
K_M       = 80.0
K_C       = 80.0
K_CM      = 80.0
n         = 6
m         = 6
mu_M      = 0.0
mu_C      = 0.0

# ---- Hill functions ----
def H_act(M):  # M self-activation
    x = (M / K_M)**n
    return x / (1.0 + x)

def H_rep(M):  # M represses C
    x = (M / K_C)**n
    return 1.0 / (1.0 + x)

def Hc(C):     # C activates M (saturating in C!)
    x = (C / K_CM)**m
    return x / (1.0 + x)


# =========================================================================
# GILLESPIE ALGORITHM
# =========================================================================
#
# REACTIONS and PROPENSITIES:
#
# R1: ∅ → M  (basal production of M)
#     Propensity: a1 = mu_M
#
# R2: ∅ → M  (C-activated production of M)
#     Propensity: a2 = (alpha/Gamma1) * Hc(C)
#
# R3: ∅ → M  (self-activated production of M)
#     Propensity: a3 = (delta/Gamma1) * H_act(M)
#
# R4: M → ∅  (degradation of M)
#     Propensity: a4 = 1.0 * M
#
# R5: ∅ → C  (basal production of C)
#     Propensity: a5 = mu_C
#
# R6: ∅ → C  (M-repressed production of C)
#     Propensity: a6 = (beta/Gamma1) * H_rep(M)
#
# R7: C → ∅  (degradation of C)
#     Propensity: a7 = (Gamma2/Gamma1) * C
#
# =========================================================================

def compute_propensities(M, C):
    """
    Compute all reaction propensities given current state (M, C)

    Returns:
        a: numpy array of 7 propensities [a1, a2, a3, a4, a5, a6, a7]
    """
    a = np.zeros(7)

    # M production reactions
    a[0] = mu_M                              # R1: basal production of M
    a[1] = (alpha/Gamma1) * Hc(C)           # R2: C-activated production of M
    a[2] = (delta/Gamma1) * H_act(M)        # R3: self-activated production of M
    a[3] = 1.0 * M                          # R4: degradation of M

    # C production reactions
    a[4] = mu_C                              # R5: basal production of C
    a[5] = (beta/Gamma1) * H_rep(M)         # R6: M-repressed production of C
    a[6] = (Gamma2/Gamma1) * C              # R7: degradation of C

    return a


def gillespie_step(M, C):
    """
    Execute one Gillespie step

    Returns:
        M_new, C_new, dt (new state and time increment)
    """
    # Compute propensities
    a = compute_propensities(M, C)
    a0 = np.sum(a)

    # If no reactions can occur, return infinite time step
    if a0 == 0:
        return M, C, np.inf

    # Draw time to next reaction from exponential distribution
    r1 = np.random.random()
    tau = (1.0 / a0) * np.log(1.0 / r1)

    # Determine which reaction occurs
    r2 = np.random.random()
    cumsum = 0.0
    reaction = 0
    for i in range(7):
        cumsum += a[i] / a0
        if r2 < cumsum:
            reaction = i
            break

    # Update state based on which reaction occurred
    M_new, C_new = M, C

    if reaction == 0:    # R1: ∅ → M
        M_new = M + 1
    elif reaction == 1:  # R2: ∅ → M
        M_new = M + 1
    elif reaction == 2:  # R3: ∅ → M
        M_new = M + 1
    elif reaction == 3:  # R4: M → ∅
        M_new = max(0, M - 1)  # Ensure non-negative
    elif reaction == 4:  # R5: ∅ → C
        C_new = C + 1
    elif reaction == 5:  # R6: ∅ → C
        C_new = C + 1
    elif reaction == 6:  # R7: C → ∅
        C_new = max(0, C - 1)  # Ensure non-negative

    return M_new, C_new, tau


def gillespie_simulation(M0, C0, T_max, max_steps=1000000):
    """
    Run Gillespie simulation

    Args:
        M0, C0: initial molecule counts
        T_max: maximum simulation time
        max_steps: maximum number of reactions

    Returns:
        times, M_trace, C_trace: arrays with time points and molecule counts
    """
    M, C = M0, C0
    t = 0.0

    times = [t]
    M_trace = [M]
    C_trace = [C]

    step = 0
    while t < T_max and step < max_steps:
        M, C, tau = gillespie_step(M, C)
        t += tau

        times.append(t)
        M_trace.append(M)
        C_trace.append(C)

        step += 1

    return np.array(times), np.array(M_trace), np.array(C_trace)


# ---- Run simulation ----
print("Running Gillespie simulation...")
print("=" * 60)

# Initial conditions (same as deterministic)
M0 = 100
C0 = 10
TotalTime = 50.0

# Run Gillespie
np.random.seed(42)  # For reproducibility
times, M_trace, C_trace = gillespie_simulation(M0, C0, TotalTime)

print(f"Simulation completed:")
print(f"  Total simulation time: {times[-1]:.2f}")
print(f"  Number of reactions: {len(times)-1}")
print(f"  Final state: M={M_trace[-1]}, C={C_trace[-1]}")
print("=" * 60)

# ---- Plotting ----
plt.figure(figsize=(10, 6))
plt.step(times, M_trace, label='$M$ (stochastic)', where='post', linewidth=1.5, alpha=0.8)
plt.step(times, C_trace, label='$C$ (stochastic)', where='post', linewidth=1.5, alpha=0.8)
plt.xlabel('Time')
plt.ylabel('Molecule Count')
plt.title('Gillespie Stochastic Simulation')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('gillespie_simulation.png', dpi=150)
print("Plot saved as 'gillespie_simulation.png'")
plt.show()
