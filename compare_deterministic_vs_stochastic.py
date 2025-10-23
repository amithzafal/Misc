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

def dMdt(M, C):
    return mu_M + (alpha/Gamma1)*Hc(C) + (delta/Gamma1)*H_act(M) - M

def dCdt(M, C):
    return mu_C + (beta/Gamma1)*H_rep(M) - (Gamma2/Gamma1)*C


# =========================================================================
# DETERMINISTIC SIMULATION (ODE)
# =========================================================================

def run_deterministic(M0, C0, TotalTime, dt=0.01):
    """Run deterministic ODE simulation"""
    Ntot = int(TotalTime/dt)
    M = np.zeros(Ntot)
    C = np.zeros(Ntot)
    T = np.linspace(0, TotalTime, Ntot)

    M[0] = M0
    C[0] = C0

    for t in range(Ntot-1):
        M[t+1] = M[t] + dt*dMdt(M[t], C[t])
        C[t+1] = C[t] + dt*dCdt(M[t], C[t])

    return T, M, C


# =========================================================================
# GILLESPIE STOCHASTIC SIMULATION
# =========================================================================

def compute_propensities(M, C):
    """Compute all reaction propensities"""
    a = np.zeros(7)
    a[0] = mu_M                              # R1: basal production of M
    a[1] = (alpha/Gamma1) * Hc(C)           # R2: C-activated production of M
    a[2] = (delta/Gamma1) * H_act(M)        # R3: self-activated production of M
    a[3] = 1.0 * M                          # R4: degradation of M
    a[4] = mu_C                              # R5: basal production of C
    a[5] = (beta/Gamma1) * H_rep(M)         # R6: M-repressed production of C
    a[6] = (Gamma2/Gamma1) * C              # R7: degradation of C
    return a


def gillespie_step(M, C):
    """Execute one Gillespie step"""
    a = compute_propensities(M, C)
    a0 = np.sum(a)

    if a0 == 0:
        return M, C, np.inf

    r1 = np.random.random()
    tau = (1.0 / a0) * np.log(1.0 / r1)

    r2 = np.random.random()
    cumsum = 0.0
    reaction = 0
    for i in range(7):
        cumsum += a[i] / a0
        if r2 < cumsum:
            reaction = i
            break

    M_new, C_new = M, C
    if reaction == 0:    # R1: ∅ → M
        M_new = M + 1
    elif reaction == 1:  # R2: ∅ → M
        M_new = M + 1
    elif reaction == 2:  # R3: ∅ → M
        M_new = M + 1
    elif reaction == 3:  # R4: M → ∅
        M_new = max(0, M - 1)
    elif reaction == 4:  # R5: ∅ → C
        C_new = C + 1
    elif reaction == 5:  # R6: ∅ → C
        C_new = C + 1
    elif reaction == 6:  # R7: C → ∅
        C_new = max(0, C - 1)

    return M_new, C_new, tau


def gillespie_simulation(M0, C0, T_max, max_steps=1000000):
    """Run Gillespie simulation"""
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


# =========================================================================
# RUN BOTH SIMULATIONS
# =========================================================================

M0 = 100
C0 = 10
TotalTime = 50.0

print("Running simulations...")
print("=" * 60)

# Deterministic
T_det, M_det, C_det = run_deterministic(M0, C0, TotalTime)
print(f"Deterministic ODE: {len(T_det)} time points")

# Stochastic (multiple trajectories)
n_trajectories = 5
np.random.seed(42)

trajectories = []
for i in range(n_trajectories):
    times, M_trace, C_trace = gillespie_simulation(M0, C0, TotalTime)
    trajectories.append((times, M_trace, C_trace))
    print(f"Gillespie trajectory {i+1}: {len(times)-1} reactions")

print("=" * 60)

# =========================================================================
# PLOTTING
# =========================================================================

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot M
ax1.plot(T_det, M_det, 'b-', linewidth=2.5, label='Deterministic (ODE)', zorder=10)
for i, (times, M_trace, C_trace) in enumerate(trajectories):
    if i == 0:
        ax1.step(times, M_trace, 'r-', where='post', linewidth=1.0,
                alpha=0.6, label='Stochastic (Gillespie)')
    else:
        ax1.step(times, M_trace, 'r-', where='post', linewidth=1.0, alpha=0.6)

ax1.set_ylabel('M (molecule count)', fontsize=12)
ax1.set_title('Comparison: Deterministic ODE vs Stochastic Gillespie', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(alpha=0.3)

# Plot C
ax2.plot(T_det, C_det, 'g-', linewidth=2.5, label='Deterministic (ODE)', zorder=10)
for i, (times, M_trace, C_trace) in enumerate(trajectories):
    if i == 0:
        ax2.step(times, C_trace, 'orange', where='post', linewidth=1.0,
                alpha=0.6, label='Stochastic (Gillespie)')
    else:
        ax2.step(times, C_trace, 'orange', where='post', linewidth=1.0, alpha=0.6)

ax2.set_xlabel('Time', fontsize=12)
ax2.set_ylabel('C (molecule count)', fontsize=12)
ax2.legend(fontsize=11)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('comparison_deterministic_vs_stochastic.png', dpi=150)
print("Comparison plot saved as 'comparison_deterministic_vs_stochastic.png'")
plt.show()
