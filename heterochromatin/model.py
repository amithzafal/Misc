"""
RNA / siRNA / H3K9-methylation model of dh heterochromatin silencing.

Deterministic form
------------------
    dR/dt = T / (1 + (M/K)^k) - d1 R - gamma R M S
    dS/dt = gamma R M S + nu R - d2 S
    dM/dt = (1 - M) [eps0 R + eps_s R S + eps_m R S M] - d3 M

R : dh RNA molecules per cell
S : siRNA molecules per cell
M : fraction of the N nucleosomes in the locus that carry H3K9me (M = n/N)

Stochastic (Gillespie) form -- reactions and propensities
---------------------------------------------------------
    1  transcription          R -> R+1         T / (1 + (M/K)^k)
    2  RNA decay              R -> R-1         d1 R
    3  RNA -> siRNA (RITS)    R -> R-1, S+1    gamma R M S
    4  indep. siRNA           S -> S+1         nu R
    5  siRNA decay            S -> S-1         d2 S
    6  methylation            n -> n+1         (N - n) (eps0 R + eps_s R S + eps_m R S M)
    7  demethylation          n -> n-1         d3 n

With n/N = M, the large-N limit of the stochastic model is exactly the ODE above.
Units: time in minutes; R and S are molecule counts, so gamma, eps_s, eps_m are
per-molecule rate constants.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

try:
    import numba as nb

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - pure python fallback
    HAVE_NUMBA = False


# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Params:
    T: float = 1.0          # max transcript output, min^-1
    K: float = 0.2          # methylation level for half repression
    k: float = 4.0          # Hill coefficient
    gamma: float = 0.012    # RNA->siRNA (RITS/RDRC) rate; calibrated so <R>_WT = 1
    nu: float = 1e-5        # methylation-independent siRNA production, min^-1
    d1: float = 0.05        # RNA decay, min^-1
    d2: float = 0.005       # siRNA decay, min^-1
    d3: float = 0.015       # H3K9me removal (Epe1 etc.), min^-1
    eps0: float = 1e-5      # nucleation (eps_nuc)
    eps_s: float = 0.002    # siRNA/RITS-guided methylation (eps_spread)
    eps_m: float = 0.005    # read-write (Clr4 chromodomain) methylation (eps_RW)
    N: int = 30             # nucleosomes in the dh locus (sets size of M noise)
    # --- optional siRNA nonlinearities (defaults = original mass-action model) ---
    # siRNA "activity" phi(S) = S * S^h/(S_c^h + S^h) / (1 + S/S_sat)
    #   S_c   : critical siRNA number (cooperative threshold); 0 = no threshold
    #   S_sat : saturation (limited RITS / RDRC / Clr4 recruitment); inf = none
    # phi_p (with *_proc) replaces S in the RNA->siRNA processing term gamma R M S,
    # phi_m (with *_meth) replaces S in the siRNA-guided methylation terms.
    S_c_proc: float = 0.0
    S_sat_proc: float = np.inf
    S_c_meth: float = 0.0
    S_sat_meth: float = np.inf
    h: float = 2.0

    def with_(self, **kw) -> "Params":
        return replace(self, **kw)

    def as_array(self) -> np.ndarray:
        return np.array([self.T, self.K, self.k, self.gamma, self.nu, self.d1,
                         self.d2, self.d3, self.eps0, self.eps_s, self.eps_m,
                         float(self.N), self.S_c_proc, self.S_sat_proc,
                         self.S_c_meth, self.S_sat_meth, self.h])

    @property
    def linear(self) -> bool:
        return (self.S_c_proc == 0 and self.S_c_meth == 0
                and np.isinf(self.S_sat_proc) and np.isinf(self.S_sat_meth))


# --------------------------------------------------------------------------
# Mutants: each is a function Params -> Params, relative to WT/basal values.
# 'epe1_factor' (how much Epe1 loss lowers d3) is not constrained by the
# experiments described so far; it is exposed as an argument.
# --------------------------------------------------------------------------
def mutant_params(p: Params, mutant: str, epe1_factor: float = 0.1,
                  r406h_factor: float = 0.1) -> Params:
    m = mutant.lower()
    if m == "wt":
        return p
    if m == "r406h":      # Clr4 catalytic weakening: all writing /10
        return p.with_(eps0=p.eps0 * r406h_factor, eps_s=p.eps_s * r406h_factor,
                       eps_m=p.eps_m * r406h_factor)
    if m == "clr4d":      # no writer
        return p.with_(eps0=0.0, eps_s=0.0, eps_m=0.0)
    if m == "epe1d":      # reduced demethylation
        return p.with_(d3=p.d3 * epe1_factor)
    if m == "w31g":       # chromodomain: read-write loop broken
        return p.with_(eps_m=0.0)
    raise ValueError(f"unknown mutant {mutant!r}")


MUTANTS = ["wt", "r406h", "clr4d", "epe1d", "w31g"]

# Mean RNA/cell targets: smFISH means (data/*.csv, ch0) except epe1D, which is
# the rough estimate (~3) without a distribution yet.
TARGET_MEAN_R = {"wt": 0.84, "r406h": 2.44, "clr4d": 22.81, "epe1d": 3.0, "w31g": 6.14}


# --------------------------------------------------------------------------
# Deterministic model
# --------------------------------------------------------------------------
def transcription(M, p: Params):
    return p.T / (1.0 + (M / p.K) ** p.k)


def phi(S, S_c, S_sat, h):
    """siRNA activity: S, optionally with a cooperative threshold and saturation."""
    out = S / (1.0 + S / S_sat)
    if S_c > 0:
        Sp = np.maximum(S, 0.0) ** h
        out = out * Sp / (S_c ** h + Sp)
    return out


def rhs(t, y, p: Params):
    R, S, M = y
    conv = p.gamma * R * M * phi(S, p.S_c_proc, p.S_sat_proc, p.h)
    Sm = phi(S, p.S_c_meth, p.S_sat_meth, p.h)
    dR = transcription(M, p) - p.d1 * R - conv
    dS = conv + p.nu * R - p.d2 * S
    dM = (1.0 - M) * (p.eps0 * R + p.eps_s * R * Sm + p.eps_m * R * Sm * M) - p.d3 * M
    return np.array([dR, dS, dM])


def integrate(p: Params, y0, t_end=2e4, n_out=400):
    sol = solve_ivp(rhs, (0, t_end), y0, args=(p,), method="LSODA",
                    t_eval=np.linspace(0, t_end, n_out), rtol=1e-8, atol=1e-10)
    return sol.t, sol.y


def _RS_given_M(M, p: Params):
    """Solve dR/dt = dS/dt = 0 for fixed M. Unique root (h is monotone in R)."""
    f = transcription(M, p)
    gM = p.gamma * M
    r_max = f / p.d1
    if gM > 0:
        r_max = min(r_max, p.d2 / gM * (1 - 1e-12))

    def S_of(R):
        return p.nu * R / (p.d2 - gM * R)

    def h(R):
        return f - p.d1 * R - gM * R * S_of(R)

    if p.nu == 0 or gM == 0:
        R = f / p.d1 if gM == 0 else brentq(h, 0, r_max)
    else:
        R = brentq(h, 0.0, r_max, xtol=1e-14, rtol=1e-12)
    return R, S_of(R)


def _jacobian(y, p: Params, h=1e-7):
    J = np.zeros((3, 3))
    f0 = rhs(0, y, p)
    for j in range(3):
        dy = np.zeros(3)
        dy[j] = h * max(1.0, abs(y[j]))
        J[:, j] = (rhs(0, y + dy, p) - f0) / dy[j]
    return J


def _RS_given_M_vec(Ms, p: Params, iters=80):
    """Vectorised bisection version of _RS_given_M for a grid of M values."""
    f = transcription(Ms, p)
    gM = p.gamma * Ms
    hi = f / p.d1
    with np.errstate(divide="ignore"):
        hi = np.where(gM > 0, np.minimum(hi, p.d2 / np.where(gM > 0, gM, 1) * (1 - 1e-12)), hi)
    lo = np.zeros_like(Ms)

    def h(R):
        return f - p.d1 * R - gM * R * p.nu * R / (p.d2 - gM * R)

    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        pos = h(mid) > 0
        lo = np.where(pos, mid, lo)
        hi = np.where(pos, hi, mid)
    R = 0.5 * (lo + hi)
    return R, p.nu * R / (p.d2 - gM * R)


def _fixed_points_multistart(p: Params):
    """Stable steady states by integrating from a spread of initial conditions
    (used when the siRNA terms are nonlinear; unstable states are not returned)."""
    found = []
    f0 = p.T / p.d1
    for M0 in (0.0, 0.1, 0.3, 0.6, 1.0):
        for S0 in (0.0, 1.0, 10.0, 100.0):
            for R0 in (0.0, f0):
                sol = solve_ivp(rhs, (0, 2e5), [R0, S0, M0], args=(p,), method="LSODA",
                                rtol=1e-8, atol=1e-10)
                y = sol.y[:, -1]
                if np.abs(rhs(0, y, p)).max() > 1e-7:
                    continue
                if all(np.abs(y - np.array([q["R"], q["S"], q["M"]])).max() >
                       1e-3 * (1 + np.abs(y).max()) for q in found):
                    ev = np.linalg.eigvals(_jacobian(y, p))
                    found.append(dict(R=y[0], S=y[1], M=y[2],
                                      stable=bool(np.all(ev.real < 0))))
    return found


def fixed_points(p: Params, n_grid=3000):
    """All steady states (R, S, M, stable) found by scanning M in [0, 1).
    For nonlinear siRNA terms, falls back to multistart integration (stable only)."""
    if not p.linear:
        return _fixed_points_multistart(p)
    Ms = np.concatenate([[0.0], np.logspace(-8, np.log10(0.999999), n_grid)])

    def g_of(M, R, S):
        return (1 - M) * (p.eps0 * R + p.eps_s * R * S + p.eps_m * R * S * M) - p.d3 * M

    def g(M):
        R, S = _RS_given_M(M, p)
        return g_of(M, R, S)

    gs = g_of(Ms, *_RS_given_M_vec(Ms, p))
    out = []
    if gs[0] == 0.0:  # M = 0 is a steady state only when nothing writes
        R, S = _RS_given_M(0.0, p)
        out.append((R, S, 0.0))
    for i in np.nonzero(np.sign(gs[:-1]) * np.sign(gs[1:]) < 0)[0]:
        M = brentq(g, Ms[i], Ms[i + 1], xtol=1e-14)
        R, S = _RS_given_M(M, p)
        out.append((R, S, M))
    res = []
    for R, S, M in out:
        ev = np.linalg.eigvals(_jacobian(np.array([R, S, M]), p))
        res.append(dict(R=R, S=S, M=M, stable=bool(np.all(ev.real < 0))))
    return res


def silent_state(p: Params, fps=None):
    """Most-methylated stable steady state (the heterochromatic branch if it exists)."""
    st = [fp for fp in (fps or fixed_points(p)) if fp["stable"]]
    return max(st, key=lambda d: d["M"]) if st else None


def active_state(p: Params, fps=None):
    st = [fp for fp in (fps or fixed_points(p)) if fp["stable"]]
    return min(st, key=lambda d: d["M"]) if st else None


def calibrate_gamma(p: Params, target_R=1.0, lo=1e-5, hi=10.0):
    """Choose gamma so the WT silent steady state has <R> = target_R (deterministic).
    Returns None if no gamma in [lo, hi] works."""
    def err(lg):
        fp = silent_state(p.with_(gamma=10 ** lg))
        if fp is None or fp["M"] < p.K:   # no silent branch -> R too high
            return np.log(transcription(0.0, p) / p.d1 / target_R) + 10
        return np.log(fp["R"] / target_R)

    a, b = np.log10(lo), np.log10(hi)
    grid = np.linspace(a, b, 40)
    e = np.array([err(x) for x in grid])
    idx = np.nonzero(np.sign(e[:-1]) * np.sign(e[1:]) < 0)[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    return 10 ** brentq(err, grid[i], grid[i + 1], xtol=1e-8)


# --------------------------------------------------------------------------
# Gillespie SSA
# --------------------------------------------------------------------------
def _ssa_cell(par, R, S, n, t_sample, seed):
    T, K, k, gamma, nu, d1, d2, d3, eps0, eps_s, eps_m, Nf, Scp, Ssp, Scm, Ssm, h = par
    N = int(Nf)
    np.random.seed(seed)
    out = np.empty((t_sample.shape[0], 3))
    t = 0.0
    j = 0
    a = np.empty(7)
    while j < t_sample.shape[0]:
        M = n / N
        Sp = S / (1.0 + S / Ssp)
        Sm = S / (1.0 + S / Ssm)
        if Scp > 0:
            Sp *= S ** h / (Scp ** h + S ** h)
        if Scm > 0:
            Sm *= S ** h / (Scm ** h + S ** h)
        a[0] = T / (1.0 + (M / K) ** k)
        a[1] = d1 * R
        a[2] = gamma * R * M * Sp
        a[3] = nu * R
        a[4] = d2 * S
        a[5] = (N - n) * R * (eps0 + eps_s * Sm + eps_m * Sm * M)
        a[6] = d3 * n
        a0 = a.sum()
        t_next = t + np.random.exponential(1.0 / a0)
        while j < t_sample.shape[0] and t_sample[j] < t_next:
            out[j, 0] = R
            out[j, 1] = S
            out[j, 2] = n
            j += 1
        t = t_next
        r = np.random.random() * a0
        c = a[0]
        if r < c:
            R += 1
            continue
        c += a[1]
        if r < c:
            R -= 1
            continue
        c += a[2]
        if r < c:
            R -= 1
            S += 1
            continue
        c += a[3]
        if r < c:
            S += 1
            continue
        c += a[4]
        if r < c:
            S -= 1
            continue
        c += a[5]
        if r < c:
            n += 1
            continue
        n -= 1
    return out


if HAVE_NUMBA:
    _ssa_cell_jit = nb.njit(cache=True)(_ssa_cell)

    @nb.njit(parallel=True, cache=True)
    def _ssa_many(par, R0, S0, n0, t_sample, seeds):
        nc = R0.shape[0]
        out = np.empty((nc, t_sample.shape[0], 3))
        for i in nb.prange(nc):
            out[i] = _ssa_cell_jit(par, R0[i], S0[i], n0[i], t_sample, seeds[i])
        return out
else:
    def _ssa_many(par, R0, S0, n0, t_sample, seeds):
        return np.stack([_ssa_cell(par, R0[i], S0[i], n0[i], t_sample, seeds[i])
                         for i in range(R0.shape[0])])


def initial_state(p: Params, kind="silent"):
    """Integer initial (R, S, n). 'silent' = deterministic silent branch of p
    (falls back to fully methylated), 'active' = unmethylated, R = T/d1."""
    if kind == "active":
        return int(round(p.T / p.d1)), 0, 0
    fp = silent_state(p)
    if fp is None or fp["M"] < 1e-3:
        return 1, 10, p.N
    return int(round(fp["R"])), int(round(fp["S"])), int(round(fp["M"] * p.N))


def simulate(p: Params, n_cells=500, t_burn=2e4, t_sample=None, init="silent",
             init_from: Params | None = None, seed=0):
    """Run n_cells independent cells.

    init       : 'silent' | 'active' | (R, S, n) tuple
    init_from  : take the initial condition from this parameter set's steady
                 state (e.g. start a mutant from the WT silent state)
    t_sample   : times (after burn-in) at which to record; default = one
                 snapshot at the end (what smFISH sees).
    Returns array (n_cells, n_times, 3) with columns R, S, n (methylated count).
    """
    if t_sample is None:
        t_sample = np.array([0.0])
    t_sample = np.asarray(t_sample, float) + t_burn
    if isinstance(init, str):
        R0, S0, n0 = initial_state(init_from or p, init)
    else:
        R0, S0, n0 = init
    if init_from is not None:
        n0 = int(round(n0 / init_from.N * p.N))
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31 - 1, n_cells)
    return _ssa_many(p.as_array(), np.full(n_cells, R0, np.int64),
                     np.full(n_cells, S0, np.int64), np.full(n_cells, n0, np.int64),
                     t_sample, seeds)


def summarize(traj, p: Params):
    """Statistics pooled over all cells and sample times.
    frac_silent = fraction of cells with M > K (transcription < T/2)."""
    R = traj[:, :, 0].ravel()
    S = traj[:, :, 1].ravel()
    M = traj[:, :, 2].ravel() / p.N
    return dict(R_mean=R.mean(), R_var=R.var(), R_fano=R.var() / max(R.mean(), 1e-12),
                frac_R0=np.mean(R == 0), S_mean=S.mean(), M_mean=M.mean(),
                frac_silent=np.mean(M > p.K))


def rna_histogram(traj, r_max=None):
    R = traj[:, :, 0].ravel().astype(int)
    r_max = r_max or R.max()
    return np.bincount(np.minimum(R, r_max), minlength=r_max + 1) / R.size


def params_dict(p: Params):
    return asdict(p)
