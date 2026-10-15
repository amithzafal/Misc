"""dh module (best fit, eps0 = 0) coupled to a ura4 reporter at cen1.

dh  : dR/dt = T/(1+(M/K)^k) - d1 R - gamma R M S/(1+S/S_p)
      dS/dt = gamma R M S/(1+S/S_p) + nu R - d2 S
      dM/dt = (1-M)(eps_s + eps_m M) R S/(1+S/S_m) - d3 M
ura4: dU/dt = T_u/(1+(M_u/K)^k) - d1 U
      dM_u/dt = (1-M_u)[kappa M + eps_u M_u] - d3 M_u

Gillespie with M = n/N, M_u = n_u/N_u. The ura4 reactions are
    U+1   T_u/(1+(M_u/K)^k)       U-1   d1 U
    n_u+1 (N_u-n_u)(kappa M + eps_u M_u)    n_u-1 d3 n_u
"""
import numpy as np
from numba import njit, prange

DH = dict(T=1.1406, K=0.2, k=4.0, gamma=0.06658, S_p=4.921, nu=1e-5,
          d1=0.05, d2=0.005, d3=0.015, eps_s=5.246e-05, eps_m=0.009437,
          S_m=12.9, N=30)
URA4 = dict(T_u=0.775, kappa=0.01, eps_u=0.01, N_u=10)

KEYS = ["T", "K", "k", "gamma", "S_p", "nu", "d1", "d2", "d3", "eps_s", "eps_m",
        "S_m", "N", "T_u", "kappa", "eps_u", "N_u"]


def strain(name, p, epe1_factor=0.5):
    """Strain = changes relative to WT. W31G: no chromodomain read-write at
    either locus. clr4D: no writing at all. epe1D: slower H3K9me removal."""
    p = dict(p)
    if name == "w31g":
        p.update(eps_m=0.0, eps_u=0.0)
    elif name == "clr4d":
        p.update(eps_s=0.0, eps_m=0.0, kappa=0.0, eps_u=0.0)
    elif name == "r406h":
        for x in ("eps_s", "eps_m", "kappa", "eps_u"):
            p[x] *= 0.1
    elif name == "epe1d":
        p["d3"] *= epe1_factor
    return p


@njit(cache=True)
def _cell(par, t_sample, seed):
    (T, K, k, gamma, S_p, nu, d1, d2, d3, eps_s, eps_m, S_m, Nf,
     T_u, kappa, eps_u, Nuf) = par
    N, Nu = int(Nf), int(Nuf)
    np.random.seed(seed)
    R, S, n, U, nu_ = 0, 20, N, 0, Nu        # start silenced at both loci
    out = np.empty((t_sample.shape[0], 4))
    a = np.empty(11)
    t, j = 0.0, 0
    while j < t_sample.shape[0]:
        M = n / N
        Mu = nu_ / Nu
        a[0] = T / (1.0 + (M / K) ** k)
        a[1] = d1 * R
        a[2] = gamma * R * M * S / (1.0 + S / S_p)
        a[3] = nu * R
        a[4] = d2 * S
        a[5] = (N - n) * R * (eps_s + eps_m * M) * S / (1.0 + S / S_m)
        a[6] = d3 * n
        a[7] = T_u / (1.0 + (Mu / K) ** k)
        a[8] = d1 * U
        a[9] = (Nu - nu_) * (kappa * M + eps_u * Mu)
        a[10] = d3 * nu_
        a0 = a.sum()
        t += np.random.exponential(1.0 / a0)
        while j < t_sample.shape[0] and t_sample[j] < t:
            out[j, 0] = R
            out[j, 1] = U
            out[j, 2] = M
            out[j, 3] = Mu
            j += 1
        r = np.random.random() * a0
        i, c = 0, a[0]
        while r >= c:
            i += 1
            c += a[i]
        if i == 0:
            R += 1
        elif i == 1:
            R -= 1
        elif i == 2:
            R -= 1
            S += 1
        elif i == 3:
            S += 1
        elif i == 4:
            S -= 1
        elif i == 5:
            n += 1
        elif i == 6:
            n -= 1
        elif i == 7:
            U += 1
        elif i == 8:
            U -= 1
        elif i == 9:
            nu_ += 1
        else:
            nu_ -= 1
    return out


@njit(parallel=True, cache=True)
def _many(par, t_sample, seeds):
    out = np.empty((seeds.shape[0], t_sample.shape[0], 4))
    for c in prange(seeds.shape[0]):
        out[c] = _cell(par, t_sample, seeds[c])
    return out


def simulate(p, n_cells=1000, t_burn=2e4, t_sample=np.arange(0, 5000, 500), seed=0):
    """Returns dict of pooled snapshots: R (dh RNA), U (ura4 RNA), M, Mu."""
    par = np.array([float(p[x]) for x in KEYS])
    seeds = np.random.default_rng(seed).integers(0, 2**31 - 1, n_cells)
    o = _many(par, np.asarray(t_sample, float) + t_burn, seeds)
    return dict(R=o[:, :, 0].ravel(), U=o[:, :, 1].ravel(), M=o[:, :, 2].ravel(),
                Mu=o[:, :, 3].ravel(), traj=o)
