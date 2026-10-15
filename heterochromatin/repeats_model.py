"""L independent dh repeats (cis RNAi loop each) + ura4 reporter next to repeat 0 (cen1).

Each repeat i has its own R_i, S_i (siRNA acting in cis) and n_i (M_i = n_i/N_i):
    R_i+1  (T/L)/(1+(M_i/K)^k)            R_i-1  d1 R_i
    R_i-1,S_i+1  gamma R_i M_i S_i/(1+S_i/S_p)
    S_i+1  nu R_i                          S_i-1  d2 S_i
    n_i+1  (N_i-n_i) R_i (eps_s + eps_m M_i) S_i/(1+S_i/S_m)
    n_i-1  d3 n_i
ura4:
    U+1  T_u/(1+(M_u/K)^k)   U-1  d1 U
    n_u+1 (N_u-n_u)(kappa M_0 + eps_u M_u)   n_u-1  d3 n_u
smFISH dh signal = sum_i R_i ; ura4 signal = U.
Optionally the siRNA pool is shared by all repeats (trans), set shared=1.
"""
import numpy as np
from numba import njit, prange

BASE = dict(T=1.1406, K=0.2, k=4.0, gamma=0.07, S_p=1.0, nu=1e-5, d1=0.05,
            d2=0.005, d3=0.015, eps_s=5e-5, eps_m=0.01, S_m=3.0, N=30, L=10,
            N0=30, T_u=0.775, kappa=0.04, eps_u=0.003, N_u=10, shared=0)
KEYS = ["T", "K", "k", "gamma", "S_p", "nu", "d1", "d2", "d3", "eps_s", "eps_m",
        "S_m", "N", "L", "N0", "T_u", "kappa", "eps_u", "N_u", "shared"]


def strain(name, p, epe1_factor=0.5, epe1_writer=1.0):
    """epe1_writer < 1: in epe1D, less Clr4 available at the centromere
    (e.g. titrated by heterochromatin spreading elsewhere), scaling all writing."""
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
        for x in ("eps_s", "eps_m", "kappa", "eps_u"):
            p[x] *= epe1_writer
    return p


@njit(cache=True)
def _cell(par, t_sample, seed):
    (T, K, k, gamma, S_p, nu, d1, d2, d3, eps_s, eps_m, S_m, Nf, Lf, N0f,
     T_u, kappa, eps_u, Nuf, shared) = par
    L, Nu = int(Lf), int(Nuf)
    np.random.seed(seed)
    Ns = np.full(L, int(Nf))
    Ns[0] = int(N0f)                       # repeat next to ura4 may differ
    R = np.zeros(L, np.int64)
    S = np.full(L, 2, np.int64)             # start silenced
    n = Ns.copy()
    U, nu_ = 0, Nu
    Tr = T / L
    nr = 7
    a = np.empty(nr * L + 4)
    out = np.empty((t_sample.shape[0], 3))
    t, j = 0.0, 0
    while j < t_sample.shape[0]:
        Stot = S.sum()
        for i in range(L):
            M = n[i] / Ns[i]
            s = Stot if shared > 0 else S[i]
            b = nr * i
            a[b] = Tr / (1.0 + (M / K) ** k)
            a[b + 1] = d1 * R[i]
            a[b + 2] = gamma * R[i] * M * s / (1.0 + s / S_p)
            a[b + 3] = nu * R[i]
            a[b + 4] = d2 * S[i]
            a[b + 5] = (Ns[i] - n[i]) * R[i] * (eps_s + eps_m * M) * s / (1.0 + s / S_m)
            a[b + 6] = d3 * n[i]
        Mu = nu_ / Nu
        b = nr * L
        a[b] = T_u / (1.0 + (Mu / K) ** k)
        a[b + 1] = d1 * U
        a[b + 2] = (Nu - nu_) * (kappa * n[0] / Ns[0] + eps_u * Mu)
        a[b + 3] = d3 * nu_
        a0 = a.sum()
        t += np.random.exponential(1.0 / a0)
        while j < t_sample.shape[0] and t_sample[j] < t:
            out[j, 0] = R.sum()
            out[j, 1] = U
            out[j, 2] = n[0] / Ns[0]
            j += 1
        r = np.random.random() * a0
        q, c = 0, a[0]
        while r >= c:
            q += 1
            c += a[q]
        if q < nr * L:
            i, e = q // nr, q % nr
            if e == 0:
                R[i] += 1
            elif e == 1:
                R[i] -= 1
            elif e == 2:
                R[i] -= 1
                S[i] += 1
            elif e == 3:
                S[i] += 1
            elif e == 4:
                S[i] -= 1
            elif e == 5:
                n[i] += 1
            else:
                n[i] -= 1
        else:
            e = q - nr * L
            if e == 0:
                U += 1
            elif e == 1:
                U -= 1
            elif e == 2:
                nu_ += 1
            else:
                nu_ -= 1
    return out


@njit(parallel=True, cache=True)
def _many(par, t_sample, seeds):
    out = np.empty((seeds.shape[0], t_sample.shape[0], 3))
    for c in prange(seeds.shape[0]):
        out[c] = _cell(par, t_sample, seeds[c])
    return out


def simulate(p, n_cells=500, t_burn=2e4, t_sample=np.arange(0, 5000, 500), seed=0):
    """Pooled snapshots: dh = total dh RNA, U = ura4 RNA, M0 = methylation of repeat 0."""
    par = np.array([float(p[x]) for x in KEYS])
    seeds = np.random.default_rng(seed).integers(0, 2**31 - 1, n_cells)
    o = _many(par, np.asarray(t_sample, float) + t_burn, seeds)
    return dict(dh=o[:, :, 0].ravel(), U=o[:, :, 1].ravel(), M0=o[:, :, 2].ravel())
