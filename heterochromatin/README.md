# dh heterochromatin: RNA / siRNA / H3K9me model (Gillespie)

```
dR/dt = T / (1 + (M/K)^k) - d1 R - gamma R M S
dS/dt = gamma R M S + nu R - d2 S
dM/dt = (1 - M)[eps0 R + eps_s R S + eps_m R S M] - d3 M
```

R and S are molecule counts per cell. M = n/N is the methylated fraction of the
N nucleosomes in the locus. The SSA has 7 reactions, listed in `model.py`, and its
large-N limit is the ODE above.

Parameter mapping: `eps0` = eps_nuc (1e-5), `eps_s` = eps_spread (0.002),
`eps_m` = eps_RW (0.005, read-write; set to 0 in W31G). The values of `gamma`,
`eps_m`, `N` and the epe1Δ `d3` factor are not fixed by the experiments yet.

| mutant | change (relative to WT) | target mean RNA |
|---|---|---|
| wt | – | 1 |
| r406h | eps0, eps_s, eps_m / 10 | 3 |
| clr4d | all eps = 0 | 20 (= T/d1, exact) |
| epe1d | d3 × `epe1_factor` (free) | 3 |
| w31g | eps_m = 0 | – |

All mutants start from the WT silent state and relax to their own stationary state. Statistics are
snapshots pooled over cells and over 10 time points after burn-in, as smFISH would sample them.

Optional siRNA nonlinearities (they default off, which is the original model). `phi(S) = S * S^h/(S_c^h+S^h) / (1+S/S_sat)` replaces S in
processing (`S_c_proc`, `S_sat_proc`) and/or in siRNA-guided methylation (`S_c_meth`, `S_sat_meth`).

## Files
- `model.py`: parameters, mutants, ODE, every fixed point and its stability, and the numba Gillespie.
- `analysis.py`: SSA calibration of gamma (sets mean WT RNA to 1), mutant evaluation, score, and smFISH loading and likelihood.
- `run_mutants.py`: calibrates, prints an ODE-vs-SSA table, and plots RNA histograms.
- `phase_diagram.py`: 2D scan over any two parameters (or `epe1_factor`), optionally re-calibrating gamma at each point.

```bash
pip install -r requirements.txt
python run_mutants.py                                   # default params, gamma calibrated by SSA
python run_mutants.py --set eps_m=0.01 N=50 --epe1-factor 0.3
python run_mutants.py --set S_sat_proc=1 S_sat_meth=5 --epe1-factor 0.5   # saturating model
python run_mutants.py --data-dir smfish/                # overlay smFISH: smfish/wt.txt, r406h.txt, ...
python phase_diagram.py --x eps_m:1e-4:0.1:12 --y gamma:3e-3:1:12
python phase_diagram.py --x eps_m:1e-4:0.1:8 --y epe1_factor:0.03:1:8 --calibrate
python phase_diagram.py --x S_sat_proc:0.3:30:8 --y S_sat_meth:0.5:50:8 --calibrate --epe1-factor 0.5
# add --show to either script to open the figure window
```
smFISH files contain one count per cell, one per line (CSV first column, header allowed).

## First observations (default parameters)
1. **In the silent state, R is set by the siRNA loop.** Setting dS/dt = 0 with nu ≈ 0 gives
   `R* ≈ d2 / (gamma M*)`, independent of T. So any mutant that only changes the eps
   values moves the mean RNA through M alone: `R_mut / R_WT ≈ M_WT / M_mut`. When writing is
   weakened, M drops a little, transcription and siRNA production rise, and R goes *down*.
   R406H then comes out below WT (0.5 vs 1) instead of 3×. Getting R406H to 3 needs a regime where the
   siRNA branch cannot compensate.
2. **The stochastic and deterministic models disagree.** WT S is only about 2–10 molecules, and
   S = 0 is nearly absorbing because nu = 1e-5. So cells lose silencing stochastically, even where
   the ODE is monostable and silent. That is why gamma is calibrated with the SSA (about 0.036, against
   about 0.013 from the ODE).
3. **The model can derepress epe1Δ.** Less demethylation means higher M, less transcription,
   fewer siRNAs, and then siRNA extinction. epe1Δ comes out near 3–14 RNA depending on
   `epe1_factor`. The best scan point is eps_m ≈ 3e-4 and epe1_factor ≈ 0.6 (see `results/`).

## Missing mechanism: siRNA saturation (`results/mutants_saturating.png`, `results/phase_saturation.png`)
In the mass-action model, the siRNA pool compensates for any loss of writer activity. Weaker
Clr4 means lower M and more transcription; that makes more siRNA, which drives eps_s R S methylation
back up and also holds R at d2/(gamma M). Making both siRNA-dependent steps **saturate** (a limited
number of RITS/CLRC or Dcr1/RDRC complexes) removes that compensation:

| model (gamma refit, epe1_factor 0.5) | WT | R406H | epe1Δ | W31G | score |
|---|---|---|---|---|---|
| mass action | 1.5 | 0.5 | 9.0 | 0.4 | 2.6 |
| S_sat_meth = 5 only | 1.0 | 1.9 | 10.5 | 0.8 | 1.5 |
| S_sat_proc = 1, S_sat_meth = 5 | 1.0 | 3.8 | 3.8 | 1.2 | 0.08 |
| scan optimum: S_sat_proc ≈ 0.6, S_sat_meth ≈ 7 | 0.9 | 3.1 | 3.2 | 1.1 | 0.008 |

Saturation of methylation is the essential part: with S_sat_meth below about 2, WT can no longer stay silent.
A *threshold* (S_c, a critical siRNA number) does not fix R406H. It acts at low S, but
the compensation that has to be removed happens at high S.

Prediction to test against smFISH: R406H derepression is graded (unimodal, Fano about 2), whereas epe1Δ
derepression is stochastic loss of siRNAs in about 17% of cells. That gives a bimodal distribution with a
mode at 0 and a tail near 20 (Fano about 15); the ODE alone predicts no epe1Δ derepression.
