# Review of `pombe_nucseg`

Everything below was run, not just read: the package was assembled with a
stand-in `__init__.py`, torch 2.14 (CPU) was installed, and each claim was
checked with a small synthetic experiment. Ordered by how likely the issue is
to corrupt results silently. Items marked **fixed** are changed in this
commit; the rest are recommendations.

## 1. Bleed-through passes the SNR gate and produces a false, offset, up-weighted label — fixed

Two rods side by side with a 3 px gap, one expressing NLS-GFP, the other not,
PSF blur σ = 1 px plus a broad halo. Original `labelgen`:

| bright neighbour | dark cell SNR | outcome |
|---|---|---|
| 1×, no halo | 4.4 | passes the gate (3.0); rejected only by luck (`too_many_nuclei`) |
| 3×, halo 0.5 | 9.4 | **qc_pass, 295 px "nucleus", centroid 12 px toward the neighbour, class `anaphase_bridge`** |
| 3×, halo 1.0 | 15.6 | same |

So the failure is exactly the one you feared, and worse: the false nucleus
hugs the shared boundary, is elongated along it, and therefore gets classed
as a bridge and weighted 6×. The 0.15 µm erosion is two pixels and does
nothing against haze.

Two causes. (a) `frame_background` estimated σ as the std of the *lower half*
of the background distribution, which is ~0.6 σ for a Gaussian, so every SNR
was inflated ~1.65×. Now 1.4826 × MAD. (b) The SNR gate only asks "is there
signal above background noise", which haze satisfies. Two gates were added
that ask the right questions:

* `min_nuc_cyto_ratio` (1.5): median GFP in the nucleus / median GFP in the
  rest of the cell. Real NLS-GFP nuclei score 8–20+ on synthetic data; haze
  is a shallow gradient and scores ~1–3.
* `max_outside_ratio` (0.5): median GFP in the background ring just outside
  the cell, adjacent to the nucleus, relative to the nucleus. A real nucleus
  sits ≥ 1 µm inside the cortex so that ring is empty (ratio 0). Haze
  continues across the boundary.

After the fix all four cases above are rejected (`low_gfp_snr` / `edge_bleed`),
and no real cell in the synthetic movie is rejected (min contrast 8.2, max
outside ratio 0.08). Both thresholds have an order of magnitude of margin;
still, calibrate on real data by looking at the two new table columns for
cells you know are dark. Rejected `edge_bleed` cells are now ranked for
annotation, since they are real failure modes.

Anything that survives these gates but is still wrong will be caught by the
gold set in the README. A time-lapse check is also cheap: expression is
stable along a track, so a cell that flips pass/fail between frames is
suspect.

## 2. `apply_manual_overrides` leaves the second daughter's auto pixels in place — fixed

Auto labels for a `two_nuclei` cell carry IDs `cid` and `cid + offset`. The
override cleared only pixels `== cid`, so overriding a two-nucleus (or
borderline) cell left the second nucleus's pixels behind, and `build_crops`
folds *any* non-zero pixel in the cell into the target. Confirmed on the
original code: IDs inside the cell before override `[10, 110]`, after
`[10, 110]`; a 323 px hand-drawn mask became a 496 px training target, and
the condition stayed `two_nuclei` although one object was drawn. This
corrupts precisely the labels you are paying for.

Fix: pass `cell_labels`; every nucleus pixel inside the cell is cleared before
the manual mask is written, and the manual mask is clipped to the cell (a
warning reports dropped pixels). Without `cell_labels` it falls back to ID
arithmetic (`id % offset == cid`). Also fixed while there:

* The condition is now re-derived from the drawn mask for every overridden
  cell, not just previously rejected ones. A hand-drawn dumbbell kept its
  `interphase` class (and weight 1) if the auto pass had called it that.
* A manual ID with no table row is warned about instead of silently written
  and never used.
* `position` argument, because the keys are `(position, frame)` but
  `nucleus_labels` is one position's stack; a manual image from another
  position would have been written into the wrong frame.

## 3. Crop size does not fit the cells — fixed (warning + `meta["cell_clipped"]`)

`crop_size=128` is in pixels, `crop_pad_um` is never used. At 0.1 µm/px the
synthetic cells reach 15 µm and **12 of 60 crops in the default
`run_pipeline` touch the crop border**. At 0.065 µm/px (6.5 µm sCMOS pixels
at 100×, which the config comment describes) a 14 µm cell is 215 px. The cell
tips are cut off, and in anaphase the daughter nuclei are at the tips: their
pixels are missing from the training label and can never be predicted at
inference. Nothing in the original pipeline reports this.

`build_crops` now warns with the longest cell and sets `cell_clipped` per
crop. Set `crop_size ≥ (longest cell + 2 µm) / pixel_size`; 192 or 256 at
0.1 µm/px, 256 at 0.065.

## 4. Inference cannot run on the target data — fixed

`predict_movie` requires the label table for `cell_y/cell_x/cell_orientation`,
and that table comes from `generate_labels_frame`, which needs GFP. On a
GFP-free movie there was no way to call it. `cell_table_from_masks` builds
the geometry from the cell masks; `predict_movie(..., table=None, ...)` uses it.

## 5. `_loss_fn` normalises by the *batch* mean weight — fixed

Verified numerically: a batch of only weight-24 crops produced exactly the
same loss as a batch of only weight-1 crops, and in a mixed batch an
interphase crop was scaled 0.26× or 1.0× depending on whether a manual crop
happened to share its batch. The weighted objective was therefore neither
what the docstring said nor consistent between batches. The weights are now
divided by the mean weight of the training set, a constant, so the loss is an
unbiased estimate of the weighted objective and the "scale ~1" property holds
on average. The consequence you were worried about, a large step from an
all-high-weight batch, is real under the corrected objective and is bounded
by the new `grad_clip` (1.0, applied after `scaler.unscale_`).

The two routes do not double-apply: `sampler` mode ignores the weights in the
loss (verified), `loss` mode samples uniformly, `both` multiplies as documented.

## 6. No-validation path never checkpoints until loss < 1 — fixed

`best = -1.0` and the no-val score is `-loss`; BCE + 0.5 Dice starts at
~1.2, so nothing was written (confirmed: one epoch, no file). If an earlier
run left `nucleus_unet.pt` behind, `load_model` would silently load stale
weights. Now `-inf`.

## 7. Intensity augmentation was applied to the cell-mask channel — fixed

Gain 0.9–1.1 and an offset were multiplied into all channels including the
mask, which is exactly {0, 1} at inference. Small train/test shift, now
restricted to the image planes. Note the "noise" term is a single scalar per
crop, not per-pixel noise; left as is, but rename or change it deliberately.

## 8. `rank_for_annotation` merge — fixed

Your suspicion was right in mechanism. `merge(how="left")` preserves the
left order only when keys are unique on the right and complete. If `metrics`
lacks `position` while the table spans several positions, the join
cross-multiplies rows (confirmed: a two-position table returned a ranking
with no error and doubled rows); duplicate keys in `metrics` (two methods
concatenated) do the same; the `priority[:len(t)]` slice hides the length
mismatch instead of raising. Replaced with a key lookup (`reindex` on a MultiIndex) that cannot
change row count or order, plus an explicit error for the ambiguous
multi-position case. Verified that a cell given IoU 0 ranks first.

Rule 4 in the docstring, "borderline two-nucleus calls", had no
implementation. It now uses `split_borderline` (item 10). `frame_stride` thins
consecutive frames of the same dividing cell, otherwise the top 200 is the
same ten cells twenty times.

## 9. AMP API — fixed

`torch.cuda.amp.GradScaler` / `torch.cuda.amp.autocast` emit FutureWarnings
on torch ≥ 2.4. Now `torch.amp.GradScaler("cuda", ...)` and
`torch.autocast(device_type="cuda", dtype=torch.float16, ...)`; the loss is
computed outside autocast in fp32 (Dice sums over 16k pixels lose precision
in fp16). Checkpoint `config` dict is plain Python and loads under the
torch ≥ 2.6 `weights_only=True` default.

## 10. Anaphase: the bridge / two-nuclei call — changed

Synthetic dumbbell, 2 µm centre separation, 0.4 µm bridge whose intensity is
a fraction of the lobes, PSF σ = 1.2 px:

| bridge / lobe intensity | Otsu components at 0.8·thr / thr / 1.2·thr | auto class | neck ratio |
|---|---|---|---|
| 0.15 | 2 / 2 / 2 | two_nuclei | 0.20 ± 0.01 |
| 0.25 | 2 / 2 / 2 | two_nuclei | 0.28 |
| 0.35 | **1 / 2 / 2** | two_nuclei | 0.37 |
| 0.45 | **1 / 1 / 2** | anaphase_bridge | 0.45 |
| 0.55 | 1 / 1 / 1 | anaphase_bridge | 0.53 |
| 0.70 | 1 / 1 / 1 | anaphase_bridge | 0.66 |

The Otsu call flips at a bridge intensity of ~0.4 and a ±20 % change in the
threshold moves the flip across 0.35–0.45, which is your point exactly.

On your two options: plain hysteresis just moves the arbitrariness to the low
threshold and grows the mask; a watershed forces a split on every dumbbell,
which is wrong for the early bridge. What is stable is the quantity both of
them are implicitly probing, the *saddle intensity between the lobes*.
`labelgen.neck_ratio` finds the two brightest maxima ≥ 1 µm apart and
binary-searches the threshold at which they become one component; the ratio
of that to the dimmer peak is 1.0 for one blob and 0.0 for two separated
nuclei. It is threshold-free, costs ~12 `ndi.label` calls on a cell crop,
and on the table above tracks bridge intensity with sd 0.01.

It is used three ways:

* `nuc_neck_ratio` in the table, so you can plot its distribution on real
  data and see where the population is bimodal.
* `split_borderline` when it falls in `neck_borderline` (0.25–0.6, the band
  where Otsu flips). These cells rank first for annotation and, while still
  auto-labelled, get `borderline_auto_weight` (0.25; set 0 to drop them).
* Two Otsu components with a neck ratio above the band are called
  `anaphase_bridge`, because the threshold cut them, not biology. The mask
  itself is left as Otsu produced it; the class and the weight are what the
  ranking and training use.

The aspect (1.7) / solidity (0.92) split for single objects is fine: on
synthetic data it produced no false anaphase calls among 79 interphase
nuclei, and the OR is right because a pre-waist elongated nucleus has high
aspect and high solidity while a dumbbell has low solidity. Two cautions.
Solidity of a small noisy Otsu blob can dip below 0.92 at low SNR, so watch
the interphase rows of the report for a spurious bridge count as SNR falls.
And early anaphase (aspect ~1.6) is classed interphase; the mask is fine
there, but the geometric baseline is already failing for those cells, so if
you want them scored separately lower `anaphase_aspect_min` to ~1.4.

Also added: two components of very different size are rejected as
`asymmetric_nuclei` (`min_daughter_area_ratio` = 0.35). Pombe divides
symmetrically, so a nucleus plus a bright speck is debris, a vacuole or a
merged cell mask, not two daughters; the original called it `two_nuclei` and
weighted it 4×.

### The better approach for the bridge label itself

Everything above makes the *classification* stable. The *pixels* of a bridge
label are still an Otsu accident, and the loss currently treats them as
certain. The clean fix is an ignore region: for bridge / borderline cells,
mark pixels between a low and a high threshold (e.g. 0.6·thr and 1.4·thr) as
"don't care", store it as a second target channel or a valid-mask, and mask
those pixels out of both the BCE and the Dice. The model then learns the
lobes from thousands of auto labels and the bridge only from hand-drawn ones.
Not implemented here because it changes the `Y` format and every consumer;
it is a half-day change and I would do it before scaling annotation.

If you have tracks (you will, for the smoothing the README already plans), a
monotonicity constraint along the track is a stronger stabiliser than
anything per-frame: interphase → bridge → two_nuclei never reverses, and any
frame that violates the order is borderline by definition.

## 11. Weighting — fixed (check + solver)

Reproduced your case: 2 manual bridge crops out of 100, weight 24 each, 22.5 %
of the mass, and under the sampler each is drawn **11× per epoch** with four
flip variants for augmentation. That is memorisation. `weight_report` now
prints `draws_per_sample` per group and warns above 3 draws or when the
weighted share exceeds 5× the count share. It also flags something you may
not have intended: with `anaphase_bridge: 6.0` and a mean weight ~1.4,
every *auto* bridge crop is already drawn ~3–4× per epoch before any manual
multiplier.

`suggest_manual_multiplier(meta, cfg, target_manual_share=0.15,
max_draws_per_sample=3)` solves for the multiplier that gives manual crops
the target share of the mass, then lowers it until no crop exceeds the draw
cap. The cap is what binds at realistic counts:

| manual / total | multiplier | manual share |
|---|---|---|
| 2 / 100 | 0.85 | 6 % |
| 50 / 20 000 | 0.9 | 0.7 % |
| 300 / 20 000 | 0.87 | 4 % |

That is the honest answer: a multiplier of 4 on top of a condition weight of
6 is 24 draws of the same crop; you cannot buy influence for a few crops
without memorising them. What actually moves the share is count. Practical
setting: keep `condition_weights` as they are, set `manual_weight_multiplier`
to ~1 until you have a few hundred annotated cells, then let the solver raise
it, and use the ignore region (above) so the auto bridge labels stop
competing with the manual ones on the pixels that matter.

## Things that are fine

* **Rotation.** `90 − degrees(orientation)` is correct for skimage ≥ 0.16's
  row-axis, counter-clockwise convention; a rod at any angle comes out with
  bbox 36 × 110 and orientation ±90°. `ndi.rotate` sorts the `axes` tuple,
  so the `(C, H, W)` channels and the 2-D masks rotate identically (max diff
  0.0). Round trip crop → rotate → un-rotate → paste: IoU 0.97–0.99, worst
  centroid offset 0.33 px over 40 random cells. The rotation centre is
  offset 0.5 px from the cell centroid, identically in both directions, so
  the round trip is exact up to nearest-neighbour resampling.
* **U-Net.** Instantiates and runs at depth 3/4, base 8/16, odd input sizes
  too. Channel arithmetic through the skips is right. 1.94 M parameters at
  the defaults, not ~1 M.
* **`evaluate_by_condition` join.** `meta.iloc[idx].reset_index(drop=True)`
  joined to a RangeIndex frame built in `idx` order is positionally correct,
  as long as `meta` and `X` are aligned; a length check now enforces that.
* **`WeightedRandomSampler` indexing.** Sampler indices map through
  `self.idx` to global rows correctly.
* `validate.py`, `synthetic.py`: nothing to report.

## Not done, deliberately

No plotting, no extra harness, no backbone change, no layout change. The
`__init__.py` in this commit is a stand-in that exports what `run_pipeline`
imports; keep yours.
