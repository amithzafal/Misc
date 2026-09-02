# pombe_nucseg

Nucleus segmentation from brightfield in *S. pombe* time-lapse, using
nuclear-GFP to generate training labels automatically.

Assumes you already have per-frame cell instance masks (your fine-tuned
Cellpose model). The cell mask does most of the work here: it removes the
instance-separation problem, lets every cell get its own GFP threshold, and
gives the cell–nucleus pairing for free.

```
pip install numpy scipy scikit-image pandas tifffile torch
python run_pipeline.py                  # synthetic data, checks the plumbing
python run_pipeline.py --train          # + trains a small U-Net
python run_pipeline.py --no-bf-contrast # null test (see below)
```

## Before you train anything

**Run step 2 of the pipeline first.** In pombe the interphase nucleus is
actively centred by microtubule pushing forces and its volume scales with cell
volume, so a circle at the cell centroid sized from the cell mask is already a
strong predictor — built entirely from information you have without any
brightfield nuclear signal. On synthetic data it scores IoU ≈ 0.65 on
interphase. Your model has to clearly beat that number, per condition, or it
has learned "put a blob in the middle" rather than anything about nuclei.

The conditions where the baseline necessarily fails — anaphase bridges and
two-nucleus cells (IoU ≈ 0.04 for the circle) — are where a model can actually
earn its keep. Score them separately. A headline average dominated by
interphase will look fine while the frames you care about are failing.

**Also run the null test.** `--no-bf-contrast` generates data where the
brightfield genuinely contains no nuclear information. Whatever the model
scores there is the floor from cell-shape priors alone. On real data the
equivalent is training on shuffled GFP–brightfield pairings.

Two acquisition choices raise the ceiling substantially: DIC instead of plain
brightfield (much better at 3–4 µm objects; phase contrast is worse because
halo dominates), and a small z-stack (nuclear contrast in transmitted light is
strongly defocus-dependent and near zero at true focus). Extra z-planes go in
as extra input channels — `build_crops` handles `(T, Z, H, W)` directly.

## Pipeline

| Module | What it does |
|---|---|
| `labelgen.py` | GFP + cell masks → nucleus labels, QC flags, condition class |
| `validate.py` | geometric baseline, per-cell IoU/Dice/centroid, stratified report |
| `dataset.py` | per-cell crops, orientation-normalised, sample weights, manual overrides |
| `model.py` | compact U-Net (~1M params) |
| `train.py` | weighted training, per-condition evaluation |
| `infer.py` | batched inference over a movie, annotation triage |
| `synthetic.py` | fake pombe frames for testing without data |

### Condition classes

Fission yeast has a **closed mitosis** — the nuclear envelope does not break
down, so NLS-GFP stays nuclear throughout. There is no cytoplasmic-signal frame
to filter out. What happens instead is that the nucleus elongates into a
dumbbell and pinches, so:

- `interphase` — one round nucleus
- `anaphase_bridge` — one elongated/waisted object mid-separation
- `two_nuclei` — two resolved nuclei
- `reject` — zero, too many, or failed QC

`anaphase_bridge` is where the auto-labels are least trustworthy: the threshold
decides where the bridge breaks, which is arbitrary. That is the argument for
hand-labelling those cases and weighting them up.

The bridge / two-nuclei call itself is no longer left to Otsu alone. Every
cell gets `nuc_neck_ratio`: the GFP intensity at the saddle between the two
brightest maxima divided by the dimmer maximum (1.0 = one blob, 0.0 = fully
separated), found by binary-searching the threshold at which the two maxima
join. It is threshold-free and, on synthetic dumbbells, varies smoothly with
bridge intensity (sd ≈ 0.01) while the Otsu component count flips under a
±20 % threshold change. Cells whose neck ratio falls in
`LabelGenConfig.neck_borderline` (default 0.25–0.6, where Otsu flips) are
marked `split_borderline`; two Otsu components with a neck ratio above the
band are called `anaphase_bridge`, because the threshold cut them, not
biology. Borderline auto labels are down-weighted in training
(`DatasetConfig.borderline_auto_weight`) and ranked first for annotation.

Bleed-through guards: a non-expressing cell next to a bright one passes the
SNR gate easily (haze from the neighbour is far above background noise) and
Otsu then draws a "nucleus" hugging the shared boundary — worse than no
label. Two extra gates reject that: `min_nuc_cyto_ratio` (nucleus vs rest of
cell contrast) and `max_outside_ratio` (signal just outside the cell next to
the nucleus, relative to the nucleus; a real nucleus sits ≥ 1 µm inside the
cortex so that ring is empty or at background). Rejections show up as
`low_nuc_contrast` / `edge_bleed`. Two nuclei of very different size are
rejected as `asymmetric_nuclei` — pombe divides symmetrically.

## Weighting manual and anaphase labels

```python
dcfg = DatasetConfig(
    condition_weights={"interphase": 1.0, "anaphase_bridge": 6.0, "two_nuclei": 4.0},
    manual_weight_multiplier=4.0,
    max_sample_weight=40.0,
)
meta = assign_weights(meta, dcfg)
print(weight_report(meta))
```

`weight = condition_weight × manual_multiplier`, capped; auto labels flagged
`split_borderline` get `borderline_auto_weight` instead of the multiplier.

**Always read `weight_report` before training.** It now prints
`draws_per_sample` — how often one crop of each group is drawn per epoch under
the sampler (equally, its relative loss weight under `"loss"`) — and warns
when that exceeds 3 or a group's weighted share is more than 5× its count
share. In a test run, 2 manually labelled anaphase crops out of 90 samples
(2.2% by count) took 22% of the weighted mass, i.e. each was drawn ~10× per
epoch with only flips for augmentation: memorisation. Do not guess the
multiplier, solve for it:

```python
from pombe_nucseg.dataset import suggest_manual_multiplier
suggest_manual_multiplier(meta, dcfg, target_manual_share=0.15, max_draws_per_sample=3)
```

gives the multiplier that puts manual crops at 15% of the weighted mass, then
lowers it until no single crop is drawn more than 3× per epoch. With a
handful of annotations the second cap binds and the share stays well below
target — that is the honest answer; only more annotation buys influence
safely. With ~300 manual cells in ~20k crops the target is reachable at a
multiplier of ~2–3.

`TrainConfig.weighting_mode` controls how weights enter training:

- `"sampler"` (default) — `WeightedRandomSampler` draws rare cases more often,
  every drawn sample contributes equally to the loss. Gradients stay well
  scaled.
- `"loss"` — uniform sampling, per-sample loss scaling. An epoch stays a true
  pass over the data, but a weight of 24 produces a correspondingly large step.
- `"both"` — effects multiply. Rarely what you want.

## Choosing what to annotate

```python
from pombe_nucseg.infer import rank_for_annotation
rank_for_annotation(table, baseline_metrics, n=200).to_csv("to_annotate.csv")
```

Ranks by: anaphase bridges, `split_borderline` calls, cells the pipeline
rejected despite good GFP signal (real failure modes), largest disagreement
with the geometric baseline. Pass `frame_stride` to thin consecutive frames of
the same dividing cell. Draw those in napari/Fiji, feed back through
`apply_manual_overrides(nuc_labels, table, manual, cell_labels=cells)` — pass
the cell masks so every auto pixel in the cell (including the second daughter,
which carries `cell_id + offset`) is cleared before the manual mask is written.
The condition is re-derived from the drawn mask. A few hundred well-chosen
cells beats a few thousand random ones.

## Plugging in real data

Replace `load_data` in `run_pipeline.py`:

```python
import tifffile
bf    = tifffile.imread("pos01_BF.tif")       # (T, H, W) or (T, Z, H, W)
gfp   = tifffile.imread("pos01_GFP.tif")      # (T, H, W)
cells = tifffile.imread("pos01_cellmasks.tif")# (T, H, W) int instance labels
```

Then set `LabelGenConfig(pixel_size_um=...)` to your actual calibration —
several defaults (minimum nuclear area, smoothing, erosion) are derived from it.

**Crop size is in pixels, not microns.** At 0.065 µm/px (6.5 µm sCMOS pixels
at 100×) a 14 µm cell is 215 px and does not fit a 128 px crop; anaphase
daughters sit at the tips, so their pixels would be missing from the label and
unpredictable at inference. `build_crops` warns and sets `meta["cell_clipped"]`
when a cell touches the crop border; set `crop_size` so that
`crop_size × pixel_size ≥ longest cell + 2 µm`.

At inference there is no GFP table: `predict_movie(model, bf, cells, None, dcfg)`
derives centroid and orientation from the cell masks.

### Things to check on real data that synthetic data cannot tell you

**Channel registration.** Your labels are only as good as the alignment.
Chromatic focus offset between transmitted light and GFP is real, and a
2-pixel systematic shift on a ~2 µm nucleus is a large fraction of the object.
Measure with TetraSpeck beads, fit an affine transform, apply it before
anything else. In microfluidics also check for cell movement between
sequential channel exposures.

**The z problem.** If GFP is a single plane, the mask you extract is a
cross-section whose apparent diameter depends on where the nucleus sits in z.
That varies cell to cell and is *not* inferable from brightfield, so it becomes
irreducible label noise. A short GFP stack with mid-plane or max-projection
extraction removes it. If you are stuck with single-plane, at least measure the
spread so you know your noise floor.

**Focus cutoff.** `LabelGenConfig.min_focus_score` is `None` by default because
the variance-of-Laplacian scale is detector-dependent. Step through focus once,
plot `focus_score`, pick a cutoff.

**A gold set.** Hand-annotate 50–100 nuclei from brightfield/DIC alone, without
looking at GFP, and check them against the GFP-derived labels. This tells you
whether your silver labels are systematically biased. If you personally cannot
place the nucleus in those crops, that is the same signal the null test gives
you, arriving from the other direction.

**Split by position, never at random.** Consecutive time-lapse frames of the
same cell are near-duplicates; a random crop split leaks them across the
boundary and inflates your score. Use `split_by_position(meta, ["pos07", "pos08"])`.

**NLS-GFP marks nucleoplasm**, so masks sit slightly inside the nuclear
envelope. Fine as long as you are consistent — just interpret downstream
measurements accordingly.

## Scaling

`predict_movie` batches per-cell crops on the GPU. At 128×128 with a ~1M-param
U-Net, hundreds of cells per frame is cheap. Since it is time-lapse, track
cells first and then smooth or propagate nuclear masks along tracks — this
catches the per-frame dropouts a frame-independent model will produce.

## Status

All modules have been run end to end on synthetic frames, including
`model`, `train` and `infer` (CPU torch 2.14). The U-Net at
`base_filters=16, depth=4` has 1.94 M parameters, not ~1 M. See `REVIEW.md`
for what was found and changed.
