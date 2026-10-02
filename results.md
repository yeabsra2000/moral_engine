# MESE-CR Results Log (Step 1–3)

## Step 2 — Pipeline feasibility (training-set accuracy)
All axes reached 100% (linear and nonlinear). Confirms the machinery works
end-to-end: capture -> probe -> read tone. Known limitation: evaluated on
training data (not generalized) + easy keyword-driven labels.

## Step 3 — Rigorous 5-fold cross-validation (held-out generalization)
Model: Qwen/Qwen2.5-1.5B-Instruct, layers 12/14/16, mean-pooled activations.
Dataset: 54 sentences (36 easy keyword / 18 hard context-only), balanced.
Chance = 50%.

| Layer | Axis       | n_active | hard | linear CV | MLP CV | MLP hard-only |
|------:|:-----------|:---------|-----:|----------:|-------:|--------------:|
| 12    | Valence    | 18 | 6 | 0.556 | 0.833 | 100% |
| 12    | Arousal    | 12 | 4 | 1.000 | 1.000 | 100% |
| 12    | Care/Harm  | 12 | 4 | 0.583 | 0.667 | 25%  |
| 12    | Fairness   | 12 | 4 | 0.667 | 0.750 | 75%  |
| 14    | Valence    | 18 | 6 | 0.556 | 0.778 | 100% |
| 14    | Arousal    | 12 | 4 | 1.000 | 1.000 | 100% |
| 14    | Care/Harm  | 12 | 4 | 0.500 | 0.667 | 25%  |
| 14    | Fairness   | 12 | 4 | 0.583 | 0.750 | 75%  |
| 16    | Valence    | 18 | 6 | 0.722 | 0.778 | 100% |
| 16    | Arousal    | 12 | 4 | 1.000 | 1.000 | 100% |
| 16    | Care/Harm  | 12 | 4 | 0.583 | 0.750 | 50%  |
| 16    | Fairness   | 12 | 4 | 0.583 | 0.667 | 50%  |

## Headline findings  ⚠️ SUPERSEDED — see M5 below
Random-split CV (as used here and in the first Step-5 run) is inflated by template
memorization: a control feature space holding ONLY the template identity reaches
0.76–0.93 on these same splits. Read the findings below as random-split numbers,
not as evidence of tone reading. Grouped CV + control is the honest protocol.

1. **Core claim supported (held-out):** a small (nonlinear) probe reads
   emotional/moral tone out of the model's internal activations with accuracy
   well above chance on data it never trained on (Valence/Arousal/Fairness
   0.75–1.00).
2. **Nonlinear "bending fabric" > linear "rigid hyperplane":** MLP beats or
   matches linear on 11/12 layer-axis cells; on Valence and Care/Harm the
   linear probe sits ~at chance while MLP works. Empirically supports the
   document's central architectural claim (pattern; needs larger N to confirm).
3. **Ceiling effect:** Arousal is 1.00 for both — labels too easy; weak axis.
4. **Negative finding:** Care/Harm hard-context cases fail (25–50%, <= chance);
   the probe infers care from keywords but not from context alone.

## Threats to validity (to fix when scaling)
- Hard-only subsets are only 4–6 sentences -> noisy percentages.
- 54 sentences total -> no statistical significance test yet.
- Single model, mean-pooling only, small hand-built label set.
- Steering (the Steerer half of MESE-CR) not yet measured.

## Step 4 — Steering (Steerer) dose-response curve
Method: ActAdd-style activation steering of Valence toward positive at layer 14;
tone measured by the Observer probe with steering switched OFF (honest measure).
3 natural prompts x 3 samples per scale. Chance = 0.0 (neither pole).

| Steering scale | Mean tone | std |
|---:|---:|---:|
| 0.0 | +0.26 | 0.80 |
| 1.5 | +0.79 | 0.17 |
| 4.0 | +0.93 | 0.07 |

Findings (INITIAL):
- Monotonic tone increase with steering strength (0.26 -> 0.79 -> 0.93).
- Variance collapses (0.80 -> 0.07): stronger steering consistently pins output positive.
- => The full observe->steer->observe loop works end-to-end on Qwen-1.5B.

## Step 6 — Multi-axis steering (all 4 axes) — FINAL NULL on 884 sentences  ⚠️
Method: measuring probe = template-grouped MLP per axis (the honest reader);
4 direction estimators (mean-difference / logistic / Fisher-LDA / probe-GRAD —
the validated measuring probe's own input gradient at the positive pole) + a SHAM
random direction, all at the same scales (0.0/1.5/4.0); tone = probe output with
the steer hook OFF (honest measure); fluency = mean-per-token logp.
Layer 14, mean-pooled activation addition. Full run: 5 prompts x 4 samples/scale.

Steering slopes (tone@4.0 − tone@0.0), tone∈[−1,+1], 0 = neutral:
(single self-consistent full run incl. the probe-grad direction)

| Axis      | reading | mean | logreg | lda | probe-grad | SHAM random | fluency |
|-----------|:-------:|:----:|:------:|:---:|:----------:|:-----------:|:-------:|
| Valence   | 0.72    | +0.05| +0.08  | 0.00| +0.02      | +0.05        | −2.5    |
| Arousal   | 0.81    | −0.04| +0.01  | −0.09| +0.00     | +0.05        | −2.5    |
| Care/Harm | 0.98    | −0.02| +0.02  | +0.10| −0.06      | +0.04        | −2.5    |
| Fairness  | 0.66    | +0.03| −0.01  | −0.01| −0.02      | +0.06        | −2.5    |

DEFINITIVE, and now a CLEAN null even under a supervised nonlinear key:
- Every axis passes the reading gate (0.66–0.98) → the measuring probe is valid,
  so "steering failed" and "measurement is noise" are SEPARATED. Adding any steer
  direction (scale up to 4.0) does not move tone beyond a random vector.
- **probe-grad (paper §8b-a) does NOT clear the sham** — slopes +0.02 / +0.00 /
  −0.06 / −0.02, all inside the sham envelope. Mechanism: cos(probe-grad, mean) =
  0.85–0.98, i.e. in mean-pooled L2-normalized activations the probe's own steepest
  ascent at the positive pole collapses ONTO the linear mean direction, so it
  steers exactly as (in)effectively as the linear keys. There is no steerable
  direction at layer 14 that survives residual-stream addition — not even the
  readout's own gradient.
- logreg ≡ mean (cos = 1.000), lda agrees (0.43–0.73), probe-grad ≈ mean
  (0.85–0.98): the 4 real directions are self-consistent, so the null is not a
  poor-direction artifact — it is a property of mean-pooled residual-stream
  addition at 1.5B at layer 14 (holding even for a supervised, nonlinear-derived
  key).
- Best real slope anywhere is +0.10 (LDA, Care/Harm) — nominally just above the
  ±0.06–0.09 sham noise floor it sits within (a random direction hit +0.13 in the
  quick run), so no axis shows a dose-response that clears the sham with
  confidence.
- Reading works; steering that reading does not ⇒ tone is nonlinearly decodable
  but not "rigid-hyperplane" steerable this way, even with a supervised key.
- Earlier Step-6 quick runs (Valence mean +0.12, n=2 prompts) were sub-sample
  noise; the full run's larger grid supersedes them.
- Fluency stays ≈ baseline (−2.5), i.e. steering does not damage English — it
  simply does not move tone.

(Historical Step-6 quick capture below is superseded by the full run above.)

| Axis      | tone@0.0 | tone@1.5 | tone@4.0 | fluency trend |
|-----------|----------|----------|----------|---------------|
| Valence   | -0.15    | +0.05    | -0.08    | -2.3 to -2.5 (ok) |
| Arousal   | +0.10    | +0.13    | +0.09    | -2.2 to -2.6 (ok) |
| Care/Harm | -0.10    | -0.08    | -0.08    | -2.3 to -2.6 (ok) |
| Fairness  | -0.09    | +0.06    | -0.13    | -2.5 to -2.8 (ok) |

## Step 7 — Steering SEARCH: layer sweep × token-pooling (the null GENERALIZES)  ✅

Script `step7_sweep.py`; log `results_step7.txt`; analyser `analyze_step7.py`.
Grid = layers {14, 20, 26} × pooling {mean, last} × 4 axes = 24 cells. Per cell we
train the measuring probe AT that layer/pooling (template-grouped CV gate), then
steer with mean / probe-grad / SHAM(random) at scales {0, 1.5, 4}. This tests paper
§8b levers (b) layer sweep and (c) last-token position.

**Reading improves under last-token pooling (positive secondary result):**
- measuring-probe grouped accuracy: mean-pooled 0.63–0.99, **last-token 0.75–1.00**.
  Care/Harm 0.98 → 0.99–1.00; Valence 0.72–0.88 → 0.99 at every layer. Held-out
  template generalization therefore gets *better* when you read the final token's
  residual instead of the mean — consistent across all three layers.
  ⚠️ **The last-token cells themselves are still uncontrolled for length.** The
  template-identity control cannot test this: one-hot(template) is held out wholesale by
  grouped CV, so it scores at chance for *every* pooling and cannot discriminate between
  poolings. The confound that matters here is sequence length (the final token's
  residual co-encodes where the sentence ended). `step7_sweep.py` computes, per cell, a
  **label-permutation** control (must collapse to ~0.50) and a **length-only** control
  (must stay well below the probe), and prints `CONFOUND WARNING` when length alone
  matches it; offline unit test `test_reading_controls.py` passes all 5 checks. **Those
  per-cell controls have not been re-run on the model**, so treat this contrast as
  *suggestive* until `step7_sweep.py` is re-run on Colab.
  ✅ The **mean-pooled** readings, by contrast, ARE now controlled on the model — same two
  controls, no warning fires (see "Reading controls on the model" in M6 below).

**Steering: no real effect at any layer or pooling.** The naive grid flags 11/24
cells as "beats the sham by > +0.10", but that is a multiple-comparison + sampling
noise artifact, shown by:

| statistic | value |
|---|---|
| mean(real − sham) over 48 (direction × cell) pairs | **+0.004**, permutation **p = 0.438** |
| sham slope spread across cells | −0.22 … +0.19 (sd 0.061 mean-pool / 0.093 last-pool) |
| real slope spread | −0.32 … +0.25 |
| sign consistency, `mean` slopes | 10 positive / 13 negative → **FLIPS** |
| sign consistency, `probe-grad` slopes | 15 positive / 8 negative → **FLIPS** |

Decisive example: probe-grad on Care/Harm, last-token = **+0.25 at L20** but
**−0.31 at L26** — same direction, same pooling, adjacent layers, opposite signs.
A genuinely steerable direction cannot flip sign between adjacent layers, and the
largest single |sham| (0.22) is as big as the largest real slope (0.25).

=> With 5 prompts × 4 samples = 20 generations/cell the per-cell noise floor is
≈ ±0.2, *larger* than the +0.10 verdict threshold; ~half the grid crossing a
threshold that tight is what chance produces. **step7 EXTENDS the step6 null:**
tone is readable but not steerable by residual-stream addition at layers 14/20/26,
under mean OR last-token pooling, with either a linear mean key or the supervised
probe-grad key. Fluency stays ≈ baseline (−2.5) in every cell.

**Protocol fix before any per-cell claim** (not yet run): replace the single sham
*draw* with a sham *distribution* (n ≥ 20 random directions → per-cell mean/sd →
z-score) and correct for the 24 comparisons. Only then can a cell be called an
effect. `step7_sweep.py` supports this via `MESE_SHAM=n`.

### Global test (the quotable number)
`analyze_step7.py` runs a permutation test on all real slopes vs all sham slopes:

```
n_real=48  n_sham=24
observed mean(real) - mean(sham) = +0.0042
permutation p (one-sided, 20000 shuffles) = 0.438
=> NO detectable steering effect anywhere in the grid
```

**p = 0.438** — one number that replaces the fragile per-cell flags.

### Run-to-run reproducibility: the flags are not stable (second full run)
A second run (`MESE_LAYERS=14,26 MESE_POOL=last`, same 8 last-token cells) is
decisive because layer 14 reproduces *bit-identically* (same seed → same RNG
stream) while layer 26 contradicts itself:

| cell | run 1 real (mean/probe-grad) | run 2 real (mean/probe-grad) | run 1 sham | run 2 sham | verdict 1 → 2 |
|---|---|---|---|---|---|
| L14 Arousal | −0.06 / +0.13 | −0.06 / +0.13 | −0.07 | −0.07 | flag → flag |
| L14 Care/Harm | −0.05 / −0.05 | −0.05 / −0.05 | +0.10 | +0.10 | null → null |
| L14 Fairness | +0.08 / −0.02 | +0.08 / −0.02 | −0.12 | −0.12 | flag → flag |
| L14 Valence | −0.08 / +0.14 | −0.08 / +0.14 | −0.03 | −0.03 | flag → flag |
| L26 Arousal | +0.14 / +0.05 | +0.10 / +0.06 | +0.08 | **−0.24** | null → **flag** |
| L26 Care/Harm | −0.06 / **−0.31** | +0.07 / **+0.24** | +0.07 | +0.00 | null → **flag** |
| L26 Fairness | +0.16 / +0.13 | **−0.03 / −0.13** | −0.01 | **−0.26** | flag → flag |
| L26 Valence | **−0.32** / +0.16 | **+0.12** / −0.07 | +0.01 | +0.04 | flag → **null** |

Three conclusions:
1. **Layer 14 is exactly reproducible** (identical to 2 dp), so the harness is
   deterministic; only the RNG state / generated text changes across runs.
2. **Layer 26 self-contradicts**: the same cell flips *null ↔ flag* between runs
   (L26 Valence), and probe-grad on Care/Harm swings −0.31 → +0.24 (Δ = 0.55).
3. **The sham is the unstable element.** On L26 Arousal the real slopes reproduced
   (+0.14/+0.05 → +0.10/+0.06) while the sham moved 0.32 (+0.08 → −0.24). In run 2,
   six of eight cells are flagged only because their *single* sham draw came out
   negative — e.g. **L26 Fairness is "flagged" with every real slope at or below
   zero** (−0.03, −0.13): an "effect" with no movement, in the wrong direction.

A permutation test on that 8-cell slice returns p = 0.027, but it is not trustworthy
precisely because the reference is one draw per cell (the same reason the flags are
unstable); with sign flips in *both* directions (4+/4− for mean and for probe-grad)
there is no consistent effect to report.




## M4 — Dataset scale-up (884 sentences) + data-quality findings
Sources: `dataset_templates.py` → easy.csv (740, magnitude-templated, polarity
guaranteed by construction) and `dataset_hard.py` → hard.csv (144 hand-authored
hard-context sentences: 18 mirrored pos/neg motif pairs per axis).
Merged by `dataset_merge.py` → dataset.py: 884 sentences, 146 template/motif
groups, per-axis active counts V 88/88, A 68/68, C 218/218, F 68/68 (balanced).

Hard-set audit (automatic, `leak_lexicon.py`): PASS — no leakage word from any
axis lexicon, every item a concrete event (determiner/name + event), 6–32 words
(mean 15.3), no duplicates, no overlap with the easy set, mirrored pairs.

Why the LLM drafts were abandoned: the 128 Qwen-drafted "neutral" scenarios were
dominated by keyword leakage ("relieved", "disappointment", "panic", "caring",
"gently"), abstract propositions ("a fair distribution is needed", "The decision
is difficult but necessary."), and fragments/duplicates ("my eyes close gently.").
`dataset_validate.py` now flags LEAK / ABSTRACT / FRAGMENT / DUP with the same
shared lexicon, and `dataset_generate.py` pre-filters drafts with it, so the
draft path is only a source of ideas, not of labels.

## M5 — Protocol validity: template memorization inflates random-split probes  ⚠️
`python test_grouped_cv.py` (CPU-only, no model) runs the SAME CV on a control
feature space that contains ONLY the template identity (+noise) and therefore no
tone information at all:

| Axis      | n   | random CV | grouped CV | gap   |
|-----------|----:|----------:|-----------:|------:|
| Valence   | 176 | 0.82      | 0.52–0.53  | +0.29 |
| Arousal   | 136 | 0.78      | 0.42–0.44  | +0.35 |
| Care/Harm | 436 | 0.93      | 0.54–0.55  | +0.38 |
| Fairness  | 136 | 0.76–0.77 | 0.45–0.46  | +0.31 |

Every cell is the envelope of **four re-runs** of `test_grouped_cv.py`, not a point value:
each drifts ~±0.01 between runs (torch CPU thread scheduling), so the second decimal is not
reproducible and an earlier revision of this table pinned it (grouped 0.54 / 0.44 / 0.55 /
0.46) as if it were. Quote the range. The conclusion is unaffected — template identity alone
is ~chance on unseen templates and 0.28–0.39 inflated under random splits.

Consequences for the write-up (this supersedes earlier claims):
- Random-split probe accuracy (as used in Step 3 and the first Step 5 run) is NOT
  comparable to chance=0.50 on this dataset: pure template identity already buys
  ~0.76–0.93. Any "probe reads tone" claim must be stated as the margin ABOVE the
  template-identity control, and grouped CV is the honest protocol.
- Step 5's corrected numbers (Arousal 0.93–0.96, Valence 0.68, Care/Fairness near
  chance) were themselves random-split, so even "Arousal robust" needs re-checking
  under grouped CV before it goes in the paper.
- `step5_robust.py` now prints the random row, the grouped row, and the
  template-identity control for every layer/axis, so this can't be forgotten.
- `step6_steer_all.py` now gates each axis on the measuring probe's grouped
  held-out accuracy (>= 0.60), tries four direction estimators (mean / logistic / Fisher-LDA / probe-grad) plus a SHAM random direction at the same scale, because the
  Step 6 null mixed up "steering failed" with "measurement is noise".

## M7 — Probe calibration: two traps found while adding reading controls  ⚠️✅
Found offline (`test_probe_calibration.py`, `test_reading_controls.py`; both ALL PASS,
CPU, no model). Both traps can silently fake a result.

**Trap 1 — a control probe with too little capacity passes vacuously.**
The length control was first built as a linear (`hidden=0`) probe on one standardized
token-count column. On synthetic data where length *fully* determines the label, that
control scored **0.61–0.77** while the same probe at the measured capacity
(`hidden=64`) scored **1.00** — a low-capacity probe sometimes cannot escape a
wrong-sign weight init in 400 Adam steps at lr=1e-3 on a 1-D input. A control that
cannot learn cannot flag a confound, so it would have "confirmed" the reading result
for free. Fixed: controls now always use the measured probe's capacity.

**Trap 2 — a single-seed control is noisy.** One CV pass on *shuffled* labels wandered
to **0.64** on a 100-row set, which would have triggered a spurious LEAK WARNING.
Controls are now averaged over 3 seeds (→ **0.55**), and the leak bar sits at 0.65,
above that noise floor. Same failure mode as the step-7 steering flags, one level down.

**Trap 3 — a control sitting far from 0.50 is not automatically a confound.** On the real
model the length control is *anti*-correlated with the label on some axes: Arousal
**0.27**, Valence **0.35**, both far from 0.50 — yet their classes have essentially
*identical* token length (14.4 vs 14.4; 9.5 vs 9.6). Nothing is learnable there, so the
deviation is a threshold that inverts on held-out templates, not signal. Had we treated
"control ≠ 0.50" as a diagnosis we would have invented a length confound on Arousal that
does not exist. Consequences: (a) read the control only through the decision rule that
matters — `length ≥ probe − 0.05`, i.e. *can length alone reproduce the reading?*; (b)
report the spread, which is why the control now prints mean ± 2·SE; (c) state the class
token means, which is what makes the direction interpretable. The single axis that does
separate by length is Care/Harm (12.5 vs 13.3 tokens; control **0.61 ±0.05** over 10
seeds vs a 0.98 probe), reproducible off-GPU with `check_length_control.py`.

**Calibration bonus (supports the nonlinearity claim).** On a synthetic set that is
linearly separable *by construction* (label = sign(x·w), d=64, n=120, grouped CV, our
exact optimizer), a linear probe earns only **0.75** and the MLP **0.78** — and 5× the
training only moves linear to 0.79, so this is estimation noise from d≫n, not
undertraining. The paper's grouped-linear accuracies (**0.46–0.66**) therefore sit
*below* what perfect linear structure would earn under this protocol, which is the
direction of evidence the "tone is not linearly recoverable here" claim needs. It also
means the linear-vs-MLP gap must be read as an *operational* comparison under fixed
training settings, not as a representational proof.

**Plumbing check.** `MESE_DRY=1 MESE_QUICK=1 python step5_robust.py` now runs the whole
Observer grid with **no model** (activations = seeded noise; own log
`results_step5_dry.txt`, deleted again afterwards). Measured: exit 0, **305 s** wall on
a laptop CPU, all four axes, **0** `CONFOUND WARNING` / `LEAK WARNING` lines across two
runs — length control 0.31–0.50 and label-permutation control 0.47–0.49 — i.e. the
controls stay silent on data with no signal (the leak branch never fires) and every new
code path executes, including the mean ± 2·SE print. Note the dry run is *not* instant
(it trains the full probe grid on noise); "seconds" would have been wrong. Values move in
the 3rd decimal between runs (torch CPU thread scheduling), so only 2 dp are stable.

**Portability fix (found by a failed Colab run).** `step5_robust.py` alone in a folder
used to die with two stacked `ModuleNotFoundError: No module named 'dataset'` tracebacks
— the `except ImportError` fallback re-raised the same opaque error and never said which
file was missing. Each experiment script now checks up front for `dataset.py`
(+ `probe_cv.py` where needed), adds its own folder to `sys.path`, and raises
`SystemExit` with the missing names and both candidate folders. Verified by running the
script alone (message + exit 1, no stray log), from a parent directory (resolves both
modules), and against a `dataset.py` without `GROUPS` (still a plain `ImportError`, so
the old fallback path is intact).

## M6 — Final CUDA numbers on the 884-sentence dataset  ✅
Source: `results_step5.txt` (Observer) + `results_step6.txt` (Steerer), layer 14.

### Observer — grouped (whole template held out), full 10x5-fold run (layers 12/14/16)
Grouped MLP accuracy per layer (mean ± 2·SE):

| Axis      | layer 12 | layer 14 | layer 16 |
|-----------|:--------:|:--------:|:--------:|
| Valence   | 0.83 ±0.07 | 0.82 ±0.06 | 0.86 ±0.06 |
| Arousal   | 0.84 ±0.07 | 0.83 ±0.07 | 0.85 ±0.07 |
| Care/Harm | 0.98 ±0.00 | 0.98 ±0.00 | 0.98 ±0.00 |
| Fairness  | 0.78 ±0.02 | 0.79 ±0.02 | 0.79 ±0.02 |

Grouped LINEAR accuracy (the "rigid hyperplane" comparator):

| Axis      | layer 12 | layer 14 | layer 16 |
|-----------|:--------:|:--------:|:--------:|
| Valence   | 0.55     | 0.54     | 0.55     |
| Arousal   | 0.46     | 0.46     | 0.46     |
| Care/Harm | 0.64     | 0.65     | 0.66     |
| Fairness  | 0.46     | 0.47     | 0.46     |

template-identity CONTROL (random split) — V 0.83, A 0.77, C 0.93, F 0.77.
random−grouped gap — V +0.08…+0.10, A +0.10, C −0.01, F +0.14…+0.15.

Interpretation (now stable, tight error bars, replicated across all three layers):
- **RQ1 ✔** — grouped MLP (0.78–0.98) is well above 0.50 chance for every axis on
  held-out templates and stable across layers 12/14/16. Care/Harm ≈ 0.98 at every
  layer. Tone IS genuinely decodable from mean-pooled activations, and NOT as
  template memorization (the control fails to generalize once a template is held
  out). The quick run's Arousal 0.72 was a small-sample dip; the full run settles
  grouped Arousal at 0.83–0.85.
- **RQ2 ✔ — nonlinear advantage is strong and clean.** Grouped LINEAR is at or
  below chance (0.46–0.66, Arousal/Fairness ≈ chance) while grouped MLP reads the
  same axes (0.78–0.98); Care/Harm linear (0.64–0.66) beats chance but stays far
  below its MLP (0.98). In mean-pooled activations the readable tone structure is
  NONLINEAR — the clearest "bending fabric > rigid hyperplane" evidence collected.
- random−grouped gap (+0.08…+0.15) and template-identity ctrl (0.77–0.93) confirm
  earlier random-split numbers were mostly template memorization (as M5 predicted).

### Reading controls on the model — length + label permutation  ✅ RUN (2026-09-30)
The two M7 reading controls are now wired into the Observer grid and have run on the
model (layer 14, QUICK run, REPS=3). The length column needs no model at all — only the
tokenizer — so it was re-derived offline at 10 seeds, exactly, with its spread measured.
The probe column below is the layer-14 row of the **full 10×5-fold** M6 table, so the
control is compared against the final probe number rather than a REPS=3 one:

| Axis      | grouped MLP (L14, 10×5-fold) | length-only (10 seeds)     | label-perm | flags |
|-----------|-----------------------------:|----------------------------|-----------:|-------|
| Valence   | 0.82                         | 0.35 ±0.05 (0.28–0.43)     | 0.51       | none  |
| Arousal   | 0.83                         | 0.27 ±0.07 (0.19–0.40)     | 0.44       | none  |
| Care/Harm | 0.98                         | **0.61 ±0.05 (0.55–0.68)** | 0.49       | none  |
| Fairness  | 0.79                         | 0.53 ±0.09 (0.40–0.68)     | 0.46       | none  |

The probe column is now the **layer-14 row of the M6 table** (`results_step5 (1).txt`,
10×5-fold) rather than a REPS=3 value, so the control and the thing it guards against come
from one saved, reproducible source. An earlier revision of this table quoted grouped
Valence as **0.92** — that is the *random-split* number from `results_step5.txt`; the
grouped number is 0.82, and comparing a control against a random-split figure would
re-import exactly the template memorization M5/M6 exist to remove.

The 10-seed column is `check_length_control.py` itself (tokenizer only, no model), quoted
to 2 dp. Four independent re-runs give means 0.34–0.36 / 0.27–0.28 / 0.60–0.61 / 0.53,
within-run **sd 0.04–0.05**, and the envelopes above; so the mean is stable to ±0.01 and
only the *extremes* of a 10-seed draw move (Care/Harm's min was 0.52 once and 0.55–0.56
three times). That is why the headline is the ±sd, not a quoted min–max. The REPS=3
in-run print is noisier, was never written to a log, and is therefore **not** cited here.
Raw per-run values, `python check_length_control.py`.

- **No CONFOUND or LEAK warning fires on any axis.** The decision rule is comparative —
  `length ≥ grouped − 0.05` — and Care/Harm's 0.61 sits 0.37 BELOW its 0.98 probe, so the
  mean-pooled readings are not a length/positional artifact. Shuffled labels collapse to
  0.44–0.51, i.e. no CV leakage.
- **⚠️ Do NOT read "far from 0.50" as "length confound": this control deviates in EITHER
  direction.** Arousal's length control is **0.27**, i.e. 0.23 *below* chance, even
  though its two classes have *identical* token length (14.4 vs 14.4 tokens) — with no
  marginal difference there is nothing to learn, so far-below-chance is a threshold that
  inverts on held-out templates, not signal. Same pattern for Valence (0.35 on 9.5 vs
  9.6 tokens). Only Care/Harm shows a genuine token-length separation at all (**12.5 vs
  13.3 tokens**, sd 1.5 vs 2.1 — a spread difference a threshold split can exploit), and
  its 0.61 is stable rather than one noisy fold (10 seeds, sd ≈0.05, 0.55–0.68 over four
  re-runs). So
  Care/Harm is the one axis where length-alone is not uninformative (0.61 vs a 0.98
  probe) — worth one sentence — while the verdict stays the probe comparison.
- **Internal check that the control is model-independent:** the template-identity control
  row is identical between the DRY (noise) and the real run — V 0.82 / A 0.77 / C 0.92 /
  F 0.76 in both — as it must be, since it is built from one-hot(template) + labels and
  never touches activations. That equality also shows the dry run exercises the
  production code path.
- The QUICK probe numbers reproduce the earlier QUICK run exactly (Arousal 0.72), so the
  new control rows are the only change.
- Still open: the SAME controls for the **last-token** cells (step7 sweep) — see the
  Step-7 caveat.

### Steering — full-run slopes (tone@4.0 − tone@0.0), sham-controlled
[Table and conclusion live in the Step-6 section above.]

### Bottom line for the paper
1. RQ1 ✔  tone is decodable from held-out text (grouped MLP 0.78–0.98 > chance > ctrl;
   stable across layers 12/14/16).
2. RQ2 ✔  strong nonlinear advantage: grouped linear ≈ chance (0.46–0.66),
   grouped MLP far above (0.78–0.98). The paper's strongest RQ2 evidence.
3. RQ3 ✖  linear (and probe-grad) steering at layer 14 does not move tone beyond a
   random sham envelope (−0.09…+0.06); all real slopes sit inside it (−0.06…+0.10),
   even when the measuring readout is demonstrably valid (gate 0.66–0.98). The
   probe-gradient key (the probe's own gradient at the positive pole) also fails.
4. RQ3-search ✖  the null GENERALIZES: layers {14,20,26} × {mean,last-token} × 4 axes
   (24 cells) — permutation p = 0.438 (mean real−sham = +0.004); the "winning" direction
   flips sign between adjacent layers (probe-grad Care/Harm last: +0.25 @ L20 vs
   −0.31 @ L26). BUT reading improves with last-token pooling (grouped probe
   0.75–1.00 vs 0.63–0.99 mean). See the Step-7 section.

## TODO (before the final table in the paper)
- ✅ DONE: full Observer grid captured (REPS=10, layers 12/14/16) — `results_step5 (1).txt`.
- ✅ DONE: probe-grad direction tested in step6 — also null (stronger null).
- ✅ DONE: layer sweep + last-token search (step7) — null generalizes; log
  `results_step7.txt`, noise analysis `analyze_step7.py`.
- 🟡 RUN NEXT (optional, closes the flags): re-check the flagged cells with a sham
  DISTRIBUTION, e.g. `MESE_LAYERS=14,26 MESE_POOL=last MESE_SHAM=20 python step7_sweep.py`
  (per-cell z-score > 3). Evidence so far says they are noise, so this is confirmation.
- 🟡 RUN NEXT (closes the last-token reading caveat): re-run the FULL `step7_sweep.py`
  (`python step7_sweep.py`) so its new per-cell length-only + label-permutation controls
  run on the model. Until then the last-token > mean reading advantage stays
  "suggestive". The mean-pooled controls have already run — see M6.
- Optional follow-up (not blocking): attention-weighted pooling could raise the
  linear probe above chance and is listed as future work, not a blocking gap.

## Current paper evidence summary (CORRECTED — final CUDA numbers are in)
1. Observer: tone is decodable from held-out templates (grouped MLP 0.78–0.98)
   and the readable structure is nonlinear (grouped linear ≈ chance, 0.46–0.66).
   Reported against a template-identity control; grouped CV is the honest protocol.
2. Nonlinear advantage: supported by grouped-CV across layers 12/14/16 (MLP
   0.78–0.98 vs linear 0.46–0.66). This is the paper's strongest RQ2 evidence.
3. Steerer: a clean, gated null — measurement valid (probe ≥ 0.60 on all axes), yet
   all direction estimators (mean / logreg / LDA / probe-grad) move tone only
   within −0.06…+0.10, no more than a random sham. The probe-grad key, the
   validated probe's own gradient at the positive pole, also fails. Report the
   null honestly; the protocol now separates "measurement noise" from "steering
   ineffective".
4. Care/Harm is the most readable axis (0.98), yet its best supervised slope
   (+0.10, LDA) barely exceeds the sham and its probe-grad key even goes negative
   (−0.06). This illustrates that readout strength does not translate to steerability,
   even under a supervised gradient key.
5. Steering search (step7): the null generalizes over layers {14,20,26} × {mean,
   last-token} pooling (24 cells). Reading improves under last-token pooling
   (grouped probe 0.75–1.00). Every per-cell "win" is noise: real−sham +0.004
   (permutation p = 0.438) and sign flips between adjacent layers.
Caveats: single model (Qwen-1.5B); mean and last-token pooling, no attention
weighting; steering scales ≤ 4.0; the search's verdict uses a single sham draw
(a sham-distribution mode, `MESE_SHAM=n`, is implemented for confirmation);
no independent human rating yet.

