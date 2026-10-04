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
tone measured by the Observer probe. **Correction (2026-10-03):** this script does
NOT read with the steer off — STEER_LAYER == OBS_LAYER == 14 and the steering
pre-hook stays armed, so it runs before the capture pre-hook and the read is of the
STEERED activation. So this table is a dose-response of the steered representation
(the classic ActAdd read), not a steer-off read as the header used to say. Numbers
unchanged; the label was wrong. (step6/step7 deliberately disarm before reading, so
their numbers are NOT comparable to this table.)
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

> ### ❌ INVALIDATED (2026-10-03) — the treatment was diluted ~20x. See M9.
> The loop that produced the table below armed the steering hook ONCE per scale and
> then disarmed it on the first sample ("clear steer to measure honestly"), so only
> 1 of every (5 prompts x 4 samples) = 20 generations was actually steered. Measured
> offline by counting live-hook state: **2 of the 40 should-be-steered generations
> were steered** (`test_steering_collapse.py`, section D). A real effect of size e
> therefore reads as ~e/20 — right at the sham noise floor — which is exactly what
> the table shows. **These numbers are not evidence of a null, and the reading below
> (written before M9) is not supported by them.** The fix (`SteerController` in
> `steering_keys.py`) arms per generation and audits the count; this experiment must
> be re-run on GPU before any steering claim is made.
>
> What is NOT affected: §4.0 / M8. The cosine-collapse mechanism is computed from
> activations, labels and the probe — it involves no generation — so it stands.

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

⚠️ The interpretation below was written BEFORE M9 was found, and the numbers it
cites cannot support it. It is kept only as the record of what was concluded from a
diluted run; the reasoning about *mechanism* (the cos values) is still correct,
because those come from §4.0/M8 and not from generation.
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

## Step 7 — Steering SEARCH: layer sweep × token-pooling (the null GENERALIZES)  ⚠️

> ### ❌ INVALIDATED (2026-10-03) — same dilution bug as Step 6. See M9.
> `step7_sweep.py` had the identical arm-once-then-disarm loop, so the steered
> fraction there was 1/(2 prompts x 1 sample) in QUICK and 1/(5x4) in a full run.
> Both use `SteerController` now, and the audit printed at the end of each run states
> how many generations actually ran with the hook live. Re-run on GPU before citing
> any slope here. The READING half (probe accuracy, the length and label-permutation
> controls) is unaffected — it involves no generation. So the positive secondary
> result below, that last-token pooling reads better than mean pooling, still stands;
> only the steering null does not.

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
  held-out accuracy (>= 0.60) and adds direction estimators (mean / logistic /
  Fisher-LDA / probe-grad) plus a SHAM random direction at the same scale, because the
  Step 6 null mixed up "steering failed" with "measurement is noise". (Extended in M8
  to seven keys, including two non-point repairs of the probe gradient.)

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

## M8 — The steering-collapse mechanism (a proved bound on the method family)  ✅ NEW

Added 2026-10-02. This is the paper's spine: §4.0 of `paper.md`. It is **GPU-free** —
the identity is proved for the probe architecture and verified on CPU by
`test_steering_collapse.py`, and the cosine table is pure vector math on activations
the grid already captured. **Status after M9: unchanged.** Nothing here involves
generation, so the mechanism stands even though the steering null it was meant to
explain was withdrawn one day later. (The heading originally read "why the null is
structural, not accidental" — that causal reading is what M9 took away; what M8 proves
is the bound, not the consequence.)

**The identity.** For the measuring probe `f(x) = W₂·SiLU(W₁x + b₁) + b₂` (the exact
architecture in steps 5/6/7),

    ∇f(x) = W₁ᵀ · diag( SiLU′(W₁x + b₁) ) · W₂ᵀ   ⟹   ∇f(x) ∈ rowspace(W₁) ∀x.

Any key formed by **averaging** ∇f over a point set — one point (positive-pole mean),
a path (μ⁻→μ⁺), or the whole positive class — is `W₁ᵀ·diag(ḡ)·W₂ᵀ`. Averaging can only
re-weight the hidden units; it **cannot change where the key points**. So "probe-grad"
is a linear functional of `W₁` that happens to sit inside a nonlinear probe.

**Verified offline** (`test_steering_collapse.py`, CPU, no model — 17/17 checks pass):
- fraction of each probe key lying *outside* `rowspace(W₁)`: **0.000000** (all three
  variants), vs **0.331** for the plain `mean` key → the containment is specific to
  gradient keys and is not vacuous.
- pairwise cos among `probe-grad` / `probe-grad-path` / `probe-grad-class`:
  **0.9985 / 1.0000 / 0.9985** → path- and class-averaging do **not** escape the point
  key (the "you only tried a point gradient" objection is closed in advance).
- on a *nonlinear* synthetic boundary the probe key points well away from `mean`
  (**cos 0.392**, vs 0.840 on the linear rule) while still inside `rowspace(W₁)` →
  the collapse onto `mean` on real data is **contingent** (BCE aligning `W₁` with the
  class-separation direction), *not* a consequence of the theorem alone. Keeping these
  two claims separate is what makes the finding non-tautological.

**On the real model** (from the committed `results_step6*.txt`, layer 14 / mean pooling —
unchanged by the `steering_keys.py` refactor, which preserves the arithmetic exactly):

| Axis | probe acc | cos(logreg, mean) | cos(lda, mean) | cos(probe-grad, mean) | cos(sham, mean) |
|---|---:|---:|---:|---:|---:|
| Valence | 0.72 | +1.000 | +0.731 | +0.982 | +0.009 |
| Arousal | 0.81 | +1.000 | +0.704 | +0.964 | −0.016 |
| Care/Harm | 0.98 | +1.000 | +0.429 | +0.960 | −0.055 |
| Fairness | 0.66 | +1.000 | +0.493 | +0.845 | −0.030 |

Across the 24 step7 cells `cos(probe-grad, mean)` = 0.72–0.98. Two consequences:
1. **`logreg` is not an independent key** — `cos(logreg, mean) = +1.000` exactly on all
   four axes (features are L2-normalized per row, classes balanced, so the BCE optimum
   lies on the mean-difference direction). Counting "mean" and "logreg" as two
   directions double-counts one of them.
2. **The better the probe reads, the harder its gradient collapses.** Care/Harm (best
   read, 0.98) is the most confined (0.960); Fairness (worst read, 0.66) the least
   (0.845). A more readable axis is a more linearly-separable one, and a point gradient
   on a linearly-separable problem is a linear discriminant by construction.

**Fourth control trap found while building this** (extends the three in M7): seeding a
sham generator with the *same* seed the caller used for the ground-truth direction
reproduces that direction **exactly** (`cos = +1.000`) — i.e. a "random" control that
*is* the answer. `steering_keys.py` now uses a distinctive `DEFAULT_SHAM_SEED` and
`test_steering_collapse.py` asserts the sham stays orthogonal (≤ 0.20).

**What changed in the code:**
- new `steering_keys.py` — ONE definition of the key family (mean / logreg / lda /
  probe-grad / probe-grad-path / probe-grad-class / random), shared by step6, step7
  and the new test. `dir_mean`/`dir_probe_grad` were duplicated in step6 and step7
  before; that duplication is removed.
- `step6_steer_all.py` and `step7_sweep.py` now print the **full cosine table** for
  every axis / cell, not just `probe-grad`. Cost measured at the real hidden size
  (d=1536): **2.8 s per call → 11 s for step6's 4 axes, ~67 s for step7's 24 cells**,
  i.e. negligible, and it needs no generation at all.
- key names widened to 16 chars in step6's summary so `probe-grad-class` does not
  break the table alignment.

**What this means for the paper:** every probe-derived key is confined to the subspace
a probe never needed in order to *read* — which is precisely the gap between a readout
and a direction for control. That bound is the contribution, and it is proved rather
than observed. What it does **not** yet do is explain an observed null: M9 (below)
showed the null was produced by a loop that applied its treatment to 1 generation in
20, so the causal link between this bound and the steering result is untested until the
re-run. The scale question remains open too (see the TODO below), and is now a sharp
test rather than a fishing trip — but only after the re-run.

## M9 — The dilution trap: the steer was only applied to 1 in 20 generations  ❌✅
Found 2026-10-03 while wiring the `MESE_DRY` mode (`dry_model.py`), and it invalidates
the Step 6 / Step 7 steering nulls. Offline, CPU, no model:
`test_steering_collapse.py` section D.

**The bug.** The steering loop in `step6_steer_all.py` and `step7_sweep.py` was:

    for scale in SCALES:
        attach_steer(delta, scale)      # arm ONCE per scale
        for p in PROMPTS:
            for _ in range(SAMPLES):
                t = generate(p)         # <- steered ONLY on the first sample
                attach_steer(delta, 0.0)  # disarm, so the tone read is honest
                ...

`attach_steer(..., 0.0)` removes the hook and nothing re-arms it, so from the second
sample onward every generation was unsteered. With 5 prompts x 4 samples, **19 of
every 20 generations in the "steered" arm were actually steered at scale 0.**
Measured by counting live-hook state: **2 of the 40** generations that should have
been steered were steered, i.e. a **0.050x** dilution.

**Why it is invisible in the results.** It dilutes the TREATMENT, not the
measurement, so every control the study added still behaves correctly:

| Control | Does it catch this? |
|---|---|
| measuring-probe gate (grouped acc ≥ 0.60) | no — the readout is valid |
| SHAM random direction | no — the sham is diluted identically, so both look ~0 |
| permutation test on the grid (p=0.438) | no — real and sham stay indistinguishable |
| fluency check | no — fluency is fine, nothing was damaged |
| full cosine table (§4.0) | no — computed from activations, not generation |

The only thing that catches it is asking *"did the treatment actually get applied?"*
— a question none of those controls asks. A measured slope of `e/20` is
indistinguishable from a noise floor of ±0.05, so **a real effect up to ~20x the
noise floor would have been reported as a clean null.** The published numbers
(±0.02–0.10, sham ±0.04–0.06) are exactly consistent with that.

**The fix.** `SteerController` in `steering_keys.py` makes arming *per generation*
structural — `generate(prompt, vec, scale, layer, generate_fn)` arms, generates and
disarms (so the caller still gets the honest steer-off read for free) — and counts
`n_gen`, `n_should_steer`, `n_steered`. `check()` raises if any generation that
should have been steered was not. step6 and step7 now end with a printed
`[steering audit]` line. In `MESE_DRY` runs both report **100%** application:
step6 `72/72`, step7 `4/4` of the should-be-steered generations.

**Also fixed while here (same loop):** empty generations were being scored. A
generation that returns no text gave `tok("")` → a 1-token sequence → `fluency`
= `mean()` of an empty tensor = **nan** (seen as `fluency=nan` on the Care/Harm row),
and `tone` of a fallback 1-token vector is meaningless. Empty generations are now
skipped and counted, and `fluency` returns nan explicitly rather than a bogus value.

**Status of the claims after M9:**

| Claim | Status |
|---|---|
| §4.0 mechanism: probe keys lie in rowspace(W₁) | ✅ STANDS (no generation involved) |
| M8: those keys collapse onto `mean` on real activations | ✅ STANDS (no generation involved) |
| Observer / reading results (probe accuracy, controls) | ✅ STANDS (no generation involved) |
| last-token reads better than mean pooling | ✅ STANDS (no generation involved) |
| Step 6/7 steering NULL | ❌ INVALIDATED — must be re-run on GPU |
| "the null is structural" | 🟡 the *mechanism* is proven; that it *causes* the null is now untested |

The honest one-line summary: the mechanism is established, but the empirical null it
was supposed to explain has to be re-measured, because the experiment that produced
it never applied its own treatment.

## M10 — Device hygiene: a CPU generator cannot feed a CUDA draw  ✅ FIXED
Found by the **first real CUDA run** of the M9-fixed step6 (2026-10-03). This is the
bug the GPU run exists to find, and it is untestable locally.

**The crash:**
```
RuntimeError: Expected a 'cuda' device type for generator but found 'cpu'
  at steering_keys.py, in all_keys -> the "random" (sham) entry:
  torch.randn(base.shape, device=base.device,
              generator=torch.Generator().manual_seed(sham_seed))
```
`torch.Generator()` defaults to `cpu`; a CUDA draw cannot consume a CPU generator.

**Why no test caught it:** on CPU the identical line runs perfectly — so the local
suite was green. The bug only exists on CUDA, which is exactly the argument for not
trusting a green CPU suite with a GPU experiment. (Note also that the dict literal
evaluated `mean`, `logreg`, `lda`, `probe-grad`, `probe-grad-path` and
`probe-grad-class` *before* the sham, so all six real keys had already computed
successfully on CUDA. The failure was isolated to the control draw — the run got as far
as `all_keys`, past `make_probe` and `probe_cv_acc`, i.e. past the real work.)

**The fix:** `sham_vector(shape, seed, device)` draws on CPU then `.to(device)`. Two
benefits: it works on either device, and the same seed now yields the **byte-identical**
sham vector on CPU and CUDA — a control that is exactly reproducible rather than only
statistically so.

**A latent sibling, fixed at the same time:** the `MESE_DRY` signal injection in
step6/step7 built `_sig` from CPU tensors (`LAB` and `_dirs` are CPU) and added it to
`X` / `XMAT`, which live on `DEVICE`. On a CPU box `DEVICE=cpu` and it works; on a GPU
box `MESE_DRY=1` would raise a device mismatch — and the README explicitly encourages
running dry modes on Colab. Both now move the tensors with an explicit `.to(...)`.

**The guard (runs on CPU):** scenario E of `test_steering_collapse.py`.
- *runtime*: `sham_vector` matches a reference CPU draw exactly, is repeatable, and
  honours the requested device;
- *static*: an **AST** scan of the six GPU-reachable source files for
  `torch.randn(..., device=..., generator=...)` where the generator itself has no
  `device=`. AST rather than regex on purpose — the first regex attempt flagged the
  very *comments* that document the bug.

**Verified by reintroducing the bug:** with the original line restored, the test fails
(exit 1) and names the offending file and call — while the runtime path still succeeds
on CPU, which is the entire reason the static guard has to exist. Reverted afterwards;
all green.

**Status:** fixed, guarded, and it cost one GPU run rather than a whole paper.

## M11 — Instrument-gate protocol mismatch, and a RETRACTED dataset alarm  ✅ FIXED / ❌ RETRACTED

Two separate things, both surfaced while investigating why the step8 dose run kept
aborting on Valence.

**1. ❌ RETRACTED — the dataset was NOT broken.** An earlier revision of this
investigation reported that the Valence group key was degenerate: *"the easy set uses
the literal id `'easy'` for every row"*, 2 distinct group ids, folds
`[140, 36, 0, 0, 0]`. **All of that was wrong.** It was produced by reading the
**docstring of `grouping_is_informative`** — which describes a failure mode the guard
is *built to catch* — as if it were a measurement of the shipped data.

Measured directly against `dataset.py`:

| check | result |
|---|---|
| distinct ids in `GROUPS` | **146** |
| literal `'easy'` entries | **0** |
| per-axis groups | Valence **32**, Arousal 28, Care/Harm 58, Fairness 28 |
| no-signal control through the real grouping | **0.502**, **no empty folds** |

Grouping is sound on every axis and needs no fix. The docstring has been rewritten so
it can no longer be misread as a data claim, and the guard is kept as a tripwire
against a future dataset edit.

**2. ✅ FIXED — the gate certified a different probe than the one that measures.**
step8 aborted Valence with `held-out accuracy = 0.59 < gate 0.60`. That number came
from `probe_cv.cv_report(hidden=32, reps=3)` while the measuring instrument —
`make_probe`, and the MLP step5 actually reports — uses **hidden=64**. With reps=3 the
mean sits inside its own noise band.

The instrument quality per axis (step5, template-grouped, 10×5-fold) is:

| axis | grouped MLP | ± 2SE | clears 0.60? |
|---|---|---|---|
| Valence | 0.84 | ± 0.17 | barely — band spans it |
| Arousal | 0.72 | ± 0.15 | yes |
| **Care/Harm** | **0.98** | **± 0.00** | **comfortably** |
| Fairness | 0.79 | ± 0.03 | yes |

So the 0.59 was a **low draw from Valence's wide ±0.17 band**, amplified by a weaker
probe — not a broken dataset, and not evidence about steering. The gate now runs
`GATE_HIDDEN=64, GATE_REPS=10`, i.e. the **same protocol step5 reports**, so it can no
longer certify an instrument the experiment does not use.

**3. A second difference, deliberately KEPT.** step8 captures the residual stream
**leaving** layer 14 (`register_forward_hook`) — the same tap `step6`/`step7` steer
through. `step5_robust.py` uses `register_forward_pre_hook`, i.e. the stream *entering*
the layer, so its "layer 14" is **one hop shallower**. That is intentional (the gate must
certify the stream steering actually writes to) and is now documented at the capture
site, so a step5-vs-step8 accuracy gap is not misread as the instrument changing quality.

**Consequence for the steering claim.** Care/Harm is the only axis whose instrument is
comfortably clear of the gate, so `step8_dose.py` now **defaults to Care/Harm**. The
tone-shift signal remains **NOT CONFIRMABLE** until that run lands with
`MESE_SHAM=15` plus a second axis.

**Status:** grouping verified sound (alarm retracted); gate aligned to the measuring
probe; default axis moved to the strongest instrument. All four CPU suites still pass.

## M12 — Dose finding on Care/Harm: the decisive run, and it is a NULL  ⚠️ instrument valid, treatment audited

Run: `MESE_AXIS=Care/Harm MESE_SHAM=15 python step8_dose.py` (2026-10-04, Qwen2.5-1.5B-Instruct, layer 14).

**Everything upstream of the verdict held.**
- gate: template-grouped held-out **0.98 ± 0.00** (hidden=64, reps=10) -> instrument valid
- `cos(mean, random) = 0.013` -> the sham points somewhere unrelated
- `[steering audit] 2040/2550 hook LIVE` -> the treatment was actually applied
- dose: scale 4 = **7.08%**, 10 = 17.7%, 25 = 44.3%, 50 = 88.5% of `||h||` (median
  `||h|| = 56.50`). Scale 4 is above the sub-5% regime that forbids a no-effect claim,
  and the sham spread below shows the readout *is* perturbable at that dose.

**The result.** Paired per-sample slopes at scale 4 — the largest scale at which the
real key's mean fluency stays within `BREAK_FL` of scale 0 — with all 15 shams re-read
at that same scale:

| quantity | value |
|---|---|
| real `mean` key | **+0.214 ± 0.071** (z vs its own sampling error **+3.02**) |
| sham distribution (n=15) | mean **−0.074**, sd **0.266**, range [−0.689, +0.343] |
| real vs sham | z = **+1.08**; empirical **2/15** reach it (add-one **p = 0.188**) |

**Why the two tests disagree — and why that IS the finding.** The real key's slope is
*reliably* nonzero (z=3.02 against n=30 per-sample differences) yet *indistinguishable*
from an ordinary random direction (z=1.08, p=0.188). Within a fixed key the measurement
is precise; across keys a random direction is just as effective. The readout responds to
*perturbation magnitude*, not to the moral direction.

**The null does not depend on the window choice.** Real-key slope vs scale 0 at every
scale: **+0.214** (4), **−0.288** (10), **−0.067** (25), **+0.191** (50) — non-monotone,
sign flips twice. The 15-sham mean is ~0 at *both* scale 4 (−0.074) and scale 10 (+0.085)
with sd ~0.27, so the real key sits *inside* the random-direction spread at every dose,
not merely at the pinned one; widening the window to scale 10 pushes the real key
*further* into the null (its slope turns negative). Separately, the `random` key produced
|slope| = **0.227**, larger than the real key's 0.214.

**Status: NULL on the decisive axis, with a valid instrument and an audited treatment.**
This is NOT a claim that the model cannot be steered. It is a claim that *this key family*
adds nothing over a random direction of the same size — which is exactly what the
`rowspace(W₁)` mechanism (M8) predicts. The earlier suggestive Valence `+0.722` at
scale 10 did not merely fail to replicate here: on Care/Harm at scale 10 the real key's
slope is **negative**. Pending confirmation: a second axis (Arousal) run identically.

## TODO (before the final table in the paper)
- 🔴 **BLOCKING — RE-RUN THE STEERER (M9).** The Step 6 and Step 7 steering numbers
  were produced by a loop that steered only 1 generation in 20, so they are not
  evidence of anything. `SteerController` now arms per generation and prints a
  `[steering audit]` line that must read 100%:
  ```
  python step6_steer_all.py     # expect "[steering audit] N/N ... "
  MESE_SHAM=20 python step7_sweep.py
  ```
  Only after that can any statement be made about whether the probe-derived keys
  steer. The mechanism (§4.0/M8) does NOT need this re-run.
- ✅ DONE (was the old null, now INVALIDATED): probe-grad direction tested in step6;
  layer sweep + last-token search (step7). The *scripts* work and the reading half of
  step7 is valid, but their steering slopes must be discarded until re-run.
- ✅ DONE: full Observer grid captured (REPS=10, layers 12/14/16) — `results_step5 (1).txt`.
- 🟡 RUN WITH THE RE-RUN (closes the flags): re-check the flagged cells with a sham
  DISTRIBUTION, e.g. `MESE_LAYERS=14,26 MESE_POOL=last MESE_SHAM=20 python step7_sweep.py`
  (per-cell z-score > 3).
- 🟡 RUN NEXT (closes the last-token reading caveat): re-run the FULL `step7_sweep.py`
  (`python step7_sweep.py`) so its new per-cell length-only + label-permutation controls
  run on the model. Until then the last-token > mean reading advantage stays
  "suggestive". The mean-pooled controls have already run — see M6.
- Optional follow-up (not blocking): attention-weighted pooling could raise the
  linear probe above chance and is listed as future work, not a blocking gap.
- 🔵 **THE SCALE RUNG (the one experiment that decides "structural" vs "small-scale").**
  Run the identical protocol on a second, larger model at matched *relative* depth.
  `steering_keys.py` makes this cheap: the key-position cosine table (the §4.0 evidence)
  needs no generation, so `MESE_LAYERS=… MESE_AXES=0 python step7_sweep.py` on an 8B
  model already yields the mechanism check in minutes.
  - **Do NOT quantize to fit.** 4-bit changes the activation geometry, which is the
    object of study, so a quantized 8B would confound the very comparison we want.
  - **VRAM reality (bf16 weights):** ~3.1 GB (1.5B) / ~6.2 GB (3B) / ~16 GB (8B).
    A Colab **T4 (16 GB) fits 1.5B and 3B but NOT 8B** — `device_map="auto"` would
    silently offload 8B to CPU and wreck activation capture. Use **L4 (24 GB)** or
    A100 for the 8B rung; if only a T4 is available, run the **1.5B → 3B ladder**
    instead (weaker bracket, but unconfounded).
  - **Model + layers** (relative depth ~50%, matching the 1.5B's layer 14 of 28):
    1.5B (28 L) → `14,20,26` ✅ have · 3B (36 L) → `18,26,33` · Llama-3.1-8B (32 L) →
    `16,23,30` or Qwen2.5-7B (28 L) → `14,21,26`. Prefer a model Sun et al. used, for
    direct comparability.
  - **Per rung:** `step5_robust.py`, `step6_steer_all.py`, `step7_sweep.py`, plus
    `MESE_SHAM=20` so cells get a z-score vs a null *distribution* rather than one draw.
  - **Pass condition for the paper:** if the gradient keys stay confined (cos ≳ 0.9 with
    `mean`) at 3B/8B, the confinement is structural at every scale we can test; if they
    break away and steering starts to work, the boundary is a small-scale artifact and
    the mechanism needs a size term. Either outcome is a result.

## Current paper evidence summary (CORRECTED — final CUDA numbers are in)
0. **Mechanism (M8, the paper's spine):** every probe-derived steering key lies in
   `rowspace(W₁)` by construction, so mean/logreg/LDA/probe-grad/-path/-class cannot
   leave the subspace the probe used to *read*; on real activations the gradient keys
   collapse onto `mean` (cos 0.845–0.982) and `cos(logreg, mean) = +1.000` exactly.
   Verified offline to machine precision. The RQ3 null is therefore a *prediction* of
   the method family, not an unexplained failure — and the collapse's contingency is
   shown on synthetic nonlinear data (cos 0.392 there), so it is not a tautology.
1. Observer: tone is decodable from held-out templates (grouped MLP 0.78–0.98)
   and the readable structure is nonlinear (grouped linear ≈ chance, 0.46–0.66).
   Reported against a template-identity control; grouped CV is the honest protocol.
2. Nonlinear advantage: supported by grouped-CV across layers 12/14/16 (MLP
   0.78–0.98 vs linear 0.46–0.66). This is the paper's strongest RQ2 evidence.
3. Steerer: a clean, gated null on the decisive axis (**M12**). On Care/Harm — the
   strongest-reading axis (grouped **0.98**) — with the hook auditing at **100%** and a
   dose of **7.08%** of `||h||`, the real key's paired tone slope is **+0.214 ± 0.071**,
   while **15** independent random directions give mean **−0.074**, sd **0.266**, range
   [−0.689, +0.343]. The real key sits **+1.08 sd** from the sham mean; **2/15** shams
   reach it (add-one **p = 0.188**). The slope is reliably nonzero (z=+3.02 vs its own
   sampling error) but **not specific**: the readout responds to perturbation *magnitude*,
   not to the moral direction. The dose–response is non-monotone (+0.21, −0.29, −0.07,
   +0.19 at scales 4/10/25/50). *(The older ±0.02–0.10 step6/step7 numbers are WITHDRAWN
   — diluted treatment, M9.)*
4. Readout strength does not translate to steerability. Care/Harm is the most readable
   axis (grouped 0.98), yet the M12 dose run cannot separate its supervised key from a
   random direction of the same size. This is the paper's central negative claim, and it
   is what the `rowspace(W₁)` mechanism (M8) predicts: the key family varies only inside
   the subspace the readout already uses, so it carries no leverage the readout lacks.
5. Steering search (step7): the null generalizes over layers {14,20,26} × {mean,
   last-token} pooling (24 cells). Reading improves under last-token pooling
   (grouped probe 0.75–1.00). Every per-cell "win" is noise: real−sham +0.004
   (permutation p = 0.438) and sign flips between adjacent layers.
Caveats: single model (Qwen-1.5B); mean and last-token pooling, no attention
weighting; the M12 dose verdict is pinned to the largest fluency-intact scale
(4.0 = 7.08% of `||h||`), so the ladder's higher rungs diagnose fluency breakage
rather than steering — though the null holds at those rungs too; the search's
verdict originally used a single sham draw and this is now fixed (sham
DISTRIBUTION mode, n=15, exercised in M12); a second axis (Arousal) is pending;
no independent human rating yet.

