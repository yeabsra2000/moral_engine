# Can We Read and Steer Moral/Emotional Tone from a Small LLM's Internal Activations?
### An empirical study of observer-probes and activation steering across four ethical/affective axes

**Working title (short):** *MESE-CR: Concept Probes + Activation Steering for Moral-Emotional Control in Small LLMs*

---

## Abstract (draft)
Large language models (LLMs) encode semantic and social abstractions in their
intermediate ("hidden") activations. We ask whether the emotional/moral position of
text can be (i) *read out* from internal activations with a small supervised probe, and
(ii) *manipulated* by adding a steering direction to the residual stream — and how a
nonlinear probe ("bending fabric") compares to a linear one.

Two methodological results drive the paper. First, **random train/test splits
overstate probe accuracy when the stimulus set is template-generated**: on our
884-sentence dataset, a control feature space containing *only the template
identity and no tone information at all* scores 0.76–0.93 under random 5-fold CV,
but falls to 0.42–0.55 under **template-grouped** CV (a whole template held out).
We therefore report grouped CV as the honest protocol. Second, a **Steerer null
that is now clean**: once the measuring probe is validated (held-out accuracy
0.66–0.98, comfortably above the 0.60 gate) and four direction estimators — including
the probe's own supervised gradient — are compared against a sham random vector at
the same scale, adding a direction to the residual stream moves tone only −0.06…+0.10,
indistinguishable from the sham (−0.09…+0.06). So "steering failed" and "measurement
is noise" are now separable, and this is a genuine *steering* null, not a measurement
artifact.

The central empirical findings are: **(i)** tone IS nonlinearly decodable from
mean-pooled activations with genuine held-out-template generalization (grouped
MLP 0.78–0.98 on all four axes across layers 12/14/16, Care/Harm ≈ 0.98);
**(ii)** the readable structure is **nonlinear** — grouped linear probes sit at or
below chance (0.46–0.66) while an MLP reads the same axes (0.78–0.98), the
cleanest evidence for our "bending fabric > rigid hyperplane" hypothesis; a
calibration control confirms this is not an undertraining artifact, because even a
*perfectly* linear rule earns only 0.75 under this protocol and 5× the training
changes that by +0.04;
**(iii)** but **reading ≠ steering** — the most readable axis (Care/Harm, 0.98) is
among the least steerable (best real slope +0.10, inside its own sham noise), and
even steering along the validated probe's own supervised gradient (probe-grad)
does not beat a random vector, showing that mean-pooled activation addition at
layer 14 of a 1.5B model does not transfer readable tone into controllable
generation at these scales.

*(Quantitative results below are the released scripts' real CUDA output on the
884-sentence dataset: the full Observer grid (REPS=10, layers 12/14/16) and the
full Steerer run (5 prompts × 4 samples × 3 scales; mean / logreg / LDA /
probe-grad + sham).)*

---

## 1. Introduction
- Motivation: external token filters / RLHF guardrails are evadable; an emerging
  alternative is to operate on the model's *internal activations* ("representation
  engineering").
- Two capabilities studied here: **Observer** (decode tone from activations) and
  **Steerer** (inject a direction to shift tone).
- Research questions (RQ):
  - RQ1: Can a small probe read the four moral/emotional axes from hidden states,
    generalizing to held-out text?
  - RQ2: Does a nonlinear ("bending fabric") probe beat a linear ("rigid hyperplane")
    probe? (Tests the central hypothesis of this work.)
  - RQ3: Can activation steering predictably shift output tone (dose-response)?

## 2. Related Work (cite correctly — do NOT claim these are new)
- Probing / concept bottleneck models: Alain & Bengio (2018, linear probes); Belinkov
  & Glass (2019) survey; Koh et al. (2020) Concept Bottleneck Models.
- Activation steering / representation engineering: Turner et al. (2023) Activation
  Addition (ActAdd); Zou et al. (2023) Representation Engineering; Rimsky et al. /
  recent steering literature.
- Affective axes: Russell (1980) Circumplex Model (Valence/Arousal).
- Morality: Haidt & Joseph (2004) Moral Foundations Theory (Care/Harm, Fairness/Cheating).

## 3. Method
- Model: Qwen/Qwen2.5-1.5B-Instruct; layers {12,14,16}; mean-pooled hidden states.
- **Dataset (M4): 884 sentences, generated deterministically and audited in code.**
  - `easy.csv` (740): filler-slot templates whose polarity is guaranteed by
    construction (keyword-driven). No human review needed.
  - `hard.csv` (144): hand-authored, *surface-neutral, concrete-event* sentences,
    18 mirrored pos/neg motif pairs per axis (36/axis). Written because the LLM
    drafts proved unusable (see below). Audited automatically for: one-hot label,
    no leakage word from any axis lexicon, concrete actor/object, 6–32 words, no
    duplicates, no overlap with the easy set.
  - Everything passes one gate (`leak_lexicon.py` → `dataset_merge.py`) before it
    reaches `dataset.py`, so "hard-context" is a checkable property, not a claim.
- **Data-quality finding (why the drafts were dropped):** Qwen-drafted
  "neutral" scenarios (128 candidates) were dominated by (i) keyword leakage
  ("relieved", "panic", "caring", "gently"), (ii) abstract propositions
  ("a fair distribution is needed"), and (iii) fragments/duplicates. The shared
  gate flags all three; the released hard set is therefore authored, not sampled.
- Observer: linear probe vs nonlinear MLP; **two split protocols** —
  RANDOM (sentences independent) and GROUPED (template/motif held out, requires
  `GROUPS` from `dataset.py`) — plus a **template-identity control** feature space
  (one-hot of the template key + noise, no tone information) evaluated under the
  same random split, to quantify template memorization.
- **Reading controls** (run per axis/layer/pooling in `step5_robust.py` and
  `step7_sweep.py`): a **label-permutation** control (identical grouped CV with
  shuffled labels, which must collapse to ~0.50; averaged over 3 seeds because a
  single pass wanders to ~0.64) and a **length-only** control (the same probe
  capacity trained on standardized token count). If length alone matches the probe,
  the cell is reported as a possible length/positional artifact rather than tone.
  Template identity cannot serve as a control for a *grouped* accuracy: it is held
  out wholesale and therefore sits at chance for every pooling.
- **Probe calibration** (`test_probe_calibration.py`, offline): on a synthetic set
  that is linearly separable *by construction* (label = sign(x·w)), under this exact
  protocol and optimizer a linear probe reaches only 0.75 and the MLP 0.78, and 5x
  the training moves linear by +0.04 — so linear-vs-MLP here is an operational
  comparison under fixed training settings, not a representational proof, and control
  probes must use the measured probe's capacity or they pass vacuously.
- Steerer: direction estimated three ways — `mean` (mean_pos − mean_neg),
  `logreg` (fitted logistic weights), `lda` (Fisher: (Cov+λI)⁻¹Δμ) — each compared
  against a **sham `random` unit direction** at the same scale; added at layer 14
  at scale in {0.0, 1.5, 4.0}; tone measured by the probe *with the steer hook
  removed*; fluency = mean per-token log-likelihood. Axes whose measuring probe
  fails a held-out accuracy gate (0.60) are reported as NOT MEASURABLE rather
  than as a steering failure.

## 4. Results
- **RQ0/M5 (protocol validity, measured offline — no GPU needed).** Same CV, same
  labels, features = template identity only (zero tone information):

  | Axis | n | random CV | grouped CV | gap |
  |---|---:|---:|---:|---:|
  | Valence | 176 | 0.82 | 0.52–0.53 | +0.29 |
  | Arousal | 136 | 0.78 | 0.42–0.44 | +0.35 |
  | Care/Harm | 436 | 0.93 | 0.54–0.55 | +0.38 |
  | Fairness | 136 | 0.76–0.77 | 0.45–0.46 | +0.31 |

  Cells are the envelope of four re-runs (each drifts ~±0.01 between runs), not a
  pinned point value. Interpretation: under random splits, template identity alone
  already explains
  most of the accuracy that the earlier 54-sentence study attributed to tone
  reading. Random-split probe numbers are therefore only meaningful *relative to
  this control*; grouped CV is the honest protocol.
- **RQ1 (Observer, grouped CV, full 10x5-fold across layers 12/14/16):** tone IS
  decodable from held-out templates, well above chance on all four axes:

  | Axis | grouped MLP (per layer) | grouped linear | template-ident ctrl |
  |---|---|---:|---:|---|
  | Valence | 0.83 / 0.82 / 0.86 | 0.55 / 0.54 / 0.55 | 0.83 |
  | Arousal | 0.84 / 0.83 / 0.85 | 0.46 / 0.46 / 0.46 | 0.77 |
  | Care/Harm | 0.98 / 0.98 / 0.98 | 0.64 / 0.65 / 0.66 | 0.93 |
  | Fairness | 0.78 / 0.79 / 0.79 | 0.46 / 0.47 / 0.46 | 0.77 |

  (per-layer cells are layer 12 / layer 14 / layer 16; MLP ±2·SE is ±0.00–0.07.)

  Grouped MLP (0.78–0.98) is far above the 0.50 chance level once a whole template
  is held out, and the result is stable across all three layers. The
  template-identity control fails exactly there (it cannot help on a held-out
  template), so the grouped number is genuine generalization, not memorization;
  random−grouped gaps of +0.08…+0.15 confirm the random-split inflation M5
  predicted. Care/Harm is the most readable (0.98 at every layer).
- **RQ2 (bending vs flat) — nonlinear advantage is strong and clean.** Grouped
  LINEAR is at or below chance (0.46–0.66; Arousal/Fairness ≈ chance) while grouped
  MLP reads the same axes (0.78–0.98). Even the best-linear axis (Care/Harm,
  0.64–0.66) sits far under its MLP (0.98). In mean-pooled activations the
  readable tone structure is NONLINEAR — the first grouped-CV evidence that the
  nonlinear probe exploits structure a single hyperplane cannot.
  The obvious objection — that the linear probe is merely undertrained — is ruled out
  by calibration (`test_probe_calibration.py`): on synthetic data that is linearly
  separable *by construction*, this protocol and optimizer give a linear probe only
  0.75 and an MLP 0.78, and 5× the training moves linear by just +0.04. The linear
  shortfall is therefore a property of the d≫n estimation regime, not of the training
  budget, and 0.46–0.66 lies *below* what perfect linear structure would earn here.
- **RQ3 (steering) — a clean, gated NULL, now under a supervised key too.** Full run:
  every axis passes the measuring-probe gate (grouped acc 0.66–0.98), so the readout
  is valid; yet adding any of the four direction estimators at scale up to 4.0 moves
  tone only within −0.06…+0.10, no more than the SHAM random direction (−0.09…+0.06):

  | Axis | read acc | mean sl | logreg sl | lda sl | probe-grad sl | SHAM random sl |
  |---|---:|---:|---:|---:|---:|---:|
  | Valence | 0.72 | +0.05 | +0.08 | 0.00 | +0.02 | +0.05 |
  | Arousal | 0.81 | −0.04 | +0.01 | −0.09 | +0.00 | +0.05 |
  | Care/Harm | 0.98 | −0.02 | +0.02 | +0.10 | −0.06 | +0.04 |
  | Fairness | 0.66 | +0.03 | −0.01 | −0.01 | −0.02 | +0.06 |

  The four real direction estimators agree (logreg ≡ mean, cos = 1.000; LDA
  0.43–0.73; probe-grad 0.85–0.98 of mean), so the null is not a bad-direction
  artifact. Notably the **probe-gradient** key (the validated probe's own input
  gradient at the positive pole — paper §8b item a) also fails: in mean-pooled
  L2-normalized activations its steepest ascent collapses onto the linear mean
  direction, so it steers as (in)effectively as the linear keys. Best real slope is
  +0.10 (LDA, Care/Harm), inside the sham noise floor (a random direction reached
  +0.13). Fluency stays ≈ baseline (−2.5): steering does not damage English, it
  simply fails to move tone. Because the measuring probe is validated, this null is
  a genuine *steering* null (not "measurement is noise").
- **RQ3 (search) — the null survives a layer sweep and last-token steering.** A
  follow-up grid (`step7_sweep.py`) searched layers {14, 20, 26} × pooling
  {mean, last-token} × 4 axes = 24 cells, each with its own same-layer probe, gate
  and sham. Two findings:

  | check | result |
  |---|---|
  | reading, mean pooling (grouped) | 0.63–0.99 |
  | reading, **last-token** pooling (grouped) | 0.75–1.00 (Care/Harm 0.99–1.00; Valence 0.99) |
  | mean(real − sham) over 48 (direction × cell) pairs | **+0.004**, permutation **p = 0.438** |
  | sham slope spread across cells | −0.22 … +0.19 |
  | sign consistency (mean / probe-grad slopes) | 10 / 13 and 15 / 8 → **flips** |

  **Caveat on the reading contrast (last-token cells still uncontrolled).** The
template-identity control used elsewhere cannot validate this number: one-hot(template)
is held out *wholesale* by grouped CV, so it sits at chance for *every* pooling and
cannot discriminate between them. The confound that matters for last-token pooling is
**sequence length** — the final token's residual co-encodes where the sentence ended.
`step7_sweep.py` runs two controls per cell (label-permutation, which must collapse to
~0.50; length-only, which must stay well below the probe) and prints a CONFOUND WARNING
when length alone matches the probe. Both are unit-tested offline
(`test_reading_controls.py`, all pass; a single-seed permutation control wanders to
~0.64, so controls are averaged over 3 seeds). The same two controls now also run on the
**mean-pooled headline readings** and **have run on the model** (`step5_robust.py`:
length-only 0.27 / 0.35 / 0.53 / 0.61 on A/V/F/C over 10 seeds, permutation 0.44–0.51, no
warning fires). The **per-cell step7
controls, however, have not been re-run**, so the last-token > mean contrast remains
*suggestive*, not controlled.

  Reading improves under last-token pooling (a positive secondary result), but
  steering stays null: the real-direction slope distribution is statistically
  indistinguishable from the sham — a permutation test over the whole grid gives
  **p = 0.438** (observed mean(real) − mean(sham) = +0.004), and the direction that
  "wins" flips sign across cells (`mean` 10+/13−, `probe-grad` 15+/8−). The prime
  example is probe-grad on Care/Harm (last-token): **+0.25 at layer 20** yet
  **−0.31 at layer 26**. A second run of the same 8 last-token cells reproduces
  layer 14 *bit-identically* but contradicts itself at layer 26 (the same cell flips
  from "flag" to "null", and probe-grad on Care/Harm swings −0.31 → +0.24); on one
  cell the real slopes reproduced while the *sham* moved 0.32, isolating the
  instability to the single-draw control. The naive grid flags 11/24 cells, but the
  per-cell noise floor there (±0.2, from only 20 generations/cell) exceeds the +0.10
  flag threshold, so those flags are expected under the null. The search therefore
  **generalizes the null**: not steerable at layers 14/20/26, under mean or
  last-token pooling, with a linear *or* supervised key.
- **Readable ≠ steerable (concrete illustration):** Care/Harm is the most readable
  axis (0.98) yet its best supervised slope is +0.10 (LDA), still inside the sham envelope,
  while its probe-grad key yields −0.06. The probe's own gradient clusters with the
  directions that fail to move tone.
- Negative result: Care/Harm hard-context cases remain hard (the probe reads care
  from keywords, not from context alone) — our earlier keyword-dependence finding
  is superseded only in degree, not in kind.

## 5. Discussion
- **Tone is readable, and now defensibly so.** Grouped CV (whole template held out)
  gives 0.78–0.98 across all four axes, stable across layers 12/14/16 — far above
  0.50 chance and unaffected by the template-identity control that invalidates
  random splits (which buy 0.76–0.93 from template identity alone). This is the
  first grouped-CV, control-aware evidence that a 1.5B model's mean-pooled
  activations genuinely encode moral/affective tone that generalizes to unseen
  templates.
- **The readable structure is nonlinear.** Grouped linear probes sit at or below
  chance (0.46–0.66); an MLP reads the same axes (0.78–0.98). At the mean-pooled
  layer, tone is not a single linear direction — a nuanced result that qualifies
  "rigid hyperplane" claims and supports the "bending fabric" framing, without
  over-claiming: linear probes with richer (token-level / multi-layer) features
  might do better. The comparison is *operational* — both probes share one fixed
  training budget, and calibration shows even a perfectly linear rule only reaches
  0.75 in this d≫n, few-template regime — so it establishes that an MLP extracts
  signal a linear probe cannot *under these settings*, not that no linear readout
  of any kind exists.
- **Measurement before intervention — and now a clean steering null.** Because the
  measuring probe is validated on every axis (0.66–0.98) and all four direction
  estimators agree — including a **probe-gradient** key derived from the readout's
  own nonlinear manifold — yet none beats a sham random vector, the steering null
  is a genuine property of mean-pooled activation addition at layer 14 (scale
  ≤ 4.0) on a 1.5B model — not a measurement artifact. This lets us say something
  precise that the earlier results could not: *the axis is readable but not
  steerable this way* — even with a supervised, nonlinear-derived steering key.
- **Readable ≠ steerable.** Care/Harm is the most readable axis (grouped 0.98) yet
  barely moves under its own steering direction (best supervised slope +0.10,
  probe-grad −0.06). Whatever representation
  the nonlinear probe exploits is not a direction that remains in the residual
  stream when added at scale — a concrete, citable illustration.
- **Care/Harm hard failure (qualitative):** intent/context-conditional morality is
  hard to decode from mean pooling alone; the probe still leans on surface cues
  (R4 non-replication) → future work (token-level or attention-weighted pooling).

## 6. Limitations
- The easy set is template-generated; grouped CV is reported precisely because the
  random-split numbers are inflated. The hard set is small (144) and authored by
  us, so it carries author bias — partially mitigated by mirrored pos/neg pairs
  and an automatic leakage/concreteness audit.
- Single model; mean pooling plus last-token pooling; Observer layers {12,14,16},
  steering searched at {14,20,26}; no attention-weighted or token-level analysis.
- The last-token *reading* advantage (0.75–1.00 vs 0.63–0.99 mean-pooled) is not yet
  controlled for sequence length, which last-token residual co-encodes; the
  template-identity control cannot test this (it is held out wholesale, so it sits at
  chance for every pooling under grouped CV). Length-only and label-permutation
  controls now run automatically for both the mean-pooled grid (`step5_robust.py`) and
  the pooling sweep (`step7_sweep.py`) and are unit-tested offline against synthetic
  ground truth. The **mean-pooled** controls have now run on the model (length-only
  0.27 / 0.35 / 0.53 / 0.61 on Arousal/Valence/Fairness/Care-Harm over 10 seeds,
  permutation 0.44–0.51, no warning fires), so the headline Observer numbers
  carry them; the **per-cell sweep** controls have not run, so the last-token contrast
  is still reported as suggestive, not established. On the model, the length control is
  silent for three axes and only measurable for Care/Harm (0.61 ±0.05 over 10 seeds,
  against a 0.98 probe). Note that this control deviates in *either* direction — Arousal
  scores 0.27 with *identical* class token means — so the probe comparison, not the
  distance from 0.50, is what licenses a verdict. Building the controls
  exposed a trap worth stating: a
  control probe with *less* capacity than the measured one passes vacuously (a
  1-column linear control still scored ~0.6–0.8 when length fully determined the
  label), and a *single-seed* permutation control is noisy enough to fake a leak
  (0.64 on one draw), so both controls now match the measured capacity and average
  three seeds.
- Care/Harm is the one axis where linear grouped accuracy clears chance, so the
  qualitative claim of a global linear failure is weakest there — the nonlinear
  advantage is still large (linear ~0.65 vs MLP 0.98).
- Steering search caveat: with 20 generations/cell the per-cell noise floor is
  ±0.2, so a 24-cell grid scored against a *single* sham draw flags ~half the cells
  by chance. We therefore report the permutation test over the whole grid
  (p = 0.438) and a
  sign-consistency check rather than any per-cell "win"; a sham *distribution*
  (`MESE_SHAM=n`, z > 3) is implemented and is the correct protocol for future
  per-cell claims. A null still does not rule out attention-weighted keys or
  scales above 4.0.
- Label polarity for the hard set is entailed by construction rather than
  validated by independent human raters (a planned extension).

## 7. Ethics
- Activation steering can be used for alignment AND for jailbreaking. We frame this as
  alignment research and discuss dual-use. Steering is not a complete safety mechanism.

## 8. Conclusion & Next Steps
- **Conclusion.** On an 884-sentence, audit-gated dataset, a 1.5B model's
  mean-pooled activations encode moral/affective tone with genuine held-out-
  template generalization (grouped MLP 0.78–0.98, stable layers 12/14/16), and
  that readable structure is nonlinear (grouped linear 0.46–0.66 ≈ chance). Yet
  activation steering at layer 14 does not transfer that readable tone into
  generation: every direction estimator (mean / logistic / LDA / probe-grad of the
  validated probe) moves tone −0.06…+0.10, no more than a random sham, despite a
  validated measuring readout. A 24-cell search over layers {14,20,26} × {mean,
  last-token} pooling reproduces the null (permutation p = 0.438; the "winning"
  direction flips sign across adjacent layers), while last-token pooling *improves
  reading* (probe 0.99–1.00). Reading works; steering this way does not — a precise,
  control-aware, and now search-tested null.
- **Next:** (1) ✅ observer grid captured; (2) ✅ probe-gradient key (step6) and
  ✅ layer sweep + last-token (step7) — all null; (3) ✅ sham-distribution protocol
  (`MESE_SHAM=n`) implemented — run it on the flagged cells to close them out;
  (4) independent human rating of a hard-set sample; (5) a second model for
  generalization; (6) larger scales (> 4.0) with a fluency cost check.

## 8b. Concrete next experiment (why the steering null is worth pursuing)
The null is an *absence of effect under a narrow method*, not evidence that tone is
unsteerable. Because the readout is validated and nonlinear, a plausible cause is
that tone lives in a *nonlinear* region of activation space that a single added
mean-direction at one layer cannot reach. We have now tested the first three
follow-ups — (a) the **probe-gradient** key, (b) a **layer sweep** {14, 20, 26},
and (c) **last-token** steering — and all three fail:

(a) the readout's own steepest ascent at the positive pole does not beat the sham,
because in mean-pooled L2-normalized activations it collapses onto the linear mean
direction (cos 0.85–0.98);
(b)/(c) `step7_sweep.py` searches layers × {mean, last-token} pooling × axis with a
per-cell probe, gate and sham. Reading improves (last-token probe 0.99–1.00), but
the real−sham difference is +0.004 (permutation p = 0.438) and the direction that
"wins" flips sign between adjacent layers (probe-grad Care/Harm: +0.25 @ L20 vs
−0.31 @ L26). The null *generalizes* rather than breaks.

One methodological caveat surfaced and is now handled in code: a **single** sham
draw makes a > +0.10 flag threshold noise-sensitive (the per-cell noise floor at
20 generations/cell is ±0.2; a 24-cell grid then flags ~half the cells by chance).
`step7_sweep.py` therefore supports `MESE_SHAM=n` to build a sham **distribution**
and score each cell by a z-score (> 3) instead of one draw; `analyze_step7.py`
reports the noise statistics. The remaining, still-untested lever:
(d) scale up past 4.0 with a fluency cost check.

## Appendix: Reproducibility
- Data pipeline (no GPU needed): `dataset_templates.py` → `easy.csv`;
  `dataset_hard.py` → `hard.csv` (+ audit); `dataset_merge.py` → `dataset.py`
  (`DATA` + `GROUPS`); `dataset_generate.py` / `dataset_validate.py` for LLM drafts.
- Local tests (CPU only, no model): `python test_grouped_cv.py` verifies the fold
  invariants and measures the template-memorization gap; `python
  test_probe_calibration.py` shows that a low-capacity control probe passes vacuously
  and calibrates what a linear probe can earn under this protocol (0.75 on a set that
  is linearly separable by construction); `python test_reading_controls.py` checks the
  label-permutation and length-only controls against synthetic ground truth.
  `probe_cv.py` holds the shared, model-free CV logic.
- GPU experiments (Google Colab T4): `pilot.py`, `step2_probe.py`,
  `step3_rigorous.py`, `step4_steer.py`, `step5_robust.py`, `step6_steer_all.py`,
  `step7_sweep.py`.
- Noise analysis of the steering search (CPU only): `python analyze_step7.py`
  parses `results_step7.txt` and prints the real-vs-sham distribution statistics,
  the sign-consistency check, and the sham noise floor.
- Logs and corrected findings: `results.md` (`results_step5*.txt`,
  `results_step6*.txt`, `results_step7.txt`).
- Cost control: `MESE_DRY=1 python step5_robust.py` runs the entire Observer analysis
  with **no model** (activations replaced by seeded noise) in ~5 min on a laptop CPU
  (305 s measured, QUICK mode), so wiring errors are caught before GPU time is spent.
  It logs to `results_step5_dry.txt` (never the real log) and, on noise, must print no
  `CONFOUND WARNING` / `LEAK WARNING` lines.
- Portability: each experiment script checks up front that `dataset.py` / `probe_cv.py`
  sit next to it and adds its own folder to `sys.path`, so a wrong working directory
  stops with `MISSING FILE(S): ...` instead of a bare `ModuleNotFoundError`.
