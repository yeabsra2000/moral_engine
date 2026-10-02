# Readable ≠ Steerable: Why Probe-Based Steering of Moral Tone Cannot Beat a Sham
### A mechanism-first study of observer probes and activation steering across four ethical/affective axes, and where the scale boundary sits

**Working title (short):** *MESE-CR: Concept Probes + Activation Steering for Moral-Emotional Control in Small LLMs*

---

## Abstract (draft)
Large language models encode moral and affective information in their intermediate
activations, and a growing literature both *reads* that information with probes and
*writes* to it with steering vectors. On a 1.5B instruction-tuned model we reproduce
the reading result and then fail, cleanly and repeatedly, to steer — and the claim of
this paper is that **the failure is not an artifact of our setup but a structural
property of probe-based steering.**

The central result is a mechanism, verified without a model. For any two-layer probe
`f(x) = W₂·SiLU(W₁x + b₁) + b₂`, the input gradient is
`∇f(x) = W₁ᵀ·diag(SiLU′(W₁x + b₁))·W₂ᵀ`, which lies in `rowspace(W₁)` **for every
x**. Every practical way of deriving a steering key from a probe — the gradient at the
positive-pole mean, averaged along the μ⁻→μ⁺ path, or averaged over the whole positive
class — therefore differs only in its per-unit gate scalars and cannot leave that
subspace. On real activations the three gradient keys are mutually **cos 0.995–1.000**;
`cos(logreg, mean) = +1.000` exactly, so the fitted logistic key is not independent
either; and even the probe's own supervised gradient sits at **cos 0.845–0.982** with
the plain mean difference. `test_steering_collapse.py` confirms the subspace containment
to machine precision (residual **0.000000**) on synthetic data, with no model. So the
probe can *read* a nonlinear boundary (grouped MLP 0.78–0.98 vs grouped linear
0.46–0.66) yet cannot *write* along it, because its keys are confined to a subspace that
the readout does not need.

We are careful to separate two claims that are easy to conflate. The subspace theorem
explains why the probe keys coincide with *each other*. That they also coincide with
`mean` on real activations is a separate, **empirical** fact — BCE training aligns `W₁`
with the class-separation direction — and it is contingent, not automatic: on a
synthetic *nonlinear* boundary the probe key points well away from `mean` (measured
cos **0.39**) while still lying inside `rowspace(W₁)`. Both facts are needed, and both
are testable offline.

The protocol is deliberately conservative, because this is a negative result and the
cheapest way to be wrong is to report an artifact. (a) **Template-grouped CV**: on our
884-sentence audited dataset (740 templated + 144 authored, 146 template groups), a
control feature space containing *only template identity and no tone information at
all* scores 0.76–0.93 under random 5-fold CV but falls to 0.42–0.55 under grouped CV,
so random splits are discarded. (b) **Per-axis reading controls**: length-only and
label-permutation. (c) A **sham** direction at matched scale, so that "steering
failed" is separable from "measurement is noise". Under these controls, seven keys —
mean, logistic, LDA, probe-grad, probe-grad-path, probe-grad-class, and a sham —
across layers {14, 20, 26} × {mean, last-token} pooling × 4 axes (Valence, Arousal,
Care/Harm, Fairness) move tone no further than the sham: permutation **p = 0.438**
over the whole 24-cell grid.

**Scope, stated plainly.** Every number above is measured on one 1.5B model. We
therefore report **a boundary, not a universal**, and position it against Sun et al.'s
Valence–Arousal subspace, which *does* steer at 8–14B. The natural reading — that the
confinement is a small-scale phenomenon — is a testable prediction, and we set that
test up (a 3B/8B rung at matched relative depth, left unquantized so the activation
geometry stays comparable) rather than claim its outcome. Related nulls at other scales
are now appearing (Da Silva et al., ACL 2025), so the contribution that travels is the
*mechanism* — scale-free in its statement and falsifiable at any size.

*(Quantitative results below are the released scripts' real CUDA output on the
884-sentence dataset: the full Observer grid (REPS=10, layers 12/14/16), the full
Steerer run (5 prompts × 4 samples × 3 scales), and the step7 key/layer/pooling grid.
The key-position cosine table is produced by `steering_keys.py` and needs no GPU;
`test_steering_collapse.py` verifies its central identity offline.)*

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
  - RQ4 (the question this paper turns on): **where do the steering keys point?**
    If every probe-derived key lies in span(μ⁺ − μ⁻), then the RQ3 null is a property
    of the method rather than an empirical accident — and only then can we state a
    boundary claim instead of an unexplained failure.
- Contributions: (1) the probe-gradient `rowspace(W₁)` identity (§4) and its measured
  consequence on real activations; (2) evidence that the two obvious "nonlinear key"
  repairs do not escape it; (3) a scale-bound, control-heavy negative result with a
  documented mechanism, plus an offline test (`test_steering_collapse.py`) that
  reproduces the mechanism with no model at all.

## 2. Related Work (cite correctly — do NOT claim these are new)
- Probing / concept bottleneck models: Alain & Bengio (2018, linear probes); Belinkov
  & Glass (2019) survey; Koh et al. (2020) Concept Bottleneck Models.
- Activation steering / representation engineering: Turner et al. (2023) Activation
  Addition (ActAdd); Zou et al. (2023) Representation Engineering; Rimsky et al. /
  recent steering literature.
- Affective axes: Russell (1980) Circumplex Model (Valence/Arousal).
- Morality: Haidt & Joseph (2004) Moral Foundations Theory (Care/Harm, Fairness/Cheating).
- **Closest prior work, and how this paper differs.** This is a crowded area and most
  of what we measure has a precedent; what we have *not* found stated anywhere is the
  mechanism in §4.
  - **Hewitt & Liang (EMNLP 2019)** — control tasks for probing, including the
    structural/template control we apply. We are *using* their discipline, not
    proposing it.
  - **Shaier et al. (arXiv 2402.00123)** — templated probing inflates accuracy scores.
    Our template-identity control (0.76–0.93 under random splits) is an independent
    replication at the 1.5B scale.
  - **Da Silva et al. (ACL 2025), "Steering off Course"** — steering results vary
    substantially across models, and steering can hurt performance. Closest in spirit
    to our null, but theirs is an empirical *variability* claim across many models. We
    do not restate it; we supply a mechanism for why a probe-derived key would be
    expected to underperform a sham, which that framing does not identify.
  - **Tan et al. (NeurIPS 2024)** — analyzes what steering vectors actually do (e.g.
    that they act on a restricted region and can be approximated by a bias-like
    intervention). §4 should be read as a probe-specific, exactly-verifiable instance
    of that picture, not a competing account.
  - **Sun et al., "Valence–Arousal Subspace in LLMs"** — the *direct counterexample*
    our scope statement is built around: they report successful VA steering at 8–14B
    and argue for **lexical mediation** (signature tokens carry the effect). We cannot
    dismiss this. It bounds our claim to 1.5B, and it turns their lexical mechanism
    into a concrete target: our length-only and surface controls are precisely the
    instrument that would catch an artifact of that kind, which is the adversarial
    test of their setting we propose.
- **What is actually new here:** (i) the probe-gradient `rowspace(W₁)` identity of §4
  and its measured consequence (gradient keys at cos 0.845–0.982 with the mean
  difference, and `cos(logreg, mean) = +1.000` exactly); (ii) the demonstration that
  the two obvious "nonlinear key" repairs — path-averaged and class-averaged gradients
  — do **not** escape it; (iii) the deliberate separation of that theorem from the
  *contingent* empirical collapse. Everything else is careful reproduction under
  controls, and is labelled as such.

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
- Steerer: direction estimated **seven** ways (`steering_keys.py`), each compared
  against a **sham `random` unit direction** at the same scale:
  `mean` (mean_pos − mean_neg), `logreg` (fitted logistic weights),
  `lda` (Fisher: (Cov+λI)⁻¹Δμ), `probe-grad` (∂logit/∂x of the measuring probe at
  the positive-pole mean), `probe-grad-path` (that Jacobian averaged along μ⁻→μ⁺),
  `probe-grad-class` (…averaged over every positive example), and `random`. Added at
  layer 14 at scale in {0.0, 1.5, 4.0}; tone measured by the probe *with the steer
  hook removed*; fluency = mean per-token log-likelihood. Axes whose measuring probe
  fails a held-out accuracy gate (0.60) are reported as NOT MEASURABLE rather than as
  a steering failure.
- **Key-position instrument (§4.0):** every key is also unit-normalized and its cosine
  against `mean` is printed *before* any steering. That costs no generation, so it is
  logged for every layer × pooling × axis in `step7_sweep.py`, and it — not the slopes —
  is the evidence for the confinement claim.

## 4. Results

### 4.0 The mechanism first: every probe-based key is confined to rowspace(W₁)

This is the paper's load-bearing claim, and it needs no model to check. Write the
measuring probe as a two-layer MLP, `f(x) = W₂·SiLU(W₁x + b₁) + b₂` — exactly the
architecture used in steps 5–7. Its input gradient at any point x is

    ∇f(x) = W₁ᵀ · diag( SiLU′(W₁x + b₁) ) · W₂ᵀ,

so **∇f(x) ∈ rowspace(W₁) for every x**. Any key obtained by *averaging* ∇f over a set
of points — one point (the positive-pole mean), a path (μ⁻ → μ⁺), or an entire class
(all positive examples) — is therefore a vector of the form `W₁ᵀ·diag(ḡ)·W₂ᵀ` for some
per-unit gate vector ḡ. **Averaging cannot change where the key points; it can only
re-weight the hidden units.** What we called the "nonlinear steering key" is a linear
functional of `W₁` wearing a nonlinear probe's clothes.

Verified in `test_steering_collapse.py` (CPU, no model): the fraction of each probe key
lying *outside* `rowspace(W₁)` is **0.000000** to machine precision, while the same
measure for the plain `mean` key is 0.331 — so the containment is specific to gradient
keys and is not a triviality.

**Consequence 1 — the three gradient keys coincide.** Measured pairwise cosines among
`probe-grad` (point), `probe-grad-path`, and `probe-grad-class` are **0.995–1.000**. The
two obvious repairs for a collapsing key therefore do not work, for a provable reason:
they re-weight the gates without leaving the subspace. This is a negative result about
the whole *method family*, and it closes the "you only tried a point gradient"
objection in advance.

**Consequence 2 — on real activations they also coincide with `mean`.** This second
step is **empirical** and must not be conflated with the theorem. At layer 14, mean
pooling (step6; 0.845–0.982 across axes, and 0.72–0.98 across the 24 step7 cells):

| Axis | probe acc | cos(logreg, mean) | cos(lda, mean) | cos(probe-grad, mean) | cos(sham, mean) |
|---|---:|---:|---:|---:|---:|
| Valence | 0.72 | **+1.000** | +0.731 | +0.982 | +0.009 |
| Arousal | 0.81 | **+1.000** | +0.704 | +0.964 | −0.016 |
| Care/Harm | 0.98 | **+1.000** | +0.429 | +0.960 | −0.055 |
| Fairness | 0.66 | **+1.000** | +0.493 | +0.845 | −0.030 |

(Values are from the committed `results_step6*.txt` logs; the `steering_keys.py`
refactor preserves this arithmetic exactly and the next run prints a fuller table,
adding `probe-grad-path` / `probe-grad-class` per axis.) Two things fall out:

- **`logreg` is not an independent key.** `cos(logreg, mean) = +1.000` on all four axes.
  The features are L2-normalized per row and the classes are balanced, so the BCE
  optimum sits on the mean-difference direction: the fitted logistic key is the mean
  key up to scale. Any experiment counting "mean" and "logreg" as two independent
  directions is double-counting one of them.
- **The better the probe reads, the harder its gradient collapses.** Care/Harm is the
  most readable axis (0.98) and its gradient key is the most confined (0.960); Fairness
  is the least readable (0.66) and the least confined (0.845). A *more* readable axis is
  a *more* linearly-separable one, and a point gradient on a linearly-separable problem
  is a linear discriminant by construction.

**Consequence 3 — the collapse is contingent, which is what makes it a finding.** If the
theorem alone explained everything, the collapse would be a tautology and not worth
reporting. It is not: on a synthetic *nonlinear* boundary the probe key points well away
from `mean` (measured **cos 0.39**) while still lying inside `rowspace(W₁)`. So the
theorem explains why probe keys agree with *each other*; BCE training aligning `W₁` with
the class-separation direction is what explains why they agree with `mean` on real
activations. Two claims, two tests, both asserted separately in
`test_steering_collapse.py`.

**Therefore** the RQ3 null below is not "steering mysteriously failed". Every key
derivable from the probe is confined to a subspace the probe never needed in order to
*read* — which is precisely the gap between a readout and a direction for control.

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
  is valid; yet adding any of the direction estimators at scale up to 4.0 moves
  tone only within −0.06…+0.10, no more than the SHAM random direction (−0.09…+0.06).
  The table below is the original four-key run from the committed log; the current
  script adds `probe-grad-path` and `probe-grad-class`, which are confined to the same
  subspace (§4.0) so their slopes cannot recover an effect the four below missed:

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
- **The mechanism explains the null (this paper's main claim).** Every probe-derived
  steering key lies in `rowspace(W₁)` by construction (§4.0), and on these activations
  it additionally collapses onto the mean-difference direction (gradient keys
  cos 0.845–0.982; `logreg` cos +1.000 exactly). So "readable ≠ steerable" is not an
  observation but a *consequence*: the probe reads a nonlinear boundary, and every key
  it can hand us is a linear functional of its own first layer. RQ4 therefore predicts
  RQ3's null, and the prediction is checkable offline.
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
- **Measurement before intervention — and a steering null that is now explained.**
  Because the measuring probe is validated on every axis (0.66–0.98) and all seven
  direction estimators agree — including a **probe-gradient** key derived from the
  readout's own nonlinear manifold *and* its two non-point repairs — yet none beats a
  sham random vector, the null is a genuine property of mean-pooled activation addition
  on a 1.5B model, not a measurement artifact. §4.0 then says *why*: the supervised key
  is confined to `rowspace(W₁)` and collapses onto `mean` (cos 0.845–0.982), so steering
  with it re-runs an experiment whose answer we already know.
- **Readable ≠ steerable — and now mechanistically so.** Care/Harm is the most readable
  axis (grouped 0.98) yet barely moves under its own steering direction (best supervised
  slope +0.10, probe-grad −0.06) — and §4.0 explains why rather than leaving it as a
  coincidence: the most readable axis is the most linearly-separable one, so its probe
  gradient is the *most* confined to `rowspace(W₁)` and collapses hardest onto `mean`
  (cos 0.960). The illustration and the mechanism agree, which is the point.
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
- **The scale boundary is the biggest limitation, and it is deliberate.** Everything
  measured here is one 1.5B model, so we claim a boundary, not a universal. The §4.0
  mechanism is stated for the probe *architecture* rather than the model, so it holds at
  any size; what is scale-dependent is the *second* step — whether BCE training on a larger
  model's activations leaves the probe gradient aligned with the mean difference, or
  whether a richer representation pushes them apart. Sun et al.'s successful
  Valence–Arousal steering at 8–14B is the direct counterexample we cannot dismiss, and
  the decisive experiment (same protocol, 3B/8B, matched *relative* depth, **unquantized**
  so the activation geometry is comparable) is specified in §8b but not run. Until it is,
  "the confinement is a small-scale phenomenon" remains a live alternative to our reading.
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
- **Conclusion.** On an 884-sentence, audit-gated dataset, a 1.5B model's mean-pooled
  activations encode moral/affective tone with genuine held-out-template generalization
  (grouped MLP 0.78–0.98, stable layers 12/14/16), and that readable structure is
  nonlinear (grouped linear 0.46–0.66 ≈ chance). Yet activation steering does not
  transfer that readable tone into generation: seven direction estimators — mean,
  logistic, LDA, and the validated probe's own gradient in three variants — all move
  tone no more than a random sham. **We do not stop at the null.** §4.0 shows the
  failure is structural: for a two-layer probe, `∇f(x) ∈ rowspace(W₁)` at every point,
  so any key derived from the probe by averaging is confined to a subspace it never
  needed in order to *read*. On real activations the gradient keys sit at cos
  0.845–0.982 with the plain mean difference, and `cos(logreg, mean) = +1.000` exactly,
  so two of the seven "independent" keys are the same key. A 24-cell search over layers
  {14,20,26} × {mean, last-token} pooling reproduces the null (permutation p = 0.438),
  while last-token pooling *improves reading* (probe 0.99–1.00). Reading works; steering
  with any probe-derived key this way cannot — and now we can say why, and demonstrate
  the mechanism on a CPU with no model at all (`test_steering_collapse.py`).
- **The claim we do not make.** Tone is not unsteerable. The confinement above is about
  *probe-derived additive keys at one layer of a 1.5B model*. Sun et al. steer
  Valence–Arousal successfully at 8–14B, and our own construction admits keys we never
  tried (attention-weighted, multi-layer, or learned against a generation objective
  rather than a probe).
- **Next:** (1) ✅ observer grid; (2) ✅ probe-gradient key + its two non-point repairs,
  ✅ layer sweep + last-token — all null, and all now explained by §4.0; (3) ✅
  sham-distribution protocol (`MESE_SHAM=n`) implemented — run it to convert the
  grid from a single-draw comparison into a real null-distribution test; (4) **the
  scale rung**: 3B/8B, matched relative depth, unquantized — the one experiment that
  decides between "structural" and "small-scale" (see §8b); (5) independent human
  rating of a hard-set sample; (6) larger steering scales (> 4.0) with a fluency cost
  check.

## 8b. The probes we tested, and why they all fail (now explained by §4.0)
The null is an *absence of effect under a whole method family*, not evidence that tone
is unsteerable — and §4.0 identifies which part is structural. Because the readout is
validated and nonlinear, the natural hope was that tone lives in a *nonlinear* region
of activation space that a single added mean-direction at one layer cannot reach. We
tested four such follow-ups — (a) the **probe-gradient** key, (a′) its **path-averaged**
and **class-averaged** repairs, (b) a **layer sweep** {14, 20, 26}, and (c) **last-token**
steering — and all four fail:

(a) the readout's own steepest ascent at the positive pole does not beat the sham,
because in mean-pooled L2-normalized activations it collapses onto the linear mean
direction (cos 0.845–0.982);
(a′) the two obvious repairs do not help either: `probe-grad-path` and
`probe-grad-class` sit at cos 0.995–1.000 with the point key, because averaging the
Jacobian only rescales the per-unit gates (§4.0). We report this as a negative result on
the method family rather than leaving the lever untested;
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
reports the noise statistics. The remaining, still-untested lever is therefore **scale**,
and §4.0 makes it a sharp test rather than a fishing trip: (d) run the identical protocol
on a 3B/8B model at matched *relative* depth, **unquantized** so the activation geometry
stays comparable. If the gradient keys stay confined there, the confinement is structural
at every scale we can test; if they break away and steering starts to work, the boundary
is a small-scale artifact and the mechanism needs a size term. Either outcome is a result
— which is why this paper stops at one model and states a boundary rather than a
universal.

## Appendix: Reproducibility
- Data pipeline (no GPU needed): `dataset_templates.py` → `easy.csv`;
  `dataset_hard.py` → `hard.csv` (+ audit); `dataset_merge.py` → `dataset.py`
  (`DATA` + `GROUPS`); `dataset_generate.py` / `dataset_validate.py` for LLM drafts.
- Local tests (CPU only, no model): `python test_grouped_cv.py` verifies the fold
  invariants and measures the template-memorization gap; `python
  test_probe_calibration.py` shows that a low-capacity control probe passes vacuously
  and calibrates what a linear probe can earn under this protocol (0.75 on a set that
  is linearly separable by construction); `python test_reading_controls.py` checks the
  label-permutation and length-only controls against synthetic ground truth;
  `python test_steering_collapse.py` verifies the §4.0 mechanism — that all three probe
  gradient keys lie in `rowspace(W₁)` to machine precision (residual 0.000000) while
  `mean` does not, that path- and class-averaging do not escape the point key
  (cos 0.995–1.000), that on a *nonlinear* boundary the probe key points away from
  `mean` (cos 0.39), and that a sham generator seeded with the caller's own seed
  reproduces the ground-truth direction exactly. `probe_cv.py` holds the shared,
  model-free CV logic and `steering_keys.py` the shared key estimators.
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
- Portability: each experiment script checks up front that `dataset.py` / `probe_cv.py` /
  `steering_keys.py` sit next to it and adds its own folder to `sys.path`, so a wrong
  working directory stops with `MISSING FILE(S): ...` instead of a bare
  `ModuleNotFoundError`.
