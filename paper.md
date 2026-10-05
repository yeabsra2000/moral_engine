# Probe-Derived Steering Keys Are Confined to the Probe's Own Row Space
### A mechanism-first study of observer probes and activation steering across four ethical/affective axes — and an open empirical question about the steering null

**Working title (short):** *MESE-CR: Concept Probes + Activation Steering for Moral-Emotional Control in Small LLMs*

> **Revision note (2026-10-03) — a withdrawn claim.** An earlier draft was titled
> *"Readable ≠ Steerable: Why Probe-Based Steering of Moral Tone Cannot Beat a Sham"*
> and reported the steering null as a *structural* result. It no longer does. Wiring the
> CPU dry-run mode next to the mechanism tests exposed a bug in the steering loop of
> `step6_steer_all.py` / `step7_sweep.py`: the hook was armed **once per scale** and
> disarmed on the **first sample**, so only 1 of every (prompts × samples) = 20
> generations in the "steered" arm was actually steered. Measured by counting live-hook
> state: **2 of the 40** generations that should have been steered were steered — a
> **0.050×** dilution. A real effect of size *e* therefore reads as ~*e*/20, i.e. into
> the sham noise floor, and **every control we had still behaves correctly under the
> bug**, because it dilutes the *treatment*, not the *measurement*: the probe gate reads
> fine, the sham is diluted identically, the permutation test stays null, fluency is
> undamaged. The loop is fixed (`SteerController`, with a printed `[steering audit]`
> asserting 100% application) but **the steering numbers in this paper are withdrawn
> pending a GPU re-run.** The mechanism below is unaffected — it is computed from
> activations, labels and the probe, and involves no generation at all. Full account:
> `results.md` §M9.

---

## Abstract (draft)
Large language models encode moral and affective information in their intermediate
activations, and a growing literature both *reads* that information with probes and
*writes* to it with steering vectors. On a 1.5B instruction-tuned model we reproduce
the reading result. What this paper actually contributes is a **mechanism that
constrains where any probe-derived steering key can point** — proved and machine-checked
offline — plus an honest account of why our own measurement of whether those keys steer
cannot yet be reported.

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
to machine precision (residual **0.000000**) on synthetic data, with no model. The
consequence is a *bound on the method*, not yet a measured null: a probe that can *read*
a nonlinear boundary (grouped MLP 0.78–0.98 vs grouped linear 0.46–0.66) derives all of
its keys from a subspace fixed by its own readout — so any steering it suggests is
confined to directions the readout never needed, and cannot test the nonlinear structure
that made the axis readable. Whether that confinement *is* what defeats steering at this
scale is the open question (see the revision note).

We are careful to separate two claims that are easy to conflate. The subspace theorem
explains why the probe keys coincide with *each other*. That they also coincide with
`mean` on real activations is a separate, **empirical** fact — BCE training aligns `W₁`
with the class-separation direction — and it is contingent, not automatic: on a
synthetic *nonlinear* boundary the probe key points well away from `mean` (measured
cos **0.39**) while still lying inside `rowspace(W₁)`. Both facts are needed, and both
are testable offline.

The protocol is deliberately conservative, because a negative result is the easiest
thing to get wrong. (a) **Template-grouped CV**: on our
884-sentence audited dataset (740 templated + 144 authored, 146 template groups), a
control feature space containing *only template identity and no tone information at
all* scores 0.76–0.93 under random 5-fold CV but falls to 0.42–0.55 under grouped CV,
so random splits are discarded. (b) **Per-axis reading controls**: length-only and
label-permutation. (c) A **sham** direction at matched scale, so that "steering
failed" is separable from "measurement is noise". (d) — added as a result of the
revision — a **treatment audit**: the run counts how many generations actually executed
with a live steering hook and fails loudly on any shortfall. Of these, (a)–(c) all report
correctly under a bug that dilutes the treatment, which is why (d) had to exist before any
steering slope could be believed.

An earlier run under (a)–(c) appeared to show seven keys —
mean, logistic, LDA, probe-grad, probe-grad-path, probe-grad-class, and a sham —
across layers {14, 20, 26} × {mean, last-token} pooling × 4 axes (Valence, Arousal,
Care/Harm, Fairness) moving tone no further than the sham, with permutation
**p = 0.438** over the whole 24-cell grid. **That measurement is withdrawn.** The loop
behind it steered only 1 generation in 20, so its slopes are equally consistent with no
effect and with an effect ~20× the noise floor; p = 0.438 was computed on diluted arms
and is therefore uninformative. The scripts, the controls and the re-run command are all
in place; what the re-run returns is an open question, and we report it as one.

**Scope, stated plainly.** The theorem is *architecture-level*: it holds at any model
size, because it is a property of a two-layer probe, not of a dataset or a checkpoint.
The empirical claims are not — the reading results are measured on one 1.5B model, so we
report **a boundary, not a universal**, and position it against Sun et al.'s
Valence–Arousal subspace, which *does* steer at 8–14B. The natural reading — that
the confinement is a small-scale phenomenon — is a testable prediction, and we set that
test up (a 3B/8B rung at matched relative depth, left unquantized so the activation
geometry stays comparable) rather than claim its outcome. Related nulls at other scales
are now appearing (Da Silva et al., ACL 2025), so the contribution that travels is the
*mechanism* — scale-free in its statement and falsifiable at any size.

*(Numbers below come from the released scripts' real CUDA output on the 884-sentence
dataset: the full Observer grid (REPS=10, layers 12/14/16) and the step7
key/layer/pooling grid. The key-position cosine table is produced by `steering_keys.py`
and needs no GPU; `test_steering_collapse.py` verifies its central identity offline.*

**Read the labels on the steering tables.** Anything reporting a *reading* accuracy, a
*cosine between keys*, or a *control* (length, label-permutation, sham) was computed
without generation and **stands**. Anything reporting a *steering slope* came from the
diluted loop described in the revision note and is **withdrawn**; those cells are marked
`⚠️ withdrawn` below rather than deleted, so the record stays complete.

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
    probe? (Answers the readout side of the central hypothesis.)
  - RQ3: Can activation steering predictably shift output tone (dose-response)?
  - RQ4 (the question this paper turns on): **where do the steering keys point?**
    §4.0 answers it in two steps, and the distinction matters. *By construction*, every
    probe-derived key lies in `rowspace(W₁)` — a subspace fixed by the readout, not by
    the data. *Empirically*, on these activations the probe gradient additionally
    collapses onto the mean-difference direction (cos 0.845–0.982). Only with both does
    the RQ3 null become an explained property of the method rather than an unexplained
    failure — and only then can we state a boundary claim.
- Contributions: (1) the probe-gradient `rowspace(W₁)` identity (§4.0) and its measured
  consequence on real activations; (2) evidence that the two obvious "nonlinear key"
  repairs do not escape it; (3) the deliberate separation of the *structural* claim from
  the *contingent* one, which is what keeps the finding from being a tautology;
  (4) a scale-bound, control-heavy negative result, plus an offline test
  (`test_steering_collapse.py`) that reproduces the mechanism with no model at all.

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

**Therefore** the confinement is a bound on the method, not an explanation of a
measured failure. Every key derivable from the probe is confined to a subspace the probe
never needed in order to *read* — which is precisely the gap between a readout and a
direction for control. What this does *not* yet establish is that the confinement is what
defeats steering in practice: our attempt to measure that is withdrawn below, because the
run never applied its own treatment (§4 / revision note, `results.md` §M9).

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
- **RQ3 (steering) — ⚠️ WITHDRAWN. Do not cite this bullet or its table.** An earlier
  full run appeared to give a clean, gated null, and the table below is what it printed.
  The loop that produced it armed the steering hook once per scale and disarmed it on the
  first sample, so only 1 of every (5 prompts × 4 samples) = 20 generations in the
  "steered" arm was steered (**measured: 2 of 40**). A real effect of size *e* would
  therefore have read as ~*e*/20 — indistinguishable from the sham's ±0.05 noise floor —
  which is exactly the pattern the table shows. The gate, the sham and the fluency check
  all behave correctly under this bug because it dilutes the *treatment*, not the
  *measurement*; only a treatment audit catches it, and there was none. The loop is fixed
  (`SteerController`, per-generation arming, `[steering audit]` printed per run) and the
  re-run is specified, but **no steering slope in this paper is evidence of anything yet.**
  `results_step6.txt` remains in the repo as the record of the diluted run, labelled.
  The *cosines* in the discussion below are unaffected: they come from §4.0, not from
  generation.

  ⚠️ withdrawn — original diluted run:

  | Axis | read acc | mean sl | logreg sl | lda sl | probe-grad sl | SHAM random sl |
  |---|---:|---:|---:|---:|---:|---:|
  | Valence | 0.72 | +0.05 | +0.08 | 0.00 | +0.02 | +0.05 |
  | Arousal | 0.81 | −0.04 | +0.01 | −0.09 | +0.00 | +0.05 |
  | Care/Harm | 0.98 | −0.02 | +0.02 | +0.10 | −0.06 | +0.04 |
  | Fairness | 0.66 | +0.03 | −0.01 | −0.01 | −0.02 | +0.06 |

  *(Table above is the diluted run; keep the cosines, discard the slopes.)* The cosines
  are real — logreg ≡ mean, cos = 1.000; LDA 0.43–0.73; probe-grad 0.85–0.98 of mean —
  and they come from §4.0, so they stand. What does not follow from this run is any
  conclusion about steering: "the probe-grad key also fails", "the null is not a
  bad-direction artifact", "best real slope +0.10", "steering does not damage English"
  and "a genuine *steering* null" were all read off a run whose steered arm was 95%
  unsteered. Because the measuring probe is validated, the *readout* half of the
  argument is sound — but "measurement is noise" was never the only failure mode we
  had to worry about, and the one we missed is the one that invalidates this bullet.
  Re-run before writing any of it again.
- **RQ3 (search) — ⚠️ steering half WITHDRAWN; reading half stands.** A
  follow-up grid (`step7_sweep.py`) searched layers {14, 20, 26} × pooling
  {mean, last-token} × 4 axes = 24 cells, each with its own same-layer probe, gate
  and sham. Two findings:

  | check | result | status |
  |---|---|---|
  | reading, mean pooling (grouped) | 0.63–0.99 | ✅ stands (no generation) |
  | reading, **last-token** pooling (grouped) | 0.75–1.00 (Care/Harm 0.99–1.00; Valence 0.99) | ✅ stands (no generation) |
  | mean(real − sham) over 48 (direction × cell) pairs | **+0.004**, permutation **p = 0.438** | ⚠️ withdrawn (diluted arms) |
  | sham slope spread across cells | −0.22 … +0.19 | ⚠️ withdrawn (diluted) |
  | sign consistency (mean / probe-grad slopes) | 10 / 13 and 15 / 8 → **flips** | ⚠️ withdrawn (diluted) |

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

  Reading improves under last-token pooling (a positive secondary result). ⚠️ **The
  steering half of this paragraph is withdrawn:** the permutation test, the sign-flip
  evidence and the conclusion that the null "generalizes" were all computed on arms in
  which only 1 generation in 20 was steered (see the revision note), so p = 0.438 and
  the +0.004 / −0.22…+0.19 / 10+/13− figures are statistics on a treatment that was
  mostly not administered. The *instability* observations in the same paragraph — a cell
  flipping flag→null across runs, a sham moving 0.32 — are genuine warnings about the
  single-draw control and remain worth having (they are why the code now wants
  `MESE_SHAM=20`), but they say nothing about whether steering works. What survives:
  the search *protocol* is specified, the per-cell gates and controls are in place, and
  re-running it is a one-line change. The honest current answer to "does the null
  generalize across layers and pooling?" is **unknown**, not "yes".
- ⚠️ **WITHDRAWN — "concrete illustration" kept only as a record of the earlier reading.**
  Care/Harm is the most readable axis (0.98); its best *diluted* slope was +0.10 (LDA) and
  its probe-grad key −0.06. The comparison it was used to make — "the probe's own gradient
  clusters with the directions that fail to move tone" — confuses a key that was confined
  by §4.0 with a treatment that was not applied. Re-do after the re-run.
- Negative result: Care/Harm hard-context cases remain hard (the probe reads care
  from keywords, not from context alone) — our earlier keyword-dependence finding
  is superseded only in degree, not in kind.

## 5. Discussion
- **The mechanism bounds the method — it does not yet explain a null.** Every probe-derived
  steering key lies in `rowspace(W₁)` by construction (§4.0), and on these activations
  it additionally collapses onto the mean-difference direction (gradient keys
  cos 0.845–0.982; `logreg` cos +1.000 exactly). So the "readability does not imply
  steerability" story would follow as a *consequence* if steering were measured to fail.
  **That measurement is withdrawn** (the loop applied its treatment to 1 generation in
  20), so the theorem currently stands as a bound on the *method family* rather than an
  explanation of an observed effect. What makes it worth having anyway: the theorem is
  checkable offline, scale-free in statement, and tells us in advance that adding
  `probe-grad-path` / `probe-grad-class` cannot recover an effect the point key missed —
  a prediction that holds (cos 0.995–1.000 with the point key) and does not depend on
  generation.
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
- **Measurement before intervention — and the intervention we could not yet measure.**
  Because the measuring probe is validated on every axis (0.66–0.98), the *readout*
  claim is sound: tone is decodable from these activations. The *steering* half of this
  pairing is what remains open: the run in which seven direction estimators all failed
  to beat a sham used a loop that administered the treatment to 1 generation in 20, so
  "none beats a sham" was never established. §4.0 tells us in advance what to expect —
  the supervised key is confined to `rowspace(W₁)` and collapses onto `mean`
  (cos 0.845–0.982), so steering with it re-runs an experiment whose answer the
  geometry already suggests — but *expecting* a null is not *measuring* one.
- **A bug that no control could see (a methodological contribution in itself).** The
  steering loop armed the hook once per scale and disarmed it on the first sample to
  take an honest steer-off reading — which silently removed the treatment from 19 of
  every 20 generations. Every control we had still reported normally: the probe gate
  tests the readout, the sham is diluted by exactly the same factor, the permutation
  test compares two dilutions, fluency is untouched, and the §4.0 cosine table never
  involves generation at all. It dilutes the *treatment*, not the *measurement*, so a
  study can have a spotless control panel and still be measuring noise. The only
  question that catches it — *how many generations actually ran steered?* — was not
  asked, and is now structurally asked (`SteerController.check()`, plus a printed
  `[steering audit]` on every run). We flag this because it is a class of error
  invisible to the usual validation strategy for negative results, and this paper's
  entire protocol had been built to catch the *other* class (measurement artifact).
- **Care/Harm hard failure (qualitative):** intent/context-conditional morality is
  hard to decode from mean pooling alone; the probe still leans on surface cues
  (R4 non-replication) → future work (token-level or attention-weighted pooling).

## 6. Limitations
- The easy set is template-generated; grouped CV is reported precisely because the
  random-split numbers are inflated. The hard set is small (144) and authored by
  us, so it carries author bias — partially mitigated by mirrored pos/neg pairs
  and an automatic leakage/concreteness audit.
- 🔴 **The steering result is not currently a result.** The `step6`/`step7` steering
  numbers reported earlier in this draft came from a loop that armed the hook once per
  scale and disarmed it on the first sample, so only 1 generation in 20 in the "steered"
  arm was actually steered (measured: 2 of 40). Slopes are therefore diluted ~20× into
  the sham noise floor, and p = 0.438 was computed on two diluted arms. The bug is fixed
  and self-auditing, but until the experiment is re-run **this paper has a proven
  mechanism and no measurement of what the mechanism is supposed to explain.** That
  asymmetry is the single largest limitation here — larger than the scale boundary —
  and it is the reason the title no longer asserts a null. Full account in
  `results.md` §M9 and the revision note.
- Single model; mean pooling plus last-token pooling; Observer layers {12,14,16},
  steering searched at {14,20,26}; no attention-weighted or token-level analysis.
- **The scale boundary is the second-biggest limitation, and it is deliberate.** Everything
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
  by chance. That is a real and independent defect, and it is why `MESE_SHAM=n`
  (z > 3) is implemented — but note it applies *on top of* the dilution bug above, so
  the earlier grid's permutation test (p = 0.438) cannot be rescued by better sham
  statistics alone: two diluted arms are not a test of anything, however many shams
  you draw. Both need fixing together in the re-run. A null — once we actually have
  one — would still not rule out attention-weighted keys or scales above 4.0.
- Label polarity for the hard set is entailed by construction rather than
  validated by independent human raters (a planned extension).

## 7. Ethics
- Activation steering can be used for alignment AND for jailbreaking. We frame this as
  alignment research and discuss dual-use. Steering is not a complete safety mechanism.

## 8. Conclusion & Next Steps
- **Conclusion.** On an 884-sentence, audit-gated dataset, a 1.5B model's mean-pooled
  activations encode moral/affective tone with genuine held-out-template generalization
  (grouped MLP 0.78–0.98, stable layers 12/14/16), and that readable structure is
  nonlinear (grouped linear 0.46–0.66 ≈ chance). What *cannot* yet be said is that
  activation steering fails to transfer that readable tone into generation: the run that
  appeared to show seven direction estimators — mean, logistic, LDA, and the validated
  probe's own gradient in three variants — all failing to beat a random sham, steered
  only 1 generation in 20 (revision note; `results.md` §M9), so that result is withdrawn.
  **What we can stand behind is the mechanism, and it is proved rather than observed.**
  §4.0 shows, for a two-layer probe, that `∇f(x) ∈ rowspace(W₁)` at every point, so any
  key derived from the probe by averaging is confined to a subspace it never needed in
  order to *read* — and `test_steering_collapse.py` verifies that to machine precision on
  a CPU with no model at all. On real activations the gradient keys sit at cos
  0.845–0.982 with the plain mean difference, and `cos(logreg, mean) = +1.000` exactly,
  so two of the seven "independent" keys are the same key — and that half of the
  picture is independent of generation, so it stands. Meanwhile last-token pooling
  *improves reading* (probe 0.99–1.00), which also stands. The honest shape of this
  paper is therefore: a proven bound on a method family, a solid reading result, a
  positive finding about pooling, a bug found in the experiment meant to connect them,
  and a re-run that decides whether "Readable ≠ Steerable" is true or merely hoped for.
- **The claim we do not make.** Tone is not unsteerable — and after M9 we cannot even
  say we showed it at 1.5B. The confinement above is about *probe-derived additive keys
  at one layer of a 1.5B model*. Sun et al. steer Valence–Arousal successfully at
  8–14B, and our own construction admits keys we never tried (attention-weighted,
  multi-layer, or learned against a generation objective rather than a probe).
- **Next:** (1) ✅ observer grid; (2) ✅ probe-gradient key + its two non-point repairs,
  ✅ layer sweep + last-token protocol — scripts in place, **results withdrawn pending
  re-run**; (3) ✅ sham-distribution protocol (`MESE_SHAM=n`) implemented;
  (3b) ✅ **dose-verdict decision rule corrected** (`results.md` M13). The empirical test's
  add-one floor is `1/(n+1)`, so `p < 0.05` was *unattainable* below n=20 and a clean
  0-of-15 separation would have been reported as a null; the verdict now prints the floor
  and returns `CANDIDATE` instead, and the per-scale fluency label is sign-aware (an
  *improvement* was reading as "MODEL DEGRADING"). 🔴 The **Arousal** dose run's
  transcript was never persisted, so it is not yet a result — re-run
  `MESE_AXIS=Arousal MESE_SHAM=25 python step8_dose.py` and save
  `results_step8_arousal_n25.txt`. Until that lands, the dose null holds **on Care/Harm
  only** and the paper cannot claim it generalizes across axes;
  (4) 🔴 **RE-RUN step6 and step7 with the fix**, checking that each prints
  `[steering audit] N/N` — this is the blocking step, and everything else waits on it;
  (5) **the scale rung**: 3B/8B, matched relative depth, unquantized — the experiment
  that decides between "structural" and "small-scale" (see §8b), which now needs (4)
  first; (6) independent human rating of a hard-set sample; (7) larger steering scales
  (> 4.0) with a fluency cost check.

## 8b. The probes we tested, and what each item actually establishes

The framing here has changed. The version of this section that preceded the revision
note claimed "all four fail"; two of them are *geometric* claims that need no
generation and still hold, and two are *empirical* steering claims that are withdrawn
because the loop administered the treatment to 1 generation in 20. Because the readout
is validated and nonlinear, the natural hope was that tone lives in a *nonlinear* region
of activation space that a single added mean-direction at one layer cannot reach. We
ran four such follow-ups — (a) the **probe-gradient** key, (a′) its **path-averaged**
and **class-averaged** repairs, (b) a **layer sweep** {14, 20, 26}, and (c) **last-token**
steering — and this is what each one shows, honestly labelled:

(a) ✅ **stands — geometry.** In mean-pooled L2-normalized activations the readout's own
steepest ascent at the positive pole collapses onto the linear mean direction
(cos 0.845–0.982). This is computed from activations and the probe; no generation is
involved. What it predicts — that this key will steer like the linear keys — is now a
prediction, not an observation, until the re-run.

(a′) ✅ **stands — provable.** The two obvious repairs do not escape the point key:
`probe-grad-path` and `probe-grad-class` sit at cos 0.995–1.000 with it, because
averaging the Jacobian only rescales the per-unit gates (§4.0). This one is the
strongest item in the section, since it is both a theorem and a measured
machine-precision result in `test_steering_collapse.py`. We report it as a bound on the
method family rather than leaving the lever untested.

(b)/(c) ⚠️ **withdrawn — empirical.** `step7_sweep.py` searches layers × {mean,
last-token} pooling × axis with a per-cell probe, gate and sham, and reading does
improve (last-token probe 0.99–1.00, which stands). But the real−sham difference of
+0.004 (permutation p = 0.438) and the sign-flip example (probe-grad Care/Harm:
+0.25 @ L20 vs −0.31 @ L26) were computed on diluted arms, so they cannot support "the
null *generalizes* rather than breaks". That sentence is withdrawn; the search
protocol, gates and controls are all in place and the re-run is one command.

Two methodological caveats were also surfaced and are handled in code, and both are
*independent* of the dilution bug — they would matter even now that the loop is fixed.
First, a **single** sham draw makes a > +0.10 flag threshold noise-sensitive (the
per-cell noise floor at 20 generations/cell is ±0.2; a 24-cell grid then flags ~half the
cells by chance): `step7_sweep.py` therefore supports `MESE_SHAM=n` to build a sham
**distribution** and score each cell by a z-score (> 3) instead of one draw, and
`analyze_step7.py` reports the noise statistics. Second, the treatment itself now audits
it (`SteerController.check()`), which is the caveat the study had been missing. Once (4)
of §8 is re-run, the remaining untested lever is **scale**, and §4.0 makes it a sharp
test rather than a fishing trip: (d) run the identical protocol on a 3B/8B model at
matched *relative* depth, **unquantized** so the activation geometry stays comparable.
If the gradient keys stay confined there, the confinement is structural at every scale
we can test; if they break away and steering starts to work, the boundary is a
small-scale artifact and the mechanism needs a size term. Either outcome is a result
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
  model-free CV logic and `steering_keys.py` the shared key estimators *and* the
  `SteerController` hook. Section D of `test_steering_collapse.py` also reproduces the
  dilution bug on a fake layer (2 of 40 generations steered) and asserts that
  `SteerController` cannot under-apply — the check that was missing when the steering
  null was written. Section E adds a device-hygiene check that *can* run on CPU: an AST
  scan asserting no `torch.randn(...)` pairs `device=` with a default CPU generator,
  which is the `RuntimeError` that only appears on CUDA (see `results.md` §M10).
- GPU experiments (Google Colab T4): `pilot.py`, `step2_probe.py`,
  `step3_rigorous.py`, `step4_steer.py`, `step5_robust.py`, `step6_steer_all.py`,
  `step7_sweep.py`.
- Noise analysis of the steering search (CPU only): `python analyze_step7.py`
  parses `results_step7.txt` and prints the real-vs-sham distribution statistics,
  the sign-consistency check, and the sham noise floor. ⚠️ The committed
  `results_step7.txt` is a **withdrawn** diluted run, so re-run step7 first; the parser
  itself is unchanged and will accept the new log.
- Logs and corrected findings: `results.md` (`results_step5*.txt`,
  `results_step6*.txt`, `results_step7.txt`).
- Cost control: `MESE_DRY=1 python step5_robust.py` runs the entire Observer analysis
  with **no model** (activations replaced by seeded noise) in ~5 min on a laptop CPU
  (305 s measured, QUICK mode), so wiring errors are caught before GPU time is spent.
  It logs to `results_step5_dry.txt` (never the real log) and, on noise, must print no
  `CONFOUND WARNING` / `LEAK WARNING` lines. **`MESE_DRY=1 python step6_steer_all.py`
  and `MESE_DRY=1 python step7_sweep.py` do the same for the steerer** using the
  `dry_model.py` stub (28-layer, d=64 char-level toy that honours the real hook API), and
  each ends with a `[steering audit] N/N` line — the check that would have caught the
  dilution bug. These modes inject a synthetic label signal purely so the measurability
  gates open and the key/control code actually executes; every number they print is
  meaningless, which is why they log to their own `_dry` files.
- Portability: each experiment script checks up front that `dataset.py` / `probe_cv.py` /
  `steering_keys.py` sit next to it and adds its own folder to `sys.path`, so a wrong
  working directory stops with `MISSING FILE(S): ...` instead of a bare
  `ModuleNotFoundError`.
