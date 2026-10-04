# MESE-CR — reading and steering moral/emotional tone in a small LLM

Reading/writing moral–emotional tone from a small LLM's activations (Observer probe)
and steering it (Steerer), on Qwen/Qwen2.5-1.5B-Instruct.

## What this is

A small, fully reproducible study of whether moral/emotional tone can be **read out of**
and **written into** the residual stream of a 1.5B instruction-tuned LLM, built on an
**884-sentence audited dataset** (740 templated + 144 authored, 146 template groups) over
four axes: Valence, Arousal, Care/Harm, Fairness.

It is written as a sequence of *checks*, because the most useful thing this project
produced is methodological: several plausible ways to "find" tone here turn out to be
artifacts, and the repo now refuses to report them.

## Findings

1. **Mechanism: every probe-based steering key is confined to one subspace.** For a
   probe `f(x) = W₂·SiLU(W₁x + b₁) + b₂`, the input gradient at *any* point is
   `W₁ᵀ·diag(SiLU′(W₁x + b₁))·W₂ᵀ`, which lies in `rowspace(W₁)`. So averaging it — at a
   single point, along the μ⁻→μ⁺ path, or over the whole positive class — only rescales
   the per-unit gates and cannot change where the key points. Measured: the three gradient
   keys are mutually **cos 0.995–1.000**, and `cos(logreg, mean) = +1.000`, so `logreg`
   is not an independent key either. `test_steering_collapse.py` confirms the containment
   to machine precision (residual **0.000000**) with no model at all.
2. **Reading works, and the recoverable structure is nonlinear.** Template-grouped CV MLP
   accuracy is **0.78–0.98** across layers 12/14/16, while a grouped *linear* probe gets
   **0.46–0.66** (Arousal and Fairness at chance). `test_probe_calibration.py` shows a
   synthetic set that is linearly separable *by construction* still earns only **0.75**
   under this protocol, so the linear probe is not merely undertrained. The axis is
   therefore nonlinear *and* readable — but the keys in (1) cannot reach it.
3. 🔴 **Hence steering is null — but that result is WITHDRAWN, and the reason it is
   worth reading.** Seven keys across layers {14, 20, 26} × {mean, last-token} pooling ×
   4 axes appeared to move tone no further than a random sham (permutation **p = 0.438**
   over the 24-cell grid). On wiring a CPU dry-run mode next to the mechanism tests, the
   steering loop turned out to arm the hook **once per scale** and disarm it on the
   **first sample**, so only 1 of every (prompts × samples) = 20 generations in the
   "steered" arm was steered — measured at **2 of 40**, a **0.050×** dilution. A real
   effect of size *e* would read as ~*e*/20, i.e. straight into the ±0.05 sham noise
   floor, which is exactly the pattern the numbers showed. **Every control we had still
   passed**, because the bug dilutes the *treatment*, not the *measurement*: the probe
   gate tests the readout, the sham is diluted by the same factor, the permutation test
   compares two dilutions, and fluency is untouched. The loop now arms per generation
   and audits itself (`SteerController`, `[steering audit] N/N` on every run); **re-run
   step6/step7 on GPU before citing any steering number.** Full account: `results.md` M9.

   **The replacement experiment is now in, and it is still a null.** `step8_dose.py` on
   **Care/Harm** — the best-reading axis (grouped **0.98**) — with the hook auditing
   **100%** and a **7.08%** dose of `||h||`, gives a real-key paired tone slope of
   **+0.214 ± 0.071** against a distribution of **15 random directions** (mean −0.074,
   sd 0.266, range [−0.689, +0.343]). That is **+1.08 sd**, with **2/15** shams reaching it
   (add-one **p = 0.188**). The shift is reliably nonzero yet **not specific**: the readout
   moves with perturbation *magnitude*, not with the moral direction, and the
   dose–response is non-monotone (+0.21, −0.29, −0.07, +0.19 at scales 4/10/25/50). A
   random direction even out-moved the real key (|−0.227| vs +0.214). This is exactly what
   the subspace mechanism predicts — and it is the stronger paper. Full account:
   `results.md` **M12**.
4. **The collapse is contingent, not tautological — which is what makes it a finding.**
   On a synthetic *nonlinear* boundary the probe key points well away from `mean`
   (measured cos **0.39**) while still lying inside `rowspace(W₁)`. So the subspace theorem
   explains why the probe keys coincide with *each other*; BCE training aligning `W₁` with
   the class-separation direction is what explains why they coincide with `mean` on real
   activations (0.85–0.98). Two separate claims — do not conflate them.
5. **Two controls carry the result.** Random train/test splits are inflated to **0.76–0.93**
   by template identity *alone* — so grouped CV is the honest protocol. A reading is only
   accepted if the length-only and label-permutation controls stay silent. Control-design
   traps found and fixed along the way: three in `results.md` M7, a fourth (a sham
   generator seeded with the caller's own seed reproduces the ground-truth direction
   exactly) guarded by `test_steering_collapse.py`, and a fifth — the **dilution trap**
   in (3), which is different in kind from the others: it is the only one where the
   whole control panel stays green while the experiment measures nothing. For that
   reason the repo now also audits the *treatment* (`SteerController`), not just the
   measurement.

Full numbers, corrections and provenance: **`results.md`**. Write-up: **`paper.md`**.

> ⚠️ **The committed `results_step6.txt` and `results_step7.txt` are diluted runs.**
> Their *reading* accuracies, *cosine tables* and *reading controls* are fine (computed
> without generation); their *steering slopes* and *p = 0.438* are not evidence of
> anything. Re-run step6/step7 before quoting them.

## Reproduce (none of this needs a GPU)

```bash
python test_grouped_cv.py          # fold invariants + template-memorization control
python test_probe_calibration.py   # catches vacuous controls; calibrates the linear probe
python test_reading_controls.py    # length/permutation controls vs synthetic ground truth
python test_steering_collapse.py   # §4 mechanism (rowspace) + the dilution/treatment audit
python check_length_control.py     # re-derives the length control (tokenizer only, no model)
MESE_DRY=1 MESE_QUICK=1 python step5_robust.py   # end-to-end wiring check, no model (~5 min)
MESE_DRY=1 python step6_steer_all.py             # steerer wiring check, no model
MESE_DRY=1 python step7_sweep.py                 # sweep wiring check, no model
```

Each dry run logs to its own `_dry` file (never a real result) and the two steerer ones
end with `[steering audit] N/N generations ran with the hook LIVE` — that line must be
100%, and it is the check whose absence let a 20×-diluted null look clean.

The GPU experiments (below) produce `results_step5.txt`, `results_step6.txt`,
`results_step7.txt`, which are committed and cited in `results.md` — **but the step6 and
step7 logs are from the diluted loop and are withdrawn pending a re-run** (see
`results.md` M9).

## A. Dataset — runs anywhere (no GPU, ~5 s)
```bash
python dataset_templates.py     # -> easy.csv    (740 templated, guaranteed polarity)
python dataset_hard.py          # -> hard.csv    (144 authored + automatic leakage audit)
python dataset_merge.py         # -> dataset.py  (DATA + GROUPS, one gate before writing)
python test_grouped_cv.py       # local self-test: fold invariants + memorization control
```
The audit must print `AUDIT CLEAN` and the merge must print `GATE REJECTED 0` /
no rejections. `dataset.py` is generated — never edit it by hand.

## B. GPU experiments — Google Colab (T4)
Setup: `Runtime → Change runtime type → T4 GPU`, then run these cells:

```python
# 1) confirm you actually got a GPU (must print Tesla T4)
!nvidia-smi

# 2) deps (Colab already ships CUDA torch; do NOT reinstall torch)
!pip install -q transformers accelerate

# 3) upload these files into /content (drag & drop, or the folder icon):
#    dataset.py  probe_cv.py  steering_keys.py  step5_robust.py  step6_steer_all.py
#    step7_sweep.py  analyze_step7.py  step8_dose.py
#    check_length_control.py  test_*.py  (CPU-only, no model needed)
#    steering_keys.py is REQUIRED by step6/step7 -- omitting it stops the run
#    immediately with "MISSING FILE(S): steering_keys.py".
#    dry_model.py only if you intend to use MESE_DRY=1 (no model download).
#    or pull them from Drive if you keep the project there:
# from google.colab import drive; drive.mount('/content/drive')
# %cd /content/drive/MyDrive/moral_engine
#
#    TROUBLESHOOTING: every experiment script checks first that dataset.py (and
#    probe_cv.py, where needed) sits next to it. If not it stops immediately with
#      "MISSING FILE(S): dataset.py"  -> re-upload those files, or cd into the
#    folder that holds them, then re-run. Scripts also add their own folder to
#    sys.path, so running them from a parent directory works.

# 4) sanity check: the dataset must import with 884 rows and 146 groups
!python -c "import dataset; print(len(dataset.DATA), len(dataset.GROUPS), len(set(dataset.GROUPS)))"

# 4b) OFFLINE SELF-TESTS first (CPU only, ~40 s, no model). These catch controls
#     that would pass vacuously, and calibrate what a linear probe can earn here.
!python test_grouped_cv.py
!python test_probe_calibration.py
!python test_reading_controls.py
#     check_length_control.py re-derives the length-only reading control EXACTLY (10
#     seeds + spread) and touches no model weights -- it downloads only the ~7 MB
#     tokenizer, so it is a cheap way to verify a length claim off-GPU:
!python check_length_control.py

# 4c) OPTIONAL, but cheapest possible check: run the analysis with NO model at all
#     (~5 min on CPU, measured 305 s). Activations are replaced by seeded noise, so it
#     only proves the script runs end to end (probes -> controls -> flags -> log)
#     before you spend GPU time. Its numbers are meaningless and it writes
#     results_step5_dry.txt, never the real log.
!MESE_DRY=1 MESE_QUICK=1 python step5_robust.py
#     The same flag works on the steerer scripts, and there it matters MORE: it is
#     what would have caught the dilution bug. Each ends with a
#     "[steering audit] N/N generations ran with the hook LIVE" line that must be 100%.
!MESE_DRY=1 python step6_steer_all.py
!MESE_DRY=1 python step7_sweep.py

# 5) SMOKE TEST first (~2-4 min). Confirms the model, hooks and log file work.
!MESE_QUICK=1 python step5_robust.py

# 6) full Observer run (~15-25 min)
!python step5_robust.py

# 7) Steerer run — do the quick version first, then the full one (~20-40 min)
!MESE_QUICK=1 python step6_steer_all.py
!python step6_steer_all.py

# 7b) SEARCH run — layer sweep + last-token test (paper 8b, levers b/c).
#     QUICK smoke first, then the full grid (~45-90 min; trim with MESE_LAYERS/MESE_POOL)
!MESE_QUICK=1 python step7_sweep.py
!python step7_sweep.py
#     cheaper focus examples:
# !MESE_LAYERS=14,20,26 MESE_POOL=last python step7_sweep.py   # last-token across 3 layers
# !MESE_LAYERS=14,26      MESE_POOL=mean python step7_sweep.py # two layers, mean pooling
#     confirm any flagged cell against a sham DISTRIBUTION (per-cell z-score):
# !MESE_LAYERS=14,26 MESE_POOL=last MESE_SHAM=20 python step7_sweep.py
#     noise stats of a finished step7 log (CPU only, no GPU needed):
# python analyze_step7.py

# 7c) DOSE run — was the Step 6 null real, or just under-scaled? THIS is the
#     decisive steering experiment (~15 min at MESE_SHAM=6, ~60-75 min at 15).
#     It DEFAULTS to Care/Harm: the strongest-reading axis (template-grouped
#     0.98 +- 0.00) and the only one whose instrument sits comfortably clear of
#     the 0.60 gate. Valence reads 0.84 +- 0.17 — wide enough that a short run
#     there can land under the gate and abort for instrument reasons, not
#     steering reasons. Bisect fluency first; only then trust the verdict.
!MESE_SHAM=6 python step8_dose.py                       # smoke (~15 min)
!MESE_AXIS=Care/Harm MESE_SHAM=15 python step8_dose.py  # confirmatory run
#     The script REFUSES to run unless the measuring probe clears a
#     template-grouped held-out gate at the SAME protocol step5 reports
#     (hidden=64, reps=10), so no verdict can rest on a weak instrument.
#     A second axis, run identically, is the cheapest confirmation:
# !MESE_AXIS=Arousal MESE_SHAM=15 python step8_dose.py
#     Rename the log between axes so the runs do not merge (it APPENDS):
# !mv results_step8_dose.txt results_step8_care_n15.txt

# 8) nothing is lost if the session drops: the runs append to these logs
!ls -la results_step5.txt results_step6.txt results_step7.txt results_step8_dose.txt
!cat results_step7.txt | tail -40

# 9) download the logs to paste back into the chat
from google.colab import files
files.download('results_step5.txt'); files.download('results_step6.txt')
files.download('results_step7.txt'); files.download('results_step8_dose.txt')
```

Optional older steps: `pilot.py`, `step2_probe.py`, `step3_rigorous.py`, `step4_steer.py`.

### B2. Second model (generalization) — one env var
Every step script takes `MESE_MODEL` (and `MESE_LAYERS` / `MESE_LAYER` for a model of a
different depth), so the whole study can be re-run on another model without edits:
```bash
# reading (observer) on a second model — small model, small layer set
!MESE_MODEL=Qwen/Qwen2.5-0.5B-Instruct MESE_LAYERS=8,12,16 python step5_robust.py
# steering at a comparable depth
!MESE_MODEL=Qwen/Qwen2.5-0.5B-Instruct MESE_LAYER=12 MESE_QUICK=1 python step6_steer_all.py
```
Qwen2.5-0.5B has 24 layers, Llama-3.2-1B has 16, Qwen2.5-1.5B has 28 — pick layers at
a similar *relative* depth (~50%) and log which model produced each log.

`MESE_DRY=1` (step 4c above) runs a script with **no model at all**. For
`step5_robust.py` the activations are seeded noise, so all accuracies it prints are
~chance and meaningless. For `step6_steer_all.py` / `step7_sweep.py` it swaps in
`dry_model.py`, a 28-layer, d=64 char-level toy that honours the real hook API, and
injects a synthetic label signal *purely so the measurability gates open* and the
key/cosine/control/steering code actually executes — otherwise every axis would fail the
gate and the dry run would test nothing. It exists to catch wiring errors (an undefined
name, a broken control, a log that cannot be written, a treatment that is not applied)
in minutes on CPU instead of after a model download, and it writes
`results_step*_dry.txt` so a wiring check can never be mistaken for a real result.
A healthy step5 dry run must print **zero** `CONFOUND WARNING` / `LEAK WARNING` lines — on
noise the length and permutation controls sit at 0.31–0.50 and 0.47–0.49, nowhere near
their firing bars (length must reach the measured probe; permutation must reach 0.65).

Paste the full output of B steps 5, 6 and 7 back into the chat: each printed block becomes
a table in `results.md` / `paper.md`. The numbers that matter most:
- grouped MLP accuracy **minus** the `template-identity CONTROL` on the same axis/layer;
- the steering `slope` **minus** the `random` (sham) row's slope (must clearly exceed it).

## C. Optional: LLM-drafted extra hard sentences (treat as ideas, not labels)
```bash
python dataset_generate.py      # drafts -> dataset_draft.csv (pre-filtered by the gate)
python dataset_validate.py      # flags LEAK / ABSTRACT / FRAGMENT / DUP
python dataset_merge.py --review dataset_review.csv   # merge only un-flagged rows
```

## D. Key files
| file | role |
|---|---|
| `leak_lexicon.py` | ONE definition of "leaky"/abstract/concrete, shared by all data tools |
| `dataset_templates.py` | easy set (keyword-driven) + template group keys |
| `dataset_hard.py` | curated hard set (surface-neutral concrete events, mirrored pairs) |
| `dataset_merge.py` | merge + final gate -> `dataset.py` (`DATA`, `GROUPS`) |
| `probe_cv.py` | model-free CV harness (random vs grouped) + template-identity control |
| `steering_keys.py` | ONE definition of the steering-key family (mean/logreg/lda/probe-grad/-path/-class/random) shared by step6, step7 and `test_steering_collapse.py`, **plus `SteerController`**, which arms the hook per generation and counts how many generations really ran steered |
| `dry_model.py` | 28-layer, d=64 char-level toy LM that honours the real hook API, used only under `MESE_DRY=1` so step6/step7 can be wired-checked with no download |
| `test_grouped_cv.py` | CPU self-test of the split logic and the memorization gap |
| `test_steering_collapse.py` | CPU self-test of the §4 mechanism (rowspace theorem; `-path`/`-class` do NOT escape it; contingency contrast), the sham seed-collision trap, **section D: the dilution trap** (old loop steers 2 of 40; `SteerController` 40 of 40; `check()` raises on a shortfall), and **section E: device hygiene** (AST scan for `randn(device=+CPU generator)`, the CUDA-only crash) |
| `check_length_control.py` | CPU self-test (tokenizer only, no model): re-derives the length-only reading control at 10 seeds with its spread |
| `step5_robust.py` | Observer results: random row, grouped row, template-identity control row, plus per-axis label-permutation + length-only controls |
| `step6_steer_all.py` | Steerer results: measurability gate, 7 directions (incl. probe-grad/-path/-class), sham, fluency, `[steering audit]` |
| `step7_sweep.py` | Steering SEARCH: layer sweep x token-pooling (mean/last) x axis, same gate + sham; `MESE_SHAM=n` builds a sham distribution; per-cell label-permutation + length-only reading controls |
| `analyze_step7.py` | CPU noise analysis of `results_step7.txt` (real-vs-sham stats, sign consistency, permutation p) |
| `step8_dose.py` | DOSE-FINDING: was the Step 6 null real, or just under-scaled? Per-scale tone + fluency on a WIDE ladder, a sham **distribution** (`MESE_SHAM`), paired per-sample slopes, and a template-grouped instrument gate at step5's protocol (hidden=64, reps=10). Defaults to Care/Harm |
| `test_reading_controls.py` | CPU self-test of those reading controls (perm -> chance; length confound must be caught) |
| `test_probe_calibration.py` | CPU calibration: catches vacuous low-capacity controls; what a linear probe can earn here |
| `results.md` | results log incl. corrected findings |
| `paper.md` | draft paper |
