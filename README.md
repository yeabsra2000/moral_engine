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

1. **Reading works — and the recoverable structure is nonlinear.** Template-grouped CV
   MLP accuracy is **0.78–0.98** across layers 12/14/16, while a grouped *linear* probe
   gets **0.46–0.66** (Arousal and Fairness at chance). `test_probe_calibration.py` shows
   a synthetic set that is linearly separable *by construction* still earns only **0.75**
   under this protocol, so the linear probe is not simply undertrained.
2. **Steering is null, and the null survives a search.** Four direction estimators (mean,
   logistic, LDA, and the probe's own gradient) across layers {14, 20, 26} × {mean,
   last-token} pooling × 4 axes move tone no further than a random sham direction:
   permutation **p = 0.438** over the whole 24-cell grid.
3. **Two controls carry the result.** Random train/test splits are inflated to **0.76–0.93**
   by template identity *alone* — so grouped CV is the honest protocol. And a reading is
   only accepted if the length-only and label-permutation controls stay silent. Three
   control-design traps were found and fixed along the way (see `results.md` M7).

Full numbers, corrections and provenance: **`results.md`**. Write-up: **`paper.md`**.

## Reproduce (none of this needs a GPU)

```bash
python test_grouped_cv.py          # fold invariants + template-memorization control
python test_probe_calibration.py   # catches vacuous controls; calibrates the linear probe
python test_reading_controls.py    # length/permutation controls vs synthetic ground truth
python check_length_control.py     # re-derives the length control (tokenizer only, no model)
MESE_DRY=1 MESE_QUICK=1 python step5_robust.py   # end-to-end wiring check, no model (~5 min)
```

The GPU experiments (below) produce `results_step5.txt`, `results_step6.txt`,
`results_step7.txt`, which are committed and cited in `results.md`.

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
#    dataset.py  probe_cv.py  step5_robust.py  step6_steer_all.py  step7_sweep.py
#    analyze_step7.py  (CPU-only; it can equally be run locally on a downloaded log)
#    check_length_control.py  (CPU-only, tokenizer only -- re-derives the length control)
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

# 4c) OPTIONAL, but cheapest possible check: run step5 with NO model at all
#     (~5 min on CPU, measured 305 s). Activations are replaced by seeded noise, so it
#     only proves the script runs end to end (probes -> controls -> flags -> log)
#     before you spend GPU time. Its numbers are meaningless and it writes
#     results_step5_dry.txt, never the real log.
!MESE_DRY=1 MESE_QUICK=1 python step5_robust.py

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

# 8) nothing is lost if the session drops: the runs append to these logs
!ls -la results_step5.txt results_step6.txt results_step7.txt
!cat results_step7.txt | tail -40

# 9) download the logs to paste back into the chat
from google.colab import files
files.download('results_step5.txt'); files.download('results_step6.txt')
files.download('results_step7.txt')
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

`MESE_DRY=1` (step 4c above) runs `step5_robust.py` with **no model at all**: the
activations are seeded noise, so all accuracies it prints are ~chance and meaningless.
It exists to catch wiring errors (an undefined name, a broken control, a log that
cannot be written) in ~5 min on CPU (~305 s measured in QUICK mode) instead of after a
model download, and it writes
`results_step5_dry.txt` so a wiring check can never be mistaken for a real result.
A healthy dry run must print **zero** `CONFOUND WARNING` / `LEAK WARNING` lines — on
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
| `test_grouped_cv.py` | CPU self-test of the split logic and the memorization gap |
| `check_length_control.py` | CPU self-test (tokenizer only, no model): re-derives the length-only reading control at 10 seeds with its spread |
| `step5_robust.py` | Observer results: random row, grouped row, template-identity control row, plus per-axis label-permutation + length-only controls |
| `step6_steer_all.py` | Steerer results: measurability gate, 4 directions (incl. probe-grad), sham, fluency |
| `step7_sweep.py` | Steering SEARCH: layer sweep x token-pooling (mean/last) x axis, same gate + sham; `MESE_SHAM=n` builds a sham distribution; per-cell label-permutation + length-only reading controls |
| `analyze_step7.py` | CPU noise analysis of `results_step7.txt` (real-vs-sham stats, sign consistency, permutation p) |
| `test_reading_controls.py` | CPU self-test of those reading controls (perm -> chance; length confound must be caught) |
| `test_probe_calibration.py` | CPU calibration: catches vacuous low-capacity controls; what a linear probe can earn here |
| `results.md` | results log incl. corrected findings |
| `paper.md` | draft paper |
