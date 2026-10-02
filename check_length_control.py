"""
MESE-CR — length control, checked exactly and to 10 seeds (CPU only, NO model)
==============================================================================
The length-only reading control in `step5_robust.py` uses one standardized column of
TOKEN COUNTS. That column depends only on the tokenizer and the text, never on the
model's weights, so its control accuracy can be re-derived off-GPU — which is what this
script does, at 10 seeds instead of the run's 3, and with the spread reported.

It exists because a single mean cannot distinguish "length really carries signal on this
axis" from "one noisy fold", and this control is the one that decides whether a reading
survives. It also documents a trap: the control deviates in EITHER direction. Arousal
scores ~0.27 while its two classes have IDENTICAL token length, so "far from 0.50" is
not by itself evidence of a confound — only the comparison against the measured probe is.

Downloads only the ~7 MB tokenizer (never the 3 GB model). Numbers appear in results.md
(M6, "Reading controls on the model").
"""
import os, sys, statistics, torch

# ---- dependency bootstrap ---------------------------------------------------
# Same inlined guard as step5/6/7 (see the long comment in step5_robust.py): a script
# uploaded to Colab WITHOUT its siblings must say which file is missing and where Python
# looked, instead of dying with a bare "ModuleNotFoundError: No module named 'dataset'".
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:                              # pasted into a notebook cell
    _HERE = os.getcwd()
for _p in (_HERE, os.getcwd()):
    if _p not in sys.path:
        sys.path.insert(0, _p)
_REQUIRED = ["dataset.py", "probe_cv.py"]
_MISSING = [f for f in _REQUIRED
            if not any(os.path.exists(os.path.join(_p, f))
                       for _p in (_HERE, os.getcwd()))]
if _MISSING:
    raise SystemExit(
        "\nMISSING FILE(S): %s\n"
        "  This script must sit in the same folder as the rest of the project.\n"
        "    script folder : %s\n"
        "    working dir   : %s\n"
        "  In Colab, upload the repo files into the working directory -- at minimum\n"
        "  THIS script + %s -- then re-run.  Check with:  !ls -la *.py\n"
        % (", ".join(_MISSING), _HERE, os.getcwd(), " + ".join(_REQUIRED)))

from dataset import DATA, GROUPS
from probe_cv import cv_rep                 # shared, unit-tested CV logic
from transformers import AutoTokenizer

SEEDS = 10
AXES = ["Valence", "Arousal", "Care/Harm", "Fairness"]

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
LEN = [len(tok(s)["input_ids"]) for s, _, _ in DATA]
_se = torch.tensor(LEN, dtype=torch.float32)
LENFEAT = ((_se - _se.mean()) / (_se.std() + 1e-8)).view(-1, 1)   # same as step5
LABELS = torch.tensor([l for _, l, _ in DATA], dtype=torch.float32)

print(f"token length: mean {_se.mean():.1f}  sd {_se.std():.1f}  "
      f"min {int(_se.min())}  max {int(_se.max())}")
print(f"step5_robust.py uses this exact feature; {SEEDS} seeds here vs REPS=3 in QUICK.\n")
for ax in range(4):
    active = LABELS[:, ax].ne(0)
    y = (LABELS[active, ax] > 0).float()
    groups = [GROUPS[i] for i in torch.nonzero(active).view(-1).tolist()]
    accs = [cv_rep(LENFEAT[active], y, 64, s, groups) for s in range(SEEDS)]
    pos = [LEN[i] for i in range(len(LEN)) if LABELS[i, ax] > 0]
    neg = [LEN[i] for i in range(len(LEN)) if LABELS[i, ax] < 0]
    print(f"{AXES[ax]:<10} n={int(active.sum()):>3} groups={len(set(groups)):>3}  "
          f"length-only (grouped): mean={statistics.fmean(accs):.3f} "
          f"sd={statistics.stdev(accs):.3f}  min={min(accs):.2f} max={max(accs):.2f}")
    print(f"{'':<10}   tokens: pos {statistics.fmean(pos):5.1f} (sd {statistics.stdev(pos):4.1f})"
          f"   neg {statistics.fmean(neg):5.1f} (sd {statistics.stdev(neg):4.1f})")

print("\nRead this against the grouped MLP probe, not against 0.50 alone:")
print("  V 0.92 | A 0.72 | C 0.98 | F 0.79   (layer 14, QUICK run)")
print("Only Care/Harm separates by length at all; the others are anti-correlated or")
print("flat, which is why their controls sit far from 0.50 in the OTHER direction too.")
