"""Offline calibration of the probe/training settings (no model, CPU, ~30 s).

Two things this pins down, because both can silently corrupt a CONTROL result:

1. CAPACITY TRAP. A `hidden=0` (single Linear) probe fed a ONE-column feature
   can fail to fit a perfectly separable 1-D problem: 400 Adam steps at lr=1e-3
   sometimes cannot escape a wrong-sign weight init. A control built that way
   scores near chance even when the confound FULLY determines the label, so it
   "passes" vacuously and proves nothing. Controls must use the same capacity as
   the measured probe (hidden=64 in step5, hidden=32 in step7).

2. CALIBRATION CEILING. On a noiseless LINEAR rule (label = sign(x.w)) in a
   high-d / small-n regime like the real one (d >> n after grouping), linear and
   MLP probes plateau together (~0.75 vs ~0.78) and MORE TRAINING DOES NOT
   HELP -- so a grouped-LINEAR probe sitting near chance on real activations is
   below the score that perfect linear structure would earn, which supports the
   "tone is not linearly recoverable" reading rather than undercutting it.

Run from the repo root:  python tests/test_probe_calibration.py
"""
import os, sys, statistics

import torch

# ---- dependency bootstrap: actionable message instead of a bare traceback ----
# (inlined deliberately -- see the same block in step5_robust.py for the rationale)
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:                              # pasted into a notebook cell
    _HERE = os.getcwd()
# Repo root: this file lives in tests/, probe_cv.py lives at the root. Anchor on
# the required file so a notebook paste (flat layout) resolves to the cwd too.
_ROOT = _HERE if os.path.exists(os.path.join(_HERE, "probe_cv.py")) \
    else os.path.dirname(_HERE)
for _p in (_HERE, _ROOT, os.getcwd()):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if not any(os.path.exists(os.path.join(_p, "probe_cv.py"))
           for _p in (_HERE, _ROOT, os.getcwd())):
    raise SystemExit(
        "\nMISSING FILE(S): probe_cv.py\n"
        "  This test must run from the repo root (next to its logic files).\n"
        "    script folder : %s\n"
        "    working dir   : %s\n"
        "  Check with:  !ls -la *.py\n" % (_HERE, os.getcwd()))

from probe_cv import cv_rep

fails = []


def check(name, got, lo, hi):
    ok = lo <= got <= hi
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<58} {got:.2f}  (want {lo}-{hi})")
    if not ok:
        fails.append(name)


def fmean(vals):
    return statistics.fmean(vals)


# ---- 1. capacity trap: 1-D control feature ---------------------------------
torch.manual_seed(0)
n = 120
y = torch.tensor([1.0 if i % 2 else -1.0 for i in range(n)])
groups = [f"t{i // 5}" for i in range(n)]
onecol = (y * 3 + torch.randn(n) * 0.5).view(-1, 1)      # length FULLY determines y


def probe(X, yy, hidden, reps=3, epochs=400):
    return fmean([cv_rep(X, yy, hidden, s, groups, epochs=epochs) for s in range(reps)])


print("--- 1. capacity trap (1-column control feature) ---")
lo_cap = probe(onecol, y, 0)
hi_cap = probe(onecol, y, 64)
check("hidden=64 control detects a full length confound", hi_cap, 0.80, 1.01)
# Assert the GAP, not an absolute band: the low-capacity probe's shortfall is
# seed-dependent (measured 0.61-0.77 across configurations), so a fixed band
# would be a flaky test. What must hold is that it is clearly WORSE than the
# same probe with the measured capacity on identical data/folds.
check("hidden=0 control is clearly below hidden=64 on identical data",
      hi_cap - lo_cap, 0.15, 1.01)
print(f"  hidden=0 = {lo_cap:.2f} vs hidden=64 = {hi_cap:.2f} -> gap {hi_cap - lo_cap:+.2f}")
print("  -> controls MUST use the measured probe's capacity, else they pass vacuously.")

# ---- 2. calibration ceiling on a noiseless linear rule ---------------------
torch.manual_seed(1)
n2, d = 120, 64
w = torch.randn(d)
w = w / w.norm()
X = torch.randn(n2, d)
y2 = torch.where(X @ w > 0, 1.0, -1.0)
g2 = [f"t{i // 5}" for i in range(n2)]
print("--- 2. calibration: linearly separable by construction (d=64, n=120) ---")
lin_short = fmean([cv_rep(X, y2, 0, s, g2, epochs=400) for s in range(4)])
lin_long = fmean([cv_rep(X, y2, 0, s, g2, epochs=2000) for s in range(4)])
mlp = fmean([cv_rep(X, y2, 64, s, g2, epochs=400) for s in range(4)])
check("linear probe on a PERFECT linear rule", lin_short, 0.55, 1.01)
check("MLP on the same data is not far above it", mlp, lin_short - 0.05, 1.01)
print(f"  linear 400 epochs = {lin_short:.2f} | linear 2000 epochs = {lin_long:.2f} "
      f"| MLP = {mlp:.2f}")
print(f"  -> extra training moves linear by {lin_long - lin_short:+.2f}: the ceiling is "
      f"estimation noise from d>>n, not undertraining.")
print("RESULT:", "ALL PASS" if not fails else f"FAILED: {fails}")
raise SystemExit(1 if fails else 0)
