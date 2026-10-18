"""
MESE-CR — local (CPU, no model) test for the split logic
========================================================
Run:  python tests/test_grouped_cv.py  (from the repo root)

Two things are verified, both on the REAL dataset.py labels/GROUPS:

1. make_folds() invariants: the folds cover every sentence exactly once, and
   under grouped splits no template/motif appears in two folds (no paraphrase
   leakage) and each fold keeps a sane positive rate.

2. template-memorization magnitude, on a SYNTHETIC feature space that contains
   ONLY the template identity (+ small noise). Such features have zero real
   signal about tone, but under RANDOM CV a probe can memorize template->label
   and score far above chance. Under GROUPED CV the held-out templates carry no
   signal, so accuracy must fall back to ~0.50. The difference is the size of
   the inflation that random-split probe numbers could contain.

Expected: grouped ~= 0.50 for every axis, random clearly higher.
"""
import os, sys, statistics
import torch

# ---- dependency bootstrap: actionable message instead of a bare traceback ----
# (inlined deliberately -- see the same block in step5_robust.py for the rationale)
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:                              # pasted into a notebook cell
    _HERE = os.getcwd()
# Repo root: this file lives in tests/, its imports live at the root. Anchor on
# the required file so a notebook paste (flat layout) resolves to the cwd too.
_ROOT = _HERE if os.path.exists(os.path.join(_HERE, "dataset.py")) \
    else os.path.dirname(_HERE)
for _p in (_HERE, _ROOT, os.getcwd()):
    if _p not in sys.path:
        sys.path.insert(0, _p)
_REQUIRED = ["dataset.py", "probe_cv.py"]
_MISSING = [f for f in _REQUIRED
            if not any(os.path.exists(os.path.join(_p, f))
                       for _p in (_HERE, _ROOT, os.getcwd()))]
if _MISSING:
    raise SystemExit(
        "\nMISSING FILE(S): %s\n"
        "  This test must run from the repo root (next to its data/logic files).\n"
        "    script folder : %s\n"
        "    working dir   : %s\n"
        "  Check with:  !ls -la *.py\n"
        % (", ".join(_MISSING), _HERE, os.getcwd()))

from probe_cv import make_folds, cv_rep, template_only_features, KFOLD
from dataset import DATA, GROUPS

AXES = ["Valence", "Arousal", "Care/Harm", "Fairness"]
REPS = 3


def check_fold_invariants():
    n = len(DATA)
    LAB = torch.tensor([lbl for _, lbl, _ in DATA], dtype=torch.float32)
    for ax in range(4):
        active = LAB[:, ax].ne(0)
        idx = active.nonzero().view(-1).tolist()
        y = (LAB[active][:, ax] > 0).float()
        groups = [GROUPS[i] for i in idx]
        for seed in range(3):
            for name, grp in (("random-rows", None), ("grouped", groups)):
                # For the random branch we must also restrict to the active rows,
                # so build folds on the same y either way; `make_folds` tests the
                # split logic, not the row subset.
                folds = make_folds(y, seed, grp)
                assert len(folds) == KFOLD
                flat = sorted(i for f in folds for i in f)
                assert flat == list(range(y.numel())), \
                    f"ax{ax} {name}: folds must cover every sample exactly once"
                sizes = [len(f) for f in folds]
                assert min(sizes) >= 5, f"ax{ax} {name}: degenerate fold sizes {sizes}"
                rates = [float((y[f] > 0).float().mean()) for f in folds]
                spread = max(rates) - min(rates)
                assert spread < 0.30, \
                    f"ax{ax} {name} seed{seed}: class balance spread too large ({spread:.2f})"
                if name == "grouped":
                    # every template/motif must live in exactly ONE fold, and its
                    # members must not be scattered (that would be paraphrase leakage)
                    where = {}
                    for f, fold in enumerate(folds):
                        for i in fold:
                            where.setdefault(groups[i], set()).add(f)
                    bad = {g: s for g, s in where.items() if len(s) != 1}
                    assert not bad, f"group split leaks across folds: {bad}"
    print("PASS: fold invariants (coverage, no leakage, non-degenerate + balanced "
          "folds on EVERY axis)")


def check_template_memorization():
    feats = template_only_features(GROUPS)
    print(f"\nSynthetic feature = one-hot(template) + noise; {feats.size(1)} templates, "
          f"{feats.size(0)} sentences, {REPS} reps x {KFOLD} folds")
    print(f"{'axis':<10} {'n':>4} {'random CV':>10} {'grouped CV':>11} {'gap':>7}")
    rows = []
    for ax in range(4):
        y = torch.tensor([lbl[ax] for _, lbl, _ in DATA], dtype=torch.float32)
        act = y.ne(0)
        Xa, ya = feats[act], y[act]
        groups = [GROUPS[i] for i in act.nonzero().view(-1).tolist()]
        rnd = statistics.fmean([cv_rep(Xa, ya, 64, s, None) for s in range(REPS)])
        grp = statistics.fmean([cv_rep(Xa, ya, 64, s, groups) for s in range(REPS)])
        rows.append((AXES[ax], int(act.sum()), rnd, grp))
        print(f"{AXES[ax]:<10} {int(act.sum()):>4} {rnd:>10.2f} {grp:>11.2f} {rnd - grp:>+7.2f}")
    ok_grouped = all(abs(g - 0.5) <= 0.20 for _, _, _, g in rows)
    ok_gap = all(r - g >= 0.15 for _, _, r, g in rows)
    assert ok_grouped, "template-only features should be ~chance under grouped CV"
    assert ok_gap, "random CV should be inflated by template memorization"
    print("\nPASS: template identity alone is ~chance on unseen templates, but strongly")
    print("      inflated under random splits -> grouped CV is the honest protocol.")


if __name__ == "__main__":
    check_fold_invariants()
    check_template_memorization()
    print("\nALL LOCAL CHECKS PASSED")
