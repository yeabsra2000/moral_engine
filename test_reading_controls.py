"""Offline unit test for step7's READING CONTROLS (no model, CPU, ~10 s).

It exec()s the REAL function source out of step7_sweep.py (via ast) so the test
cannot drift from the script, then checks:
  1. a genuine signal scores high          -> probe works
  2. label-permutation control -> ~chance  -> no leakage in the CV
  3. length-only control -> ~chance when length is uninformative
  4. length-only control -> HIGH when length IS confounded (the last-token risk)
     i.e. the CONFOUND WARNING would fire, which is the point of the control.
"""
import ast, random, statistics
from collections import defaultdict

import torch

SRC = open("step7_sweep.py", encoding="utf-8").read()
tree = ast.parse(SRC)
WANT = {"group_folds", "probe_cv_acc", "control_acc"}
funcs = {n.name: ast.get_source_segment(SRC, n) for n in tree.body
         if isinstance(n, ast.FunctionDef) and n.name in WANT}
assert WANT <= set(funcs), f"missing {WANT - set(funcs)}"

ns = {"torch": torch, "random": random, "statistics": statistics,
      "defaultdict": defaultdict, "DEVICE": "cpu"}
for name in ("group_folds", "probe_cv_acc", "control_acc"):
    exec(funcs[name], ns)
probe_cv_acc, control_acc = ns["probe_cv_acc"], ns["control_acc"]

torch.manual_seed(0); random.seed(0)
N, D = 100, 8                              # 25 templates x 4 near-duplicate rows
groups = [f"t{i // 4}" for i in range(N)]
y = torch.tensor([1.0 if i % 2 else -1.0 for i in range(N)])

def make_X(signal=True):
    X = torch.randn(N, D)
    if signal:                             # class signal lives in dims 0-1
        X[:, 0] += y * 2.0
        X[:, 1] += y * 1.5
    return X

def se_informative():                      # length carries NO tone info
    return torch.tensor([float(10 + (i * 7) % 13) for i in range(N)])

def se_confounded():                       # length is a proxy for the label
    return torch.tensor([float(20 + 12 * float(y[i]) + random.gauss(0, 1))
                         for i in range(N)])

fails = []
def check(name, got, lo, hi):
    ok = lo <= got <= hi
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<52} {got:.2f}  (want {lo}-{hi})")
    if not ok:
        fails.append(name)

print("--- step7 reading controls: synthetic checks ---")
X = make_X(True)
check("real signal probe accuracy", probe_cv_acc(X, y, groups), 0.80, 1.01)
check("label-permutation control",
      control_acc(X, y, groups, se_informative(), "perm"), 0.20, 0.65)
check("length-only control (length uninformative)",
      control_acc(X, y, groups, se_informative(), "length"), 0.15, 0.65)
X0 = make_X(False)                         # no tone signal, length IS the proxy
check("length-only control (length confounded -> must catch it)",
      control_acc(X0, y, groups, se_confounded(), "length"), 0.60, 1.01)
check("no-signal probe accuracy", probe_cv_acc(X0, y, groups), 0.15, 0.65)
print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
raise SystemExit(1 if fails else 0)
