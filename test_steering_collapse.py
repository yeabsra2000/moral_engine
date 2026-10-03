"""Offline test of the STEERING-COLLAPSE mechanism (no model, CPU, ~30 s).

Locks in the §4 mechanism claim, which is the paper's spine: every probe-based
steering key is confined to a fixed subspace, so probe-based steering cannot
access the nonlinear structure that makes tone READABLE.

Three separately-checkable facts:

1. ROWSPACE THEOREM (exact, machine precision). For a probe
   f(x) = W2 . SiLU(W1 x + b1) + b2 the input gradient is
       grad f(x) = W1^T . diag(SiLU'(W1 x + b1)) . W2^T
   which lies in rowspace(W1) for EVERY x. So any key built by averaging
   grad f over any point set -- one point, a path, or a whole class -- is in
   rowspace(W1) and differs from the others only by the per-unit scalars
   SiLU'(z_i). Asserted by projecting onto rowspace(W1) via QR.

2. THE OBVIOUS FIX DOES NOT WORK. `probe-grad-path` (Jacobian averaged along
   mu_neg -> mu_pos) and `probe-grad-class` (averaged over every positive
   example) were added to test whether the single-point collapse is an artifact
   of the evaluation point. They are mutually cos 0.9985-1.0000 with the plain
   `probe-grad`, i.e. they do NOT escape it. Consequence for the paper: the
   nonlinear key was not left untested.

3. THE COLLAPSE ONTO `mean` IS CONTINGENT, NOT A TAUTOLOGY. On activations where
   the boundary is nonlinear, the probe keys point well AWAY from `mean`
   (measured cos ~0.44), while still lying entirely in rowspace(W1). So the
   theorem explains why the probe keys coincide with EACH OTHER; BCE training
   aligning W1 with the class-separation direction is what explains why they
   coincide with `mean` on real activations (0.85-0.98). Two separate claims --
   do not conflate them.

Also guards the seed-collision trap found while building this: seeding a sham
generator with the same seed the caller used for the ground-truth direction
reproduces that direction exactly (cos = +1.000), i.e. a "random" control that
IS the answer.

Run from the repo root:  python test_steering_collapse.py
"""
import os, sys

import torch

# ---- dependency bootstrap: actionable message instead of a bare traceback ----
# (inlined deliberately -- see the same block in step5_robust.py for the rationale)
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:                              # pasted into a notebook cell
    _HERE = os.getcwd()
for _p in (_HERE, os.getcwd()):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if not any(os.path.exists(os.path.join(_p, "steering_keys.py"))
           for _p in (_HERE, os.getcwd())):
    raise SystemExit(
        "\nMISSING FILE(S): steering_keys.py\n"
        "  This test must run from the repo root (next to its logic files).\n"
        "    script folder : %s\n"
        "    working dir   : %s\n"
        "  Check with:  !ls -la *.py\n" % (_HERE, os.getcwd()))

from steering_keys import (DEFAULT_SHAM_SEED, SteerController, all_keys, cos, dir_mean,
                           dir_probe_grad, dir_probe_grad_class, dir_probe_grad_path)

fails = []


def check(name, got, lo, hi):
    ok = lo <= got <= hi
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<60} {got:+.3f}  (want {lo} to {hi})")
    if not ok:
        fails.append(name)


def make_probe(Xn, y, hidden=32, iters=600, seed=7):
    """Same shape/capacity as the measuring probes in step5/step6/step7."""
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(Xn.size(1), hidden), torch.nn.SiLU(),
                              torch.nn.Linear(hidden, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    yt = (y > 0).float()
    for _ in range(iters):
        opt.zero_grad()
        torch.nn.functional.binary_cross_entropy_with_logits(net(Xn).view(-1), yt).backward()
        opt.step()
    net.eval()
    return net


def outside_rowspace(v, W1):
    """Fraction of ||v|| that lies outside rowspace(W1). 0 = fully contained."""
    basis = torch.linalg.qr(W1.t()).Q            # (d, hidden) orthonormal
    proj = basis @ (basis.t() @ v)
    return ((v - proj).norm() / (v.norm() + 1e-12)).item()


# ---- scenario A: linearly separable by construction ------------------------
torch.manual_seed(0)
d, n = 64, 120
w = torch.randn(d); w = w / w.norm()
X = torch.randn(n, d)
y = torch.where(X @ w > 0, 1.0, -1.0)
Xn = X / (X.norm(dim=1, keepdim=True) + 1e-8)    # callers L2-normalize per row
netA = make_probe(Xn, y)
keysA = all_keys(Xn, y, netA)
print("--- A. linearly separable rule (the calibration scenario) ---")
check("logreg is NOT independent of mean", cos(keysA["logreg"], keysA["mean"]), 0.85, 1.001)
check("sham stays orthogonal (catches seed collision)",
      cos(keysA["random"], keysA["mean"]), -0.20, 0.20)
print("  probe keys vs plain probe-grad (the 'obvious fix' test):")
for k in ("probe-grad-path", "probe-grad-class"):
    check(f"  {k} does NOT escape the point key",
          cos(keysA[k], keysA["probe-grad"]), 0.99, 1.001)

print("--- A2. rowspace theorem (exact) ---")
for k in ("probe-grad", "probe-grad-path", "probe-grad-class"):
    check(f"  {k} lies in rowspace(W1)", outside_rowspace(keysA[k], netA[0].weight), -1e-9, 1e-6)
check("control: `mean` is NOT confined to rowspace(W1)",
      outside_rowspace(keysA["mean"], netA[0].weight), 0.10, 1.001)

# ---- scenario B: NONLINEAR boundary, class means still along x1 -------------
# A V-shaped ("wedge") region: the class means differ along x1, the decision
# boundary is not a hyperplane. This is how we separate the two claims -- the
# theorem (probe keys coincide with each other) from the empirical collapse
# (probe keys coincide with `mean`).
torch.manual_seed(0)
Xb = torch.randn(n, d)
Xb[:, 2:] *= 0.3
lab = torch.where(Xb[:, 0] > Xb[:, 1].abs() + 0.5, 1.0,
                  torch.where(Xb[:, 0] < -Xb[:, 1].abs() - 0.5, -1.0,
                              torch.zeros(n)))
keep = lab != 0
Xb, yb = Xb[keep], lab[keep]
Xbn = Xb / (Xb.norm(dim=1, keepdim=True) + 1e-8)
netB = make_probe(Xbn, yb)
keysB = all_keys(Xbn, yb, netB)

print("--- B. nonlinear `wedge` boundary ---")
with torch.no_grad():
    accB = ((torch.sigmoid(netB(Xbn).view(-1)) > 0.5).long()
            == (yb > 0).long()).float().mean().item()
check("probe actually reads this nonlinear axis (train acc)", accB, 0.90, 1.001)
for k in ("probe-grad-path", "probe-grad-class"):
    check(f"  {k} still does not escape the point key",
          cos(keysB[k], keysB["probe-grad"]), 0.99, 1.001)
for k in ("probe-grad", "probe-grad-path", "probe-grad-class"):
    check(f"  {k} lies in rowspace(W1)", outside_rowspace(keysB[k], netB[0].weight), -1e-9, 1e-6)
# The CONTINGENCY claim: on a nonlinear boundary the probe key points away from
# `mean` while still being inside rowspace(W1). If this ever sits ~1.0, the
# theorem alone would explain the real-data collapse and claim #3 would be wrong.
check("collapse onto `mean` is CONTINGENT (points away here)",
      cos(keysB["probe-grad"], keysB["mean"]), -1.001, 0.80)
print(f"  (measured cos(probe-grad, mean) = {cos(keysB['probe-grad'], keysB['mean']):+.3f} "
      f"vs {cos(keysA['probe-grad'], keysA['mean']):+.3f} on the linear rule)")

# ---- scenario C: the seed-collision trap, as a regression guard -------------
print("--- C. seed-collision trap (regression guard) ---")
torch.manual_seed(0)
w_same = torch.randn(d)
r_same = torch.randn(d, generator=torch.Generator().manual_seed(0))
check("sham seeded 0 == the caller's own direction (cos)", cos(r_same, w_same), 0.999, 1.001)
check("DEFAULT_SHAM_SEED is not 0", float(DEFAULT_SHAM_SEED != 0), 1.0, 1.001)
r_fix = torch.randn(d, generator=torch.Generator().manual_seed(DEFAULT_SHAM_SEED))
check("default sham seed avoids the collision", abs(cos(r_fix, w_same)), -1e-9, 0.20)
print("  -> a sham generator seeded with the caller's seed IS the ground truth.")
print("     `all_keys` uses a distinctive seed and the orthogonality check above.")

# ---- scenario D: the dilution trap (steering under-application) -------------
# Found while wiring MESE_DRY. The old loop armed the hook ONCE per scale and
# disarmed it on the first sample (to read tone honestly), so only 1 in
# len(PROMPTS)*SAMPLES generations was ever really steered. With 5 prompts x 4
# samples that is 1 in 20: a real effect of size e reads as ~e/20, exactly at the
# sham noise floor, and the null looks "clean" no matter how many controls you add
# -- because the bug dilutes the TREATMENT, not the measurement.
print("--- D. dilution trap: was the steer actually applied? ---")


class _FakeHandle:
    def __init__(self, layer, fn):
        self.layer, self.fn = layer, fn

    def remove(self):
        if self.fn in self.layer.post:
            self.layer.post.remove(self.fn)


class _FakeLayer:
    """Minimal stand-in for a decoder block: only register/remove matter here."""

    def __init__(self):
        self.post = []

    def register_forward_hook(self, fn):
        self.post.append(fn)
        return _FakeHandle(self, fn)

    def is_armed(self):
        return bool(self.post)


PROMPTS, SAMPLES, SCALES = 5, 4, [0.0, 1.5, 4.0]

# (1) Reproduce the OLD pattern and show how badly it under-applies.
layer, handle = _FakeLayer(), [None]


def _old_attach(scale):
    if handle[0]:
        handle[0].remove()
        handle[0] = None
    if scale:
        handle[0] = layer.register_forward_hook(lambda mod, inp, out: out)


n_gen = n_steered = n_should = 0
for scale in SCALES:
    _old_attach(scale)                     # arm once per scale
    for _ in range(PROMPTS):
        for _ in range(SAMPLES):
            n_gen += 1
            n_should += int(bool(scale))
            n_steered += int(layer.is_armed())    # <- the generation happens here
            _old_attach(0.0)                      # disarm to read tone honestly
    _old_attach(0.0)
check("OLD loop: steered / should-have (fraction)", n_steered / n_should, 0.0, 0.06)
print(f"       OLD loop steered only {n_steered} of {n_should} "
      f"({n_gen} generations) -> a real effect is diluted "
      f"{n_steered / n_should:.3f}x.")

# (2) SteerController must steer EVERY generation that should be steered.
STEER = SteerController([_FakeLayer() for _ in range(28)])
for scale in SCALES:
    for _ in range(PROMPTS):
        for _ in range(SAMPLES):
            STEER.generate("prompt", torch.zeros(4), scale, 14, lambda p: "text")
check("SteerController: hooked generations / should-have", STEER.n_steered / STEER.n_should_steer,
      1.0, 1.001)
check("SteerController: disarmed after generate (honest read)", float(not STEER.is_armed()), 1.0, 1.001)
steered, total = STEER.check()             # must not raise
check("SteerController.check() passes when fully applied",
      float(steered), float(STEER.n_should_steer), float(STEER.n_should_steer))
print(f"       (check() returned n_steered={steered}, n_gen={total}; "
      f"{STEER.n_should_steer} of those were scale!=0 and ALL were steered)")

# (3) ... and check() must LOUDLY fail if a generation is skipped, because the
#     whole point is that this bug is otherwise invisible in the results.
bad = SteerController([_FakeLayer() for _ in range(4)])
bad.n_gen, bad.n_should_steer, bad.n_steered = 20, 16, 1     # simulate the old loop
try:
    bad.check()
    check("SteerController.check() raises on under-application", 0.0, 1.0, 2.0)
except AssertionError as exc:
    check("SteerController.check() raises on under-application", 1.0, 1.0, 2.0)
    print(f"       (message: {str(exc).splitlines()[0][:76]}...)")

print("RESULT:", "ALL PASS" if not fails else f"FAILED: {fails}")
raise SystemExit(1 if fails else 0)
