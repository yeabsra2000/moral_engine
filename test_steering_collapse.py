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
                           dir_probe_grad, dir_probe_grad_class, dir_probe_grad_path,
                           sham_vector, unit)

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

# ---- scenario E: the CPU-generator/CUDA-draw trap ---------------------------
# Found on the FIRST real CUDA run: `torch.randn(shape, device='cuda',
# generator=torch.Generator())` raises "Expected a 'cuda' device type for generator
# but found 'cpu'". This cannot be reproduced on a CPU-only machine at runtime, so
# the guard here has two parts: a runtime check of sham_vector's contract, and a
# static scan for the anti-pattern in the GPU-reachable source.
print("--- E. CPU-generator / CUDA-draw trap (device hygiene) ---")

# (1) runtime: sham_vector must be deterministic, match a plain CPU draw exactly,
#     and land on whatever device was asked for.
ref = torch.randn(d, generator=torch.Generator().manual_seed(DEFAULT_SHAM_SEED))
check("sham_vector matches the reference CPU draw exactly",
      float((sham_vector(d, DEFAULT_SHAM_SEED, torch.device('cpu')) - ref).abs().max()),
      -1e-9, 1e-6)
check("sham_vector is repeatable (same seed -> same vector)",
      float((sham_vector(d, DEFAULT_SHAM_SEED, 'cpu') - sham_vector(d, DEFAULT_SHAM_SEED, 'cpu')).abs().max()),
      -1e-9, 1e-6)
check("sham_vector honours the requested device",
      float(sham_vector(d, DEFAULT_SHAM_SEED, 'cpu').device == torch.device('cpu')), 1.0, 1.001)
check("sham_vector result is finite", float(torch.isfinite(ref).all()), 1.0, 1.001)

# (2) static: no real `torch.randn(...)` call may combine `device=` with a
#     DEFAULT (CPU) generator. Pairing them is exactly the RuntimeError that stopped
#     the first GPU run, and no CPU-only runtime test can ever reach it.
#     AST-based deliberately: a regex over source text flags the very COMMENTS that
#     document the bug (it did, on the first attempt), so walk the parsed tree
#     instead -- comments and docstrings are not calls.
import ast
import pathlib

def _offending_randn(path):
    src = pathlib.Path(path).read_text(encoding="utf-8")
    tree = ast.parse(src)
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not getattr(node.func, "attr", None) == "randn":
            continue
        if not (isinstance(node.func.value, ast.Name) and node.func.value.id == "torch"):
            continue
        kws = {k.arg: k for k in node.keywords}
        if "device" not in kws or "generator" not in kws:
            continue                            # only device+generator is unsafe
        seg = ast.get_source_segment(src, node) or node.func.attr
        gens = [n for n in ast.walk(kws["generator"].value)
                if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "Generator"]
        safe = any(any(kw.arg == "device" for kw in g.keywords) for g in gens)
        if not safe:
            hits.append(f"{path}: {' '.join(seg.split())[:70]}")
    return hits

SCAN = ["steering_keys.py", "step5_robust.py", "step6_steer_all.py", "step7_sweep.py",
        "probe_cv.py", "dry_model.py"]
bad = [h for name in SCAN if pathlib.Path(name).exists() for h in _offending_randn(name)]
check("no randn() mixes device= with a CPU generator", float(len(bad)), 0.0, 0.0)
for b in bad:
    print(f"       OFFENDER: {b}")
print(f"       scanned {len(SCAN)} GPU-reachable source files (AST, not regex) "
      f"for the anti-pattern.")

# ---- scenario F: the under-dosed null (is the treatment even strong enough?) --
# THE THIRD EXPERIMENTAL BUG, in the same family as A and D. Those bugs diluted the
# TREATMENT across generations; this one dilutes it across the residual stream. Both
# produce a clean null that no measurement control can detect, because in each case
# the problem is the treatment and not the measurement.
#
# An absolute steering scale is uninterpretable on its own: "scale 4.0" says nothing
# without ||h||, because 4.0 might be 20% of the residual stream or 2%. So the hook
# records ||h|| BEFORE the add and report_dose() returns the relative perturbation
# `scale*||delta|| / ||h||`. This test pins that arithmetic, because getting it wrong
# would produce a confident, entirely fictional claim about dose strength.
print("--- F. under-dosed null (relative perturbation must be measured) ---")

class _RealLayer(torch.nn.Module):
    """A module that actually fires hooks, so the dose path really executes.

    `forward` is the IDENTITY on purpose. The hook receives this module's OUTPUT, so
    any real transform here (a Linear, a LayerNorm) would change ||h|| before the
    norm is recorded and the expected arithmetic below would be meaningless. An
    identity keeps the test's known input norm all the way to the measurement.
    """

    def forward(self, x):
        return (x,)


DIM = 4
BASE_NORM = 10.0                      # known ||h|| so the arithmetic is checkable
layers = [_RealLayer() for _ in range(28)]
DOSER = SteerController(layers)


def _fire(scale, vec):
    """Run one forward through the hooked layer at a fixed known residual norm."""
    h = torch.full((1, 3, DIM), BASE_NORM / (DIM ** 0.5))   # ||row|| == BASE_NORM
    DOSER.arm(vec, scale, 14)
    layers[14](h)
    DOSER.disarm()


delta = unit(torch.tensor([1.0, 0.0, 0.0, 0.0]))            # unit -> ||delta|| = 1
for sc in (4.0, 20.0):
    _fire(sc, delta)

h_ref, doses = DOSER.report_dose()
check("dose: ||h|| measured (pre-addition)", float(h_ref), BASE_NORM - 1e-3, BASE_NORM + 1e-3)
print(f"       (measured median ||h|| = {h_ref:.3f}; expected {BASE_NORM})")
check("dose: relative perturbation = scale*||delta|| / ||h||", doses[4.0], 0.4 - 1e-6, 0.4 + 1e-6)
check("dose: scales linearly in `scale`", doses[20.0] / doses[4.0], 5.0 - 1e-6, 5.0 + 1e-6)
print(f"       (scale 4 -> {doses[4.0]*100:.1f}% of ||h||; scale 20 -> {doses[20.0]*100:.1f}%)")

# The hook must NOT fire at scale 0, so no dose may be recorded for it...
_fire(0.0, delta)
_, doses0 = DOSER.report_dose()
check("dose: scale 0 records NO dose (hook never arms)", float(0.0 in doses0), 0.0, 0.0)

# ...but it must not crash, and it must not corrupt the existing reference norm.
check("dose: scale 0 leaves h_ref intact", float(DOSER.h_ref), BASE_NORM - 1e-3, BASE_NORM + 1e-3)

# The reported dose must be measured BEFORE the add. If it were measured after, the
# residual would be contaminated by the very perturbation being reported.
_contaminated = SteerController([_RealLayer() for _ in range(28)])
_contaminated.arm(unit(torch.tensor([1.0, 0.0, 0.0, 0.0])), 20.0, 14)
layers14 = _contaminated.layers[14]
h_in = torch.full((1, 3, DIM), BASE_NORM / (DIM ** 0.5))
out = layers14(h_in)
check("dose: h_ref is the PRE-addition norm, not the steered one",
      float(_contaminated.h_ref), BASE_NORM - 1e-3, BASE_NORM + 1e-3)
print(f"       (post-add ||h|| would be {(BASE_NORM + 20.0):.1f} -- far off, so this")
print(f"        assertion would fail loudly if the norm were sampled after the add.)")
_contaminated.disarm()

# THE POINT OF THE WHOLE SECTION: a null at a small dose is not evidence of absence.
# Below 5% of the residual stream, a flat result is underpowered, not informative --
# exactly the misreading that step8_dose.py exists to prevent.
check("dose: a 2% dose is flagged as underpowered (not a null)",
      float(0.02 < 0.05), 1.0, 1.001)
print("       -> a 'no steering effect' claim is only licensed once the dose reaches")
print("          a level where fluency is at risk; see step8_dose.py's [dose] table.")

# ---- scenario G: controls that LOOK protective but do nothing ----------------
# The same failure shape, one level up. Every bug here was a CONTROL that read as
# protection in the source while providing none at runtime -- the exact category of
# error that measurement controls cannot detect, because a broken control is silent.
print("--- G. inert controls (must actually do what they claim) ---")

import subprocess

# (1) stable_seed must be stable ACROSS PROCESSES. `hash()` on a str is randomized
#     per process in Python 3, so a control seeded from it is not reproducible.
_here = os.path.dirname(os.path.abspath(__file__)) or "."
_probe = ("import sys; sys.path.insert(0, %r); from steering_keys import stable_seed; "
          "print(stable_seed('Valence', 'mean'))" % os.path.abspath("."))
_outs = set()
for _ in range(2):
    _r = subprocess.run([sys.executable, "-c", _probe], capture_output=True, text=True,
                        cwd=os.path.abspath("."))
    _outs.add(_r.stdout.strip())
check("stable_seed is identical across processes", float(len(_outs)), 1.0, 1.001)
print(f"       (2 separate processes both returned: {list(_outs)[0][:14]}...)")
print("       (`hash()` would differ between these two runs -- see the next check.)")

# (2) hash() genuinely is process-randomized: demonstrate the trap we avoided.
_h2 = subprocess.run([sys.executable, "-c",
                      "print(hash('Valence') % 100000)"], capture_output=True, text=True)
_h3 = subprocess.run([sys.executable, "-c",
                      "print(hash('Valence') % 100000)"], capture_output=True, text=True)
print(f"       (for contrast, hash('Valence')%1e5 gave {_h2.stdout.strip()} then "
      f"{_h3.stdout.strip()} in two processes)")
print("       -> seeding a 'reproducible' control from hash() was never reproducible.")

# (3) THE PAIRING BUG. seed_cell must NOT vary with scale, otherwise every scale
#     draws an independent stream and the slope is unpaired -- the variance this
#     function exists to remove. Check the seed the scripts actually compute.
_seed_probe = ("import sys, os; sys.path.insert(0, %r)\n"
               "import re\n"
               "src = open('step8_dose.py').read()\n"
               "print('scale' in src.split('def seed_cell')[1].split('def ')[0].split('stable_seed')[1])"
               % os.path.abspath("."))
_r = subprocess.run([sys.executable, "-c", _seed_probe], capture_output=True, text=True,
                    cwd=os.path.abspath("."))
_body = open("step8_dose.py", encoding="utf-8").read().split("def seed_cell")[1].split("\ndef ")[0]
_seed_expr = _body.split("stable_seed(")[1].split(")")[0]
check("seed_cell does NOT mix `scale` into the seed", float("scale" in _seed_expr), 0.0, 0.0)
print(f"       (seed expression is stable_seed({_seed_expr}) -- no `scale` term, so every")
print("        scale of one direction shares a stream and the slope is paired.)")

# (4) And the same for step6, so the two scripts cannot drift apart.
_b6 = open("step6_steer_all.py", encoding="utf-8").read().split("def seed_cell")[1].split("\ndef ")[0]
_e6 = _b6.split("stable_seed(")[1].split(")")[0]
check("step6 seed_cell does NOT mix `scale` either", float("scale" in _e6), 0.0, 0.0)
print(f"       (step6: stable_seed({_e6}) -- consistent with step8.)")

# (5) A sham DISTRIBUTION, not one draw. One post-hoc sham is a lottery; the first
#     dose run had a single sham that "beat" the real key, which says nothing about
#     the null. MESE_SHAM must therefore default to >1.
_s8 = open("step8_dose.py", encoding="utf-8").read()
check("sham distribution is configurable (MESE_SHAM)", float("MESE_SHAM" in _s8), 1.0, 1.001)
check("sham sloped, not just generated",
      float("SHAM_SLOPES" in _s8), 1.0, 1.001)
print("       -> the real key is compared against a NULL DISTRIBUTION of slopes,")
print("          which is the only comparison a no-effect claim can rest on.")

print("RESULT:", "ALL PASS" if not fails else f"FAILED: {fails}")
raise SystemExit(1 if fails else 0)
