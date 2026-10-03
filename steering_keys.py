"""
MESE-CR — steering-key estimators (no model dependency -> unit-testable offline)
===============================================================================
ONE definition of the "steering direction" family, shared by step6_steer_all.py,
step7_sweep.py and test_steering_collapse.py -- the same convention as
leak_lexicon.py for the data tools and probe_cv.py for the CV logic.

The paper's mechanism claim is about where these keys POINT, not only about how
well they steer, so keeping them in one place makes the cosine table in §4
reproducible from the scripts alone:

  mean              mean_pos - mean_neg                   (ActAdd-style)
  logreg            weights of a fitted logistic regression (discriminative)
  lda               whitened mean difference, Cov^-1 * d   (Fisher direction)
  probe-grad        d/dx logit of the NONLINEAR probe AT the positive-pole mean
  probe-grad-path   ... averaged ALONG the segment mu_neg -> mu_pos (interior pts)
  probe-grad-class  ... averaged at EVERY positive example, not just their mean
  random            SHAM control (must point nowhere near the mean)

Why `-path` and `-class` exist (the falsification test):
`probe-grad` is the steepest ascent of the READABLE nonlinear structure, but it
is evaluated at a SINGLE point -- and there the probe's Jacobian
W2 diag(SiLU'(W1 x0)) W1 is just a linear functional, which BCE has already
driven onto the class-separation direction. A point derivative cannot see
curvature; averaging the Jacobian along mu_neg -> mu_pos, or over the whole
positive class, can. So these two keys test the paper's central claim that
probe-based steering is confined to span(mu_pos - mu_neg): if they too collapse
(cos ~ 1 with `mean`) the confinement is a property of the method, and if they
break away but still do not steer, the nonlinear structure is real but not
causally reachable by additive steering.

All functions return an UNNORMALIZED vector unless their name says otherwise;
`all_keys` returns a dict of unit-norm keys, `mean` first (the reference that
the cosine table is reported against).
"""
import torch

# Distinctive on purpose: a sham generator seeded with 0 would reproduce the
# caller's own `torch.manual_seed(0); torch.randn(...)` stream exactly, turning
# the "random" control into the ground-truth direction. See `all_keys` below.
DEFAULT_SHAM_SEED = 20250802


def unit(v):
    return v / (v.norm() + 1e-8)


def cos(a, b):
    """Cosine similarity, ignoring scale. +1.000 = same direction, 0 = orthogonal."""
    af, bf = a.flatten(), b.flatten()
    return (af @ bf / (af.norm() * bf.norm() + 1e-12)).item()


# ---- the linear keys --------------------------------------------------------
def dir_mean(Xa, y):
    """ActAdd-style raw mean difference. Reference direction for the cos table."""
    return Xa[y > 0].mean(dim=0) - Xa[y < 0].mean(dim=0)


def dir_logreg(Xa, y, epochs=800, lr=1e-2):
    """Fitted logistic-regression weight vector (discriminative linear key).

    NOTE: features reaching this function are L2-normalized per row (see the
    callers) and the classes are balanced, so in the near-separable regime the
    BCE optimum satisfies w ~ (mu_pos - mu_neg) exactly. Measured cos(logreg,
    mean) = +1.000 on all four axes at layer 14 -- i.e. `logreg` is NOT an
    independent steering key, it is the mean key up to scale. Stated in §4.
    """
    dev = Xa.device
    Xd = Xa.to(dev)
    yt = (y.to(dev) > 0).float()
    w = torch.zeros(Xd.size(1), device=dev, requires_grad=True)
    b = torch.zeros(1, device=dev, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr, weight_decay=1e-3)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Xd @ w + b, yt)
        loss.backward(); opt.step()
    return w.detach()


def dir_lda(Xa, y, shrink=1e-1):
    """Fisher direction: (within-class Cov + shrink*I)^-1 (mu_pos - mu_neg).

    Uses ALL the data, unlike mean_pos - mean_neg which only sees the means --
    so it differs from `mean` by a whitening. Measured cos(lda, mean) = 0.43-0.73,
    the only key that points meaningfully elsewhere, and it still does not steer.
    """
    dev = Xa.device
    Xd, yd = Xa.to(dev), y.to(dev)
    pos, neg = Xd[yd > 0], Xd[yd < 0]
    mu_p, mu_n = pos.mean(dim=0), neg.mean(dim=0)
    d = mu_p - mu_n
    centered = torch.cat([pos - mu_p, neg - mu_n], dim=0)
    n = centered.size(0)
    cov = (centered.t() @ centered) / max(n - 2, 1)
    lam = shrink * torch.trace(cov) / cov.size(0)
    A = cov + lam * torch.eye(cov.size(0), device=cov.device, dtype=cov.dtype)
    return torch.linalg.solve(A, d)


# ---- the nonlinear (probe) keys ---------------------------------------------
def dir_probe_grad(net, Xa, y):
    """Supervised, nonlinear key: the probe's input gradient at the positive-pole
    mean (steepest ascent of the READABLE structure, evaluated at ONE point).

    net: the trained measuring probe (Sequential(Linear -> SiLU -> Linear)).
    Returns a unit-norm vector. This is the key step6/step7 already used.
    """
    dev = next(net.parameters()).device
    yd = (y.to(dev) > 0)
    pos = Xa.to(dev)[yd]
    x0 = pos.mean(dim=0).detach().clone().requires_grad_(True)
    net(x0.unsqueeze(0)).squeeze(0).sum().backward()   # autograd through SiLU is fine
    return unit(x0.grad.detach())


def dir_probe_grad_path(net, Xa, y, n_steps=17):
    """Probe gradient averaged ALONG the segment mu_neg -> mu_pos (interior only).

    The endpoints are excluded on purpose: including mu_pos would blend the plain
    `probe-grad` key back in and blur the contrast. Averaging the Jacobian along
    the path is what a point derivative cannot do -- it is the natural "does the
    probe have usable curvature?" test.
    """
    dev = next(net.parameters()).device
    Xd = Xa.to(dev)
    yd = (y.to(dev) > 0)
    mu_n, mu_p = Xd[~yd].mean(dim=0), Xd[yd].mean(dim=0)
    ts = torch.linspace(0.0, 1.0, n_steps, device=dev)[1:-1]     # interior points
    g = torch.zeros_like(mu_p)
    for t in ts:
        x = (mu_n + t * (mu_p - mu_n)).detach().clone().requires_grad_(True)
        net(x.unsqueeze(0)).squeeze(0).sum().backward()
        g = g + x.grad.detach()
    return unit(g / max(int(ts.numel()), 1))


def dir_probe_grad_class(net, Xa, y):
    """Probe gradient averaged at EVERY positive example (a distribution, not a mean).

    For a LINEAR probe this is the weight vector at every point, i.e. it reduces
    to `logreg`/`mean`; for an MLP it can differ, which is exactly what makes it
    a test of whether the collapse is an artifact of the single evaluation point.
    """
    dev = next(net.parameters()).device
    pos = Xa.to(dev)[(y.to(dev) > 0)].detach().clone().requires_grad_(True)
    net(pos).squeeze(-1).sum().backward()
    return unit(pos.grad.detach().mean(dim=0))


# ---- the whole key set for one (axis, layer, pooling) cell ------------------
def sham_vector(shape, seed, device):
    """A deterministic SHAM direction for `shape`, allocated on `device`.

    Drawn on CPU and then moved, for two reasons:

    1. BUG FIX (2026-10-03, caught only on the first CUDA run). The natural form is
           torch.randn(shape, device=device, generator=torch.Generator().manual_seed(s))
       which raises "RuntimeError: Expected a 'cuda' device type for generator but
       found 'cpu'" -- `torch.Generator()` defaults to CPU and cannot feed a CUDA
       draw. No CPU-only test can reach that path, which is why it survived until the
       GPU run. Keeping the draw on CPU and moving it sidesteps the whole issue.
    2. A control should be reproducible across devices: this way the same seed gives
       the *identical* sham vector whether the script runs on CPU or CUDA, so a
       result can be reproduced exactly rather than approximately.
    """
    v = torch.randn(shape, generator=torch.Generator().manual_seed(int(seed)))
    return v.to(device)      # `.to` is a no-op when the device already matches


def all_keys(Xa, y, net, sham_seed=DEFAULT_SHAM_SEED):
    """Every steering key for one cell, unit-norm, `mean` first.

    `mean` is first because it is the reference the §4 cosine table reports
    against. logreg/lda/probe-grad-* are pure vector math on ALREADY-CAPTURED
    activations, so adding them to a grid costs no extra generation -- only
    actually STEERING with a key costs GPU time.

    `sham_seed` is a DISTINCTIVE constant on purpose. A caller that does
    `torch.manual_seed(0); w = torch.randn(d)` and then seeds a sham generator
    with 0 gets the sham == w exactly, i.e. a "random" control that is the
    ground-truth direction (measured cos(sham, w) = +1.000). See
    test_steering_collapse.py, which asserts the sham stays orthogonal.
    """
    base = dir_mean(Xa, y)
    keys = {
        "mean":             base,
        "logreg":           dir_logreg(Xa, y),
        "lda":              dir_lda(Xa, y),
        "probe-grad":       dir_probe_grad(net, Xa, y),
        "probe-grad-path":  dir_probe_grad_path(net, Xa, y),
        "probe-grad-class": dir_probe_grad_class(net, Xa, y),
        # NB: must go through sham_vector -- `torch.randn(..., device=base.device,
        # generator=torch.Generator())` crashes on CUDA (CPU generator, CUDA draw).
        "random":           sham_vector(base.shape, sham_seed, base.device),
    }
    return {k: unit(v) for k, v in keys.items()}


def cos_table(keys, ref="mean"):
    """[(name, cos vs ref)] for every key except the reference itself."""
    r = keys[ref]
    return [(k, cos(v, r)) for k, v in keys.items() if k != ref]


def print_cos_table(keys, ref="mean", indent="    "):
    """Print where each key POINTS relative to the reference (the §4 evidence)."""
    print(f"{indent}(where each key POINTS: cos vs the {ref} key; "
          f"+1.000 = same direction as the plain mean difference)")
    for k, c in cos_table(keys, ref):
        print(f"{indent}  cos({k:>16}, {ref}) = {c:+.3f}")


# ---- steering hook controller -------------------------------------------------
# WHY THIS CLASS EXISTS -- the "dilution trap".
# The natural way to write the steering loop is
#     for scale in SCALES:
#         attach_steer(vec, scale)        # arm ONCE per scale
#         for p in PROMPTS:
#             for _ in range(SAMPLES):
#                 text = generate(p)      # steered for the FIRST sample only
#                 attach_steer(vec, 0.0)  # disarm, so the tone read is honest
# It arms once and disarms on the first sample, so only 1 in len(PROMPTS)*SAMPLES
# generations is actually steered. With 5 prompts x 4 samples that is 1 in 20, so a
# real effect of size e reads as about e/20 -- right at the sham noise floor. The
# result is a "clean null" that is an artifact of the loop, with every control in
# the world unable to detect it, because the bug dilutes the TREATMENT, not the
# measurement. Measured offline by counting live-hook state: 2 of the 40
# generations that should have been steered were steered. See
# test_steering_collapse.py::test_steering_hook_is_live_for_every_steered_generation.
#
# SteerController makes arming PER GENERATION structural: `generate` arms, calls the
# generator, then disarms (so the caller still gets an honest steer-off read for
# free), and counts both the generations that should have been steered and the ones
# that were. `check()` turns a shortfall into a hard error instead of a silent null.

class SteerController:
    """Owns one steering hook; counts how many generations were really steered."""

    def __init__(self, layers):
        self.layers = layers        # indexable modules with register_forward_hook
        self.handle = None
        self.vec = None
        self.scale = 0.0
        self.layer = None
        self.n_gen = 0              # generations attempted
        self.n_should_steer = 0     # generations at scale != 0
        self.n_steered = 0          # generations that ran with the hook live

    def is_armed(self):
        return self.handle is not None

    def disarm(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def arm(self, vec, scale, layer):
        """(Re)arm the hook. A falsy `scale` (0/None) means 'no steering at all'."""
        self.disarm()
        if not scale:
            return
        self.vec, self.scale, self.layer = vec, float(scale), layer

        def _h(mod, inp, out):
            # decoder layer returns (hidden_states, past_kv); weights may sit on cpu
            if isinstance(out, tuple):
                h, rest = out[0], out[1:]
                return ((h + self.scale * self.vec.to(h.device)),) + rest
            return out + self.scale * self.vec.to(out.device)

        self.handle = self.layers[layer].register_forward_hook(_h)

    def generate(self, prompt, vec, scale, layer, generate_fn):
        """ARM -> generate -> DISARM. The caller cannot forget to re-arm.

        The hook is already removed when this returns, so the returned text can be
        measured with the steer OFF (an honest read) without the caller doing
        anything -- which is exactly the step that used to break the loop.
        """
        self.arm(vec, scale, layer)
        self.n_gen += 1
        self.n_should_steer += int(bool(scale))
        self.n_steered += int(self.is_armed())
        try:
            return generate_fn(prompt)
        finally:
            self.disarm()

    def check(self):
        """Raise if any generation that should have been steered was not."""
        if self.n_steered != self.n_should_steer:
            raise AssertionError(
                f"STEERING UNDER-APPLIED: {self.n_steered} of "
                f"{self.n_should_steer} generations that should have been steered "
                f"were steered (of {self.n_gen} total). A real effect would be "
                f"diluted by {self.n_steered / max(self.n_should_steer, 1):.3f}x and "
                "would read as a null. This is the 'dilution trap' -- arm the hook "
                "per GENERATION (see SteerController.generate).")
        return self.n_steered, self.n_gen

