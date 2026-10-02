"""
MESE-CR — M2: Steer ALL 4 axes + fluency check
==============================================
Rounds out the Steerer. For each axis (toward POSITIVE pole, layer 14):
  - trains a measuring probe (tone reader; steer measured with it OFF),
  - REPORTS that probe's held-out accuracy first. If the probe cannot read the
    axis above MIN_PROBE_ACC, the axis is declared UNMEASURABLE and skipped --
    otherwise "steering failed" would be indistinguishable from "measurement
    is noise" (exactly the ambiguity in the first version of this script),
  - computes the steering delta in FIVE ways and runs all of them:
        mean      = mean_pos - mean_neg            (original ActAdd style)
        logreg    = weights of a fitted logistic regression (discriminative)
        lda       = whitened mean difference (Cov^-1 * d, Fisher direction)
        probe-grad= gradient of the nonlinear measuring probe's logit at the
                    positive-pole mean (supervised direction that respects the
                    READABLE, nonlinear structure)
        random    = SHAM control: random unit vector, same scale (must NOT move
                    tone; if it does, the measurement is noise)
  - applies each at scale in {0, 1.5, 4}, samples text, and reports:
        (a) probe-measured tone move (should rise with scale)
        (b) FLUENCY via mean per-token log-likelihood (should stay high)

Run in Google Colab (GPU). Requires dataset.py in the same folder.
"""
import os, sys, random, statistics, datetime

# ---- dependency bootstrap: actionable message instead of a bare traceback ----
# (inlined deliberately -- see the same block in step5_robust.py for the rationale)
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:                              # pasted into a notebook cell
    _HERE = os.getcwd()
for _p in (_HERE, os.getcwd()):
    if _p not in sys.path:
        sys.path.insert(0, _p)
_REQUIRED = ["dataset.py"]
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

from collections import defaultdict
from dataclasses import dataclass
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from dataset import DATA, GROUPS          # GROUPS = template/motif key per row
except ModuleNotFoundError:                    # dataset.py truly missing -> real error
    raise
except ImportError:                            # older dataset.py without GROUPS
    from dataset import DATA
    GROUPS = None

print("CUDA:", torch.cuda.is_available())
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if DEVICE == "cuda" else torch.float32
MODEL_ID = os.environ.get("MESE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
LAYER = int(os.environ.get("MESE_LAYER", "14"))   # steering layer (see N_LAYERS check below)
AXES = ["Valence", "Arousal", "Care/Harm", "Fairness"]
SCALES = [0.0, 1.5, 4.0]
DIRECTIONS = ["mean", "logreg", "lda", "probe-grad", "random"]
PROMPTS = [
    "The long hike ended in heavy rain, but I still",
    "Our project did not win the award, but the team",
    "I missed the bus this morning, but I found",
    "When the letter finally arrived, she opened it and",
    "The village meeting ended late, and on the way home",
]
SAMPLES = 4          # generations per prompt per scale
GEN_LEN = 18
MIN_PROBE_ACC = 0.60  # below this the axis cannot be measured -> no steering claim

# ---- QUICK mode: MESE_QUICK=1 python step6_steer_all.py  (smoke test first!) ----
QUICK = os.environ.get("MESE_QUICK", "") not in ("", "0", "false", "False")
if QUICK:
    PROMPTS = PROMPTS[:3]
    SAMPLES = 2
    SCALES = [0.0, 4.0]
    DIRECTIONS = ["mean", "probe-grad", "random"]

# ---- tee everything to a log file: a dropped Colab session must not lose the run ----
LOG = os.environ.get("MESE_LOG", "results_step6.txt")
class _Tee:
    """Mirror stdout into a log file; never let logging break the run."""
    def __init__(self, *streams): self.streams = streams
    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
            except Exception:
                pass
        return len(s)
    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass
_log = open(LOG, "a", encoding="utf-8")
_log.write(f"\n===== step6 run {datetime.datetime.now():%Y-%m-%d %H:%M:%S} "
           f"quick={QUICK} =====\n")
sys.stdout = _Tee(sys.__stdout__, _log)
print(f"[log] writing to {LOG}  |  QUICK={QUICK} "
      f"(prompts={len(PROMPTS)}, samples={SAMPLES}, scales={SCALES}, directions={DIRECTIONS})")
print("Full run: ~20-40 min on a T4. QUICK run: ~4-6 min. Keep the browser tab open.")

torch.manual_seed(0); random.seed(0)
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map="auto")
model.eval()

# ---- depth guard: a different model may have fewer layers than MESE_LAYER ----
N_LAYERS = len(model.model.layers)
if not (0 <= LAYER < N_LAYERS):
    sys.exit(f"LAYER={LAYER} but {MODEL_ID} has {N_LAYERS} layers "
             f"(valid 0..{N_LAYERS - 1}); pass MESE_LAYER=<int>.")
print(f"model={MODEL_ID}  depth={N_LAYERS}  steering layer={LAYER}")

# ---- capture layer-14 mean activations per sentence ----
captured = []
def hook(mod, args):
    hs = args[0]
    if isinstance(hs, tuple): hs = hs[0]
    captured.append(hs.detach().float().mean(dim=1).squeeze(0).to("cpu"))
    return args
handle = model.model.layers[LAYER].register_forward_pre_hook(hook)
for sentence, _, _ in DATA:
    ids = tok(sentence, return_tensors="pt").to(model.device)
    with torch.no_grad(): model(**ids)
handle.remove()
X = torch.stack(captured).to(DEVICE); X = X / (X.norm(dim=1, keepdim=True) + 1e-8)
LAB = torch.tensor([lbl for _, lbl, _ in DATA], dtype=torch.float32)

def make_probe(Xa, y):  # small MLP measuring probe on the active samples for this axis
    xs, ys = Xa, (y > 0).float().to(Xa.device)
    net = torch.nn.Sequential(torch.nn.Linear(X.size(1), 32), torch.nn.SiLU(), torch.nn.Linear(32, 1)).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(500):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(xs).view(-1), ys)
        loss.backward(); opt.step()
    net.eval(); return net

# ---- is this axis even MEASURABLE? (template-grouped held-out probe accuracy) ----
def group_folds(y, groups, k=5):
    """Assign whole groups to folds, stratified by class (honest split)."""
    folds = [[] for _ in range(k)]
    if groups is None:
        idx = list(range(y.numel()))
        random.Random(0).shuffle(idx)
        for j, i in enumerate(idx):
            folds[j % k].append(i)
        return folds
    lab_of = {}
    for g, lab in zip(groups, y.tolist()):
        lab_of[g] = lab
    by_class = defaultdict(list)
    for g, lab in lab_of.items():
        by_class[lab].append(g)
    for lab in sorted(by_class):
        gs = by_class[lab][:]
        random.Random(hash(lab) % 1000).shuffle(gs)
        for j, g in enumerate(gs):
            folds[j % k].extend([i for i, gg in enumerate(groups) if gg == g])
    return folds

def probe_cv_acc(Xa, y, groups, hidden=32, k=5):
    """Mean held-out accuracy of the measuring probe (steer irrelevant here)."""
    accs = []
    for f, te in enumerate(group_folds(y, groups, k)):
        tr = [i for i in range(y.numel()) if i not in set(te)]
        if not tr or not te:
            continue
        net = torch.nn.Sequential(torch.nn.Linear(Xa.size(1), hidden), torch.nn.SiLU(),
                                  torch.nn.Linear(hidden, 1)).to(DEVICE)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        ytr = (y[tr] > 0).float().to(DEVICE)
        for _ in range(500):
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(Xa[tr]).view(-1), ytr)
            loss.backward(); opt.step()
        with torch.no_grad():
            pred = (torch.sigmoid(net(Xa[te]).view(-1)) > 0.5).long().cpu()
        accs.append((pred == (y[te] > 0).long().cpu()).float().mean().item())
    return statistics.fmean(accs) if accs else float("nan")

# ---- four ways to estimate the steering direction ----
def dir_mean(Xa, y):
    return Xa[y > 0].mean(dim=0) - Xa[y < 0].mean(dim=0)

def dir_logreg(Xa, y, epochs=800, lr=1e-2):
    w = torch.zeros(Xa.size(1), device=DEVICE, requires_grad=True)
    b = torch.zeros(1, device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr, weight_decay=1e-3)
    t = (y > 0).float().to(DEVICE)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Xa @ w + b, t)
        loss.backward(); opt.step()
    return w.detach()

def dir_lda(Xa, y, shrink=1e-1):
    """Fisher direction: (within-class Cov + shrink*I)^-1 (mu_pos - mu_neg).
    Uses ALL the data, unlike mean_pos-mean_neg which only sees means."""
    pos, neg = Xa[y > 0], Xa[y < 0]
    mu_p, mu_n = pos.mean(dim=0), neg.mean(dim=0)
    d = mu_p - mu_n
    dev = torch.cat([pos - mu_p, neg - mu_n], dim=0)
    n = dev.size(0)
    cov = (dev.t() @ dev) / max(n - 2, 1)
    lam = shrink * torch.trace(cov) / cov.size(0)
    A = cov + lam * torch.eye(cov.size(0), device=cov.device, dtype=cov.dtype)
    return torch.linalg.solve(A, d)

def dir_probe_grad(net, Xa, y):
    """Supervised, nonlinear steering direction from the measuring probe's gradient.

    The readable tone structure is nonlinear (grouped linear ~ chance, MLP far
    above). A mean/logreg/LDA key is a single hyperplane; this instead returns the
    direction in activation space that most increases the probe's positive-pole
    logit AT the positive-pole mean (the point we want to steer toward), so it
    respects the probe's learned nonlinear manifold.

    net: the already-trained measuring probe (Sequential(Linear->SiLU->Linear)).
    Xa : this axis's active mean-pooled activations (DEVICE, dense).
    y  : this axis's labels; which rows are the positive pole (> 0).
    Returns unit-norm gradient vector (d,).
    """
    dev = next(net.parameters()).device
    pos = Xa.to(dev)[y > 0]                       # positive-pole examples
    x0 = pos.mean(dim=0).detach().clone().requires_grad_(True)  # (d,) evaluation point
    logit = net(x0.unsqueeze(0)).squeeze(0)       # scalar logit of the probe
    (logit.sum()).backward()                      # autograd through SiLU is fine
    g = x0.grad.detach()
    return unit(g)                               # normalized steering key

def unit(v):
    return v / (v.norm() + 1e-8)

def tone(net, v):
    dev = next(net.parameters()).device
    with torch.no_grad():
        p = torch.sigmoid(net(v.to(dev).unsqueeze(0))).item()   # measuring probe, steer OFF
    return 2.0 * p - 1.0

def capture_vec(text):
    """capture mean-pooled layer-14 activation of text (steer OFF for honest tone)."""
    cap = []
    h = model.model.layers[LAYER].register_forward_pre_hook(
        lambda m, a: (cap.append(a[0].detach().float().mean(dim=1).squeeze(0).to("cpu")), a)[1])
    ids = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad(): model(**ids)
    h.remove()
    v = cap[0]; return v / (v.norm() + 1e-8)

@dataclass
class Ctx: dirn=None; handle=None
steer_ctx = Ctx()
def attach_steer(vec, scale):
    if steer_ctx.handle: steer_ctx.handle.remove(); steer_ctx.handle = None
    if scale:
        steer_ctx.dirn = vec
        s = scale
        def _h(mod, inp, out):
            # decoder layer returns (hidden_states, past_kv); weights may sit on cpu (device_map)
            if isinstance(out, tuple):
                h, rest = out[0], out[1:]
                return ((h + s * steer_ctx.dirn.to(h.device)),) + rest
            return out + s * steer_ctx.dirn.to(out.device)
        steer_ctx.handle = model.model.layers[LAYER].register_forward_hook(_h)

def generate(prompt):
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=GEN_LEN, do_sample=True, temperature=0.8,
                         top_p=0.9, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids['input_ids'].shape[1]:], skip_special_tokens=True)

def fluency(text):
    """mean per-token log-likelihood of the text (higher = more fluent English)."""
    ids = tok(text, return_tensors="pt").to(model.device)['input_ids']
    with torch.no_grad():
        logits = model(ids).logits
    logp = torch.nn.functional.log_softmax(logits[:, :-1], dim=-1)
    return logp.gather(-1, ids[:, 1:].unsqueeze(-1)).mean().item()

print("=" * 78)
print(f"Steering toward POSITIVE on all 4 axes @ layer {LAYER}. directions = {DIRECTIONS}")
print("tone in -1..1 (measured with the steer hook OFF); fluency = mean per-token logp.")
print(f"Measuring-probe gate: template-grouped held-out accuracy must be >= {MIN_PROBE_ACC}.")
print("=" * 78)

CELLS = []   # (axis, n pos/neg, probe acc, direction, slope, fluency)
for axis in range(4):
    act = LAB[:, axis].ne(0)
    y = LAB[:, axis][act].to(DEVICE)
    Xa = X[act]
    groups = ([GROUPS[i] for i in torch.nonzero(act).view(-1).tolist()] if GROUPS else None)
    n_pos, n_neg = int((y > 0).sum()), int((y < 0).sum())
    acc = probe_cv_acc(Xa, y, groups)
    print(f"\n### {AXES[axis]}: {int(act.sum())} sentences ({n_pos} pos / {n_neg} neg), "
          f"{len(set(groups)) if groups else int(act.sum())} groups")
    print(f"    measuring probe held-out accuracy = {acc:.2f}   (gate {MIN_PROBE_ACC})")
    if not (acc >= MIN_PROBE_ACC):
        print("    -> AXIS NOT MEASURABLE: skipping steering. Any 'effect' here could not")
        print("       be told apart from probe noise, so reporting one would be dishonest.")
        CELLS.append((AXES[axis], f"{n_pos}/{n_neg}", f"{acc:.2f}", "unmeasurable", "-", "-"))
        continue
    net = make_probe(Xa, y)
    base = dir_mean(Xa, y)
    dirs = {"mean": base, "logreg": dir_logreg(Xa, y), "lda": dir_lda(Xa, y),
            "probe-grad": dir_probe_grad(net, Xa, y),
            "random": torch.randn_like(base)}
    dirs = {k: unit(v) for k, v in dirs.items()}
    for k in ("logreg", "lda", "probe-grad", "random"):
        cos = torch.nn.functional.cosine_similarity(
            dirs[k].view(1, -1), dirs["mean"].view(1, -1)).item()
        print(f"    cos({k:>6}, mean) = {cos:+.3f}")
    for name in DIRECTIONS:
        delta = dirs[name].to(model.dtype)
        tones, flu = {}, {}
        for scale in SCALES:
            attach_steer(delta, scale)
            ts, fs = [], []
            for p in PROMPTS:
                for _ in range(SAMPLES):
                    t = generate(p)
                    attach_steer(delta, 0.0)        # clear steer to measure honestly
                    ts.append(tone(net, capture_vec(t)))
                    fs.append(fluency(t))
            attach_steer(delta, 0.0)
            tones[scale], flu[scale] = statistics.mean(ts), statistics.mean(fs)
        slope = tones[SCALES[-1]] - tones[SCALES[0]]
        line = "  ".join(f"{s:>3}={tones[s]:+.2f}" for s in SCALES)
        print(f"  [{name:>6}] tone  {line}   slope={slope:+.2f}   "
              f"fluency={statistics.mean(list(flu.values())):.2f}")
        CELLS.append((AXES[axis], f"{n_pos}/{n_neg}", f"{acc:.2f}", name, f"{slope:+.2f}",
                      f"{statistics.mean(list(flu.values())):.2f}"))

print("\n" + "=" * 78)
print("SUMMARY  (slope = tone at max scale - tone at 0; 'random' is the SHAM control)")
print(f"{'axis':<10} {'n +/-':<9} {'probe':>5}  {'direction':>11}  {'slope':>6}  {'fluency':>7}")
for r in CELLS:
    print(f"{r[0]:<10} {r[1]:<9} {r[2]:>5}  {r[3]:>11}  {r[4]:>6}  {r[5]:>7}")
print("A real steer = slope clearly larger than the SHAM row with fluency near baseline.")
print("If the sham moves tone as much as the real directions, the probe is measuring")
print("noise, not tone -- and the honest conclusion is 'not measurable at this scale'.")

