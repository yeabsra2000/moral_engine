"""
MESE-CR — M2: Steer ALL 4 axes + fluency check
==============================================
Rounds out the Steerer. For each axis (toward POSITIVE pole, layer 14):
  - trains a measuring probe (tone reader; steer measured with it OFF),
  - REPORTS that probe's held-out accuracy first. If the probe cannot read the
    axis above MIN_PROBE_ACC, the axis is declared UNMEASURABLE and skipped --
    otherwise "steering failed" would be indistinguishable from "measurement
    is noise" (exactly the ambiguity in the first version of this script),
  - computes the steering delta in SEVEN ways and runs them all. The first five
    are the original key set; the last two exist to TEST the paper's claim that
    probe-based steering is confined to span(mu_pos - mu_neg) (see §4 and
    steering_keys.py):
        mean             = mean_pos - mean_neg        (original ActAdd style)
        logreg           = weights of a fitted logistic regression
        lda              = whitened mean difference (Cov^-1 * d, Fisher)
        probe-grad       = gradient of the nonlinear measuring probe's logit at
                           the positive-pole mean -- ONE evaluation point
        probe-grad-path  = ... averaged ALONG the segment mu_neg -> mu_pos
        probe-grad-class = ... averaged at EVERY positive example
        random           = SHAM control: random unit vector, same scale (must NOT
                           move tone; if it does, the measurement is noise)
    Where each key POINTS (cos vs `mean`) is printed before any steering, because
    that cosine table -- not the slopes -- is the paper's mechanism evidence, and
    it costs no generation at all.
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
_REQUIRED = ["dataset.py", "steering_keys.py"]
if os.environ.get("MESE_DRY", "") not in ("", "0", "false", "False"):
    _REQUIRED.append("dry_model.py")     # only a DRY run needs it; see MESE_DRY below
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
from steering_keys import all_keys, print_cos_table, unit, SteerController

print("CUDA:", torch.cuda.is_available())
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if DEVICE == "cuda" else torch.float32
MODEL_ID = os.environ.get("MESE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
LAYER = int(os.environ.get("MESE_LAYER", "14"))   # steering layer (see N_LAYERS check below)
AXES = ["Valence", "Arousal", "Care/Harm", "Fairness"]
SCALES = [0.0, 1.5, 4.0]
DIRECTIONS = ["mean", "logreg", "lda", "probe-grad",
              "probe-grad-path", "probe-grad-class", "random"]
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

# ---- MESE_DRY=1: swap the real model for dry_model.py's stub ------------------
# Same call surface (hooks / generate / logits) but a 28-layer d=64 char-level toy,
# so the key + cosine + control + steering code can be wiring-checked on a laptop
# CPU with no download. Its numbers are MEANINGLESS and it logs to its own file.
# Read BEFORE the QUICK block below, because a DRY run also wants the tiny grid.
DRY = os.environ.get("MESE_DRY", "") not in ("", "0", "false", "False")

# ---- QUICK mode: MESE_QUICK=1 python step6_steer_all.py  (smoke test first!) ----
# A DRY run implies QUICK: the point is the code paths, not the numbers.
QUICK = os.environ.get("MESE_QUICK", "") not in ("", "0", "false", "False") or DRY
if QUICK:
    PROMPTS = PROMPTS[:3]
    SAMPLES = 2
    SCALES = [0.0, 4.0]
    DIRECTIONS = ["mean", "probe-grad", "random"]

# ---- tee everything to a log file: a dropped Colab session must not lose the run ----
# A DRY run gets its OWN log by default, so a wiring check can never mix into (or be
# mistaken for) real model output. Override with MESE_LOG if you want it elsewhere.
LOG = os.environ.get("MESE_LOG",
                     "results_step6_dry.txt" if DRY else "results_step6.txt")
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
           f"quick={QUICK} dry={DRY} =====\n")
sys.stdout = _Tee(sys.__stdout__, _log)
print(f"[log] writing to {LOG}  |  QUICK={QUICK} DRY={DRY} "
      f"(prompts={len(PROMPTS)}, samples={SAMPLES}, scales={SCALES}, directions={DIRECTIONS})")
if DRY:
    print("*** DRY RUN: model NOT loaded, activations are a toy stub. Results are MEANINGLESS.")
else:
    print("Full run: ~20-40 min on a T4. QUICK run: ~4-6 min. Keep the browser tab open.")

torch.manual_seed(0); random.seed(0)
if DRY:
    print("[DRY RUN] model NOT loaded; dry_model.py stub (28 layers, d=64, char vocab).")
    print("          Validates plumbing (hooks -> capture -> keys -> steering -> log) only.")
    from dry_model import build_dry
    tok, model = build_dry()
else:
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
if DRY:
    # The stub's activations carry no tone signal, so every axis would fail the
    # measurability gate and the key/steering code below would never execute --
    # i.e. the wiring check would test nothing that matters. Inject a synthetic
    # label-correlated component ONLY so the gates open. Still meaningless.
    _gs = torch.Generator().manual_seed(7)
    # drawn on CPU (the generator is CPU), then moved: a CPU `_sig` added to a CUDA X
    # raises a device mismatch, and MESE_DRY=1 is explicitly encouraged on Colab.
    _dirs = torch.randn(4, X.size(1), generator=_gs).to(X.device)
    _dirs = _dirs / _dirs.norm(dim=1, keepdim=True)
    _sig = sum(LAB[:, a].to(X.device).unsqueeze(1) * _dirs[a].unsqueeze(0)
               for a in range(4))
    X = X + 1.5 * _sig
    X = X / (X.norm(dim=1, keepdim=True) + 1e-8)
    print("[DRY RUN] injected a synthetic label-correlated component into X so the axis")
    print("          gates open and the key/cosine/steering code actually runs.")

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

# ---- steering-direction keys -----------------------------------------------
# All key estimators live in steering_keys.py (ONE definition, shared with
# step7_sweep.py and test_steering_collapse.py): dir_mean, dir_logreg, dir_lda,
# dir_probe_grad, dir_probe_grad_path, dir_probe_grad_class, and `all_keys`.
# They were duplicated here before; keeping two copies is how the two scripts
# silently drift apart.

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

# ---- steering hook: owned by SteerController (steering_keys.py) --------------
# The controller ARMS the hook per GENERATION and disarms it before returning, so
# the honest steer-off read below is automatic. Doing this by hand is how the
# 'dilution trap' happens: arm once per scale, disarm on the first sample, and only
# 1 in (prompts*samples) generations is steered -- a real effect then reads as ~e/20,
# i.e. as a null. `STEER.check()` below raises if that ever recurs.
STEER = SteerController(model.model.layers)

def generate(prompt):
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=GEN_LEN, do_sample=True, temperature=0.8,
                         top_p=0.9, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids['input_ids'].shape[1]:], skip_special_tokens=True)

def fluency(text):
    """mean per-token log-likelihood of the text (higher = more fluent English)."""
    ids = tok(text, return_tensors="pt").to(model.device)['input_ids']
    if ids.size(1) < 2:          # empty/1-token text: mean() of nothing is nan, not 0
        return float("nan")      # (callers skip empty text; this is belt-and-braces)
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
n_empty = 0  # generations that came back empty (nothing to score; skipped)
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
    keys = all_keys(Xa, y, net)      # every steering key, unit-norm; `mean` first
    print_cos_table(keys)            # §4 evidence: where each key POINTS (free)
    for name in DIRECTIONS:
        delta = keys[name].to(model.dtype)
        tones, flu = {}, {}
        for scale in SCALES:
            ts, fs = [], []
            for p in PROMPTS:
                for _ in range(SAMPLES):
                    # ARM PER GENERATION (SteerController arms, generates, disarms),
                    # so the honest steer-off read below cannot break the treatment.
                    t = STEER.generate(p, delta, scale, LAYER, generate)
                    if not t.strip():           # empty generation -> nothing to score
                        n_empty += 1
                        continue
                    ts.append(tone(net, capture_vec(t)))   # hook already OFF: honest
                    fs.append(fluency(t))
            tones[scale] = statistics.mean(ts) if ts else float("nan")
            flu[scale] = statistics.mean(fs) if fs else float("nan")
        slope = tones[SCALES[-1]] - tones[SCALES[0]]
        line = "  ".join(f"{s:>3}={tones[s]:+.2f}" for s in SCALES)
        print(f"  [{name:>16}] tone  {line}   slope={slope:+.2f}   "
              f"fluency={statistics.mean(list(flu.values())):.2f}")
        CELLS.append((AXES[axis], f"{n_pos}/{n_neg}", f"{acc:.2f}", name, f"{slope:+.2f}",
                      f"{statistics.mean(list(flu.values())):.2f}"))

# ---- STEERING AUDIT: did the treatment actually get applied? ------------------
# This is the check whose absence let a 20x-diluted null look "clean".
n_steered, n_gen = STEER.check()
print(f"\n[steering audit] {n_steered}/{n_gen} generations ran with the hook LIVE "
      f"({STEER.n_should_steer} should have been; the rest are the scale=0 baseline).")
if n_empty:
    print(f"[steering audit] {n_empty} empty generation(s) skipped (no text to score).")

print("\n" + "=" * 78)
print("SUMMARY  (slope = tone at max scale - tone at 0; 'random' is the SHAM control)")
print(f"{'axis':<10} {'n +/-':<9} {'probe':>5}  {'direction':>16}  {'slope':>6}  {'fluency':>7}")
for r in CELLS:
    print(f"{r[0]:<10} {r[1]:<9} {r[2]:>5}  {r[3]:>16}  {r[4]:>6}  {r[5]:>7}")
print("A real steer = slope clearly larger than the SHAM row with fluency near baseline.")
print("If the sham moves tone as much as the real directions, the probe is measuring")
print("noise, not tone -- and the honest conclusion is 'not measurable at this scale'.")

