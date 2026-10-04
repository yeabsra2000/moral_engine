"""
step8_dose.py -- DOSE-FINDING: was the Step 6 null real, or just under-scaled?
=================================================================================
WHY THIS FILE EXISTS
-------------------
The corrected Step 6 run (2026-10-03) reported slopes indistinguishable from
sampling noise: max |z| = 2.6 over 28 cells, where a pure-noise run peaks at
~2.4. But that run cannot distinguish two very different worlds:

  (a) steering genuinely does not move tone in this model  -> a REAL null
  (b) the perturbation was never strong enough to matter    -> a MIS-SCALED run

Those are the same class of mistake as the dilution bug we just fixed: a broken
experiment that reads as a clean null, which every measurement control in the
world is unable to detect because the problem is the TREATMENT, not the
measurement.

The tell that points at (b): fluency was ~-2.5 at EVERY scale including the
strongest. If adding `scale * delta` four times over leaves the model
completely intact, the most likely reading is that the dose was sub-threshold,
not that the direction was causally inert.

WHAT THIS RUNS
--------------
ONE axis per run (default Care/Harm -- see AXIS below), TWO directions (`mean`
and the `random` sham), a WIDE scale ladder, and per-scale fluency. Fluency is
the readout that tells us where the model breaks. The decision rule is at the
bottom of the output.

Everything is configurable so the same script can be re-run as a confirmation
without editing code:

    python step8_dose.py                           # default axis = Care/Harm
    MESE_AXIS=Arousal MESE_SHAM=15 python step8_dose.py
    MESE_SCALES=0,10,50,100,200 python step8_dose.py
    MESE_DRY=1 python step8_dose.py     # CPU wiring check, meaningless numbers

The DEFAULT AXIS IS Care/Harm. The steering verdict must be read off the most
reliable instrument, and Care/Harm is the only axis whose template-grouped
held-out accuracy (~0.98) is comfortably clear of the 0.60 gate; Valence sits on
the boundary (~0.6-0.8 depending on probe protocol), so a run there can abort or
pass on instrument noise rather than on any steering effect.
"""
import os, sys, statistics, datetime, random, time
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Same two files are mandatory; a helpful error beats a bare ImportError on Colab.
_REQUIRED = ("dataset.py", "steering_keys.py")
_MISSING = [f for f in _REQUIRED
            if not any(os.path.exists(os.path.join(p, f))
                       for p in (_HERE, os.getcwd()))]
if _MISSING:
    raise SystemExit(f"\nMISSING FILE(S): {_MISSING}\n  Upload the repo files next to this script.\n")

DRY = os.environ.get("MESE_DRY", "") not in ("", "0", "false", "False")
if DRY:
    AutoModelForCausalLM = AutoTokenizer = None
else:
    from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from dataset import DATA, GROUPS
except ImportError:
    from dataset import DATA
    GROUPS = None
from steering_keys import (all_keys, unit, SteerController, stable_seed,
                           sham_vector)

# ---- configuration (all overridable; see module docstring) --------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
MODEL_ID = os.environ.get("MESE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
LAYER = int(os.environ.get("MESE_LAYER", "14"))
# DEFAULT IS Care/Harm, NOT Valence. Valence is the WEAKEST instrument on this
# dataset: its template-grouped held-out accuracy sits right on the 0.60 gate,
# so the run can abort (or squeak through) on instrument quality rather than on
# the steering question. Care/Harm is the strongest (grouped ~0.98, stable at
# layers 12/14/16) and has the most data (436 rows / 58 template groups), so a
# verdict there is decided by the treatment, not by the measuring device.
AXIS = os.environ.get("MESE_AXIS", "Care/Harm")
# Wide ladder on purpose: the point is to find where fluency BREAKS. A ladder that
# stops at 4.0 cannot distinguish "no effect" from "no dose".
SCALES = [float(s) for s in os.environ.get(
    "MESE_SCALES", "0.0,4.0,10.0,25.0,50.0").split(",")]
DIRECTIONS = ["mean", "random"]        # `random` is the SHAM control, always
PROMPTS = [
    "The long hike ended in heavy rain, but I still",
    "Our project did not win the award, but the team",
    "I missed the bus this morning, but I found",
    "When the letter finally arrived, she opened it and",
    "The village meeting ended late, and on the way home",
]
SAMPLES = int(os.environ.get("MESE_SAMPLES", "6"))
GEN_LEN = 18
MIN_PROBE_ACC = 0.60
# GATE PROTOCOL. The gate below must certify the SAME probe that then measures
# tone, or it certifies a different instrument than the experiment uses.
# History: this gate once ran hidden=32/reps=3 while `make_probe` (the measuring
# device) used hidden=64, and reps=3 leaves the mean within its own noise band --
# so a run could abort (or pass) on a probe that was never the instrument.
#   hidden=64 -> matches make_probe and step5_robust.py's reported MLP.
#   reps=10   -> matches step5_robust.py, so this number is directly comparable
#                to the reading accuracies quoted in the paper.
# Both are overridable, but there is rarely a reason to.
GATE_HIDDEN = int(os.environ.get("MESE_GATE_HIDDEN", "64"))
GATE_REPS = int(os.environ.get("MESE_GATE_REPS", "10"))
# Fluency drop from the scale-0 baseline that we call "the model is broken".
BREAK_FL = 0.20

LOG = os.environ.get("MESE_LOG", "results_step8_dose_dry.txt" if DRY
                     else "results_step8_dose.txt")
class _Tee:
    def __init__(self, *s): self.s = s
    def write(self, t):
        for st in self.s:
            try: st.write(t)
            except Exception: pass
        return len(t)
    def flush(self):
        for st in self.s:
            try: st.flush()
            except Exception: pass
_log = open(LOG, "a", encoding="utf-8")
_log.write(f"\n===== step8 dose run {datetime.datetime.now():%Y-%m-%d %H:%M:%S} "
           f"axis={AXIS} dry={DRY} =====\n")
sys.stdout = _Tee(sys.__stdout__, _log)
print(f"[log] writing to {LOG}  |  axis={AXIS} scales={SCALES} "
      f"directions={DIRECTIONS} samples={SAMPLES} dry={DRY}")

torch.manual_seed(0); random.seed(0)
if DRY:
    print("[DRY RUN] model NOT loaded; dry_model.py stub. Numbers are MEANINGLESS.")
    from dry_model import build_dry
    tok, model = build_dry()
else:
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE,
                                                 device_map="auto")
model.eval()

# Depth guard, same as step6: a different MESE_MODEL may not have LAYER layers, and
# an IndexError there would be far less obvious than an explicit message.
N_LAYERS = len(model.model.layers)
if not (0 <= LAYER < N_LAYERS):
    sys.exit(f"LAYER={LAYER} but {MODEL_ID} has {N_LAYERS} layers "
             f"(valid 0..{N_LAYERS - 1}); pass MESE_LAYER=<int>.")
print(f"model={MODEL_ID}  depth={N_LAYERS}  steering layer={LAYER}  dtype={DTYPE}")


def make_probe(Xa, y, epochs=400):
    # `.to(DEVICE)` is REQUIRED, not cosmetic: Xa lives on the model device, so a
    # CPU-resident probe raises "Expected all tensors to be on the same device" the
    # moment it meets Xa. step6_steer_all.py's probe_cv_acc does the same thing.
    net = torch.nn.Sequential(torch.nn.Linear(Xa.size(1), 64), torch.nn.SiLU(),
                              torch.nn.Linear(64, 1)).to(Xa.device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    lossf = torch.nn.BCEWithLogitsLoss()
    yb = (y > 0).float().view(-1, 1)
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(net(Xa), yb)
        loss.backward(); opt.step()
    return net


def tone(net, v):
    with torch.no_grad():
        return float(2.0 * torch.sigmoid(net(v.view(1, -1))) - 1.0)


def capture_vec(text):
    """Unit-norm pooled hidden state at LAYER for `text`.

    `.float()` is REQUIRED: the model runs in bfloat16, but the measuring probe is
    float32. Feeding a bfloat16 vector to a float32 Linear raises "mat1 and mat2 must
    have the same dtype". step6 casts in exactly the same place.
    """
    cap = []
    h = model.model.layers[LAYER].register_forward_hook(
        lambda m, i, o: cap.append(o[0] if isinstance(o, tuple) else o))
    ids = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad(): model(**ids)
    h.remove()
    v = cap[0].float().mean(dim=1).squeeze(0)
    return v / (v.norm() + 1e-8)


STEER = SteerController(model.model.layers)


def generate(prompt):
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=GEN_LEN, do_sample=True,
                         temperature=0.8, top_p=0.9, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids['input_ids'].shape[1]:], skip_special_tokens=True)


def fluency(text):
    ids = tok(text, return_tensors="pt").to(model.device)['input_ids']
    if ids.size(1) < 2:
        return float("nan")
    with torch.no_grad():
        logits = model(ids).logits
    logp = torch.nn.functional.log_softmax(logits[:, :-1], dim=-1)
    return logp.gather(-1, ids[:, 1:].unsqueeze(-1)).mean().item()


def seed_cell(direction):
    """Seed from (direction) ONLY -- deliberately NOT from `scale`.

    Common random numbers: every scale of one direction draws from the SAME stream, so
    the slope tone(max) - tone(0) is a PAIRED difference and prompt difficulty cancels.

    BUG FIX (2026-10-03). Two real defects lived in the previous version of this line,
    both of which made it look protective while doing nothing:
      1. it added `int(round(scale * 1000))` to the seed, so every scale got an
         INDEPENDENT stream -- the exact unpaired comparison it claimed to fix;
      2. it seeded from `hash(str)`, which Python 3 randomizes PER PROCESS, so even
         the per-direction seed was not reproducible between runs.
    Now: deterministic via stable_seed, and identical across the scale ladder.
    """
    s = stable_seed(AXIS, direction)
    torch.manual_seed(s); random.seed(s)


# ---- capture activations + labels -------------------------------------------
def embed_all():
    acts, labs = [], []
    for row in DATA:
        text, y = row[0], row[1]
        cap = []
        h = model.model.layers[LAYER].register_forward_hook(
            lambda m, i, o: cap.append(o[0] if isinstance(o, tuple) else o))
        ids = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad(): model(**ids)
        h.remove()
        # `.float()` here for the same reason as capture_vec: bfloat16 activations
        # cannot meet the float32 probe. Also L2-normalised per row, matching step6,
        # so the steering keys are computed from the SAME features step6 measured.
        #
        # CAPTURE POINT (do not "align" this casually): this is a FORWARD hook, so
        # it reads the residual stream LEAVING layer LAYER -- the same tap step6/step7
        # steer through. step5_robust.py uses a FORWARD_PRE hook, i.e. the stream
        # ENTERING the layer, so its "layer 14" is one hop shallower than ours. That
        # is a deliberate difference, not a bug: the gate must certify the stream the
        # steering actually writes to. Treat a step5-vs-step8 accuracy gap partly as
        # this one-hop offset before concluding the instrument changed quality.
        acts.append(cap[0].float().mean(dim=1).squeeze(0))
        labs.append(list(y) if isinstance(y, (list, tuple)) else [y])
    X = torch.stack(acts)
    X = X / (X.norm(dim=1, keepdim=True) + 1e-8)
    return X, torch.tensor(labs, dtype=torch.float32)


print("=" * 78)
print(f"DOSE-FINDING on {AXIS} @ layer {LAYER}. scales={SCALES}")
print("Per-scale tone AND fluency. Fluency is the signal for 'too strong'.")
print("=" * 78)
X, LAB = embed_all()
# Self-check: the feature dtype and the probe dtype MUST agree. Two real bugs in
# this script's history were dtype/device mismatches that a CPU-only dry run cannot
# reproduce (bf16 activations vs a float32 probe; CPU probe vs CUDA features). So
# assert the contract explicitly, before anything downstream can fail obscurely.
assert X.dtype == torch.float32, f"features must be float32, got {X.dtype}"
assert X.device.type == ("cuda" if torch.cuda.is_available() else "cpu"), \
    f"features landed on {X.device}, not the model device"
print(f"[dtype check] features {X.dtype} on {X.device}; probe will match")
AXES_ALL = ["Valence", "Arousal", "Care/Harm", "Fairness"]
ai = AXES_ALL.index(AXIS)
act = LAB[:, ai].ne(0)
y = LAB[:, ai][act].to(DEVICE)
Xa = X[act]
n_pos, n_neg = int((y > 0).sum()), int((y < 0).sum())
print(f"{AXIS}: {int(act.sum())} sentences ({n_pos} pos / {n_neg} neg)")
net = make_probe(Xa, y)
with torch.no_grad():
    pv = torch.sigmoid(net(Xa)).view(-1).cpu().numpy()
acc = float(((pv > 0.5).astype(float)
             == (y.view(-1) > 0).float().cpu().numpy()).mean())
# IN-SAMPLE accuracy is optimistic by construction: `net` was just fit on Xa. The
# probe is the MEASURING INSTRUMENT here -- it is what converts text into the tone
# number -- so an overfit instrument inflates every tone reading, including the sham
# ones. Report the honest template-grouped held-out accuracy alongside it, because
# step6 established that this gap is large on this dataset (Valence: 0.72 grouped
# vs 0.82 random-split, and the in-sample number is 1.00).
_cv, _groups = None, None
try:
    if GROUPS:
        _groups = [GROUPS[i] for i in torch.nonzero(act).view(-1).tolist()]
except Exception as _e:
    print(f"[warn] GROUPS unavailable ({_e}) -> held-out accuracy not computed.")


def grouping_is_informative(grp, yv):
    """Does the group key actually support a grouped split on THIS axis?

    Guards the two ways a grouped split degenerates into a number that LOOKS like
    held-out accuracy but is not:
      1. TOO FEW groups -- fewer than k, so most folds come out empty and
         "held-out accuracy" is really a 1-2 fold average (a local fold
         reimplementation once produced 3-of-5 empty folds in exactly this way,
         which is why the real folds now come from probe_cv.make_folds).
      2. ALL groups mirrored -- if EVERY group id spans both polarities, a group
         can neither go wholly into one fold nor leave each fold single-class, so
         a class-stratified grouped split is impossible.

    SCOPE -- what this guard is NOT. The literal-id 'easy' collapse is a failure
    mode this guard is built to CATCH, not a property of the shipped data. Read
    it as a hypothetical, never as a measurement (an earlier revision of this
    investigation did exactly that and wrongly reported the dataset as broken).

    Measured against dataset.py as of 2026-10: GROUPS holds 146 distinct template
    ids and ZERO literal 'easy' entries; per axis the counts are Valence 32,
    Arousal 28, Care/Harm 58, Fairness 28 -- all >= 5; and a no-signal control run
    through this exact grouping returns ~0.50 with no empty folds. Grouping is
    therefore sound on every axis, and this function returns ok=True throughout.
    It stays as a tripwire so a future dataset edit cannot silently break the gate.

    Returns (ok, reason, n_groups).
    """
    uniq = set(grp)
    if len(uniq) < 5:
        return False, f"only {len(uniq)} distinct group(s) for 5 folds", len(uniq)
    lab = {}
    for gg, yy in zip(grp, yv.tolist()):
        lab.setdefault(gg, set()).add(yy)
    mixed = sum(1 for v in lab.values() if len(v) > 1)
    if mixed == len(lab):
        return False, (f"all {len(lab)} group ids span BOTH polarities (mirrored "
                       f"templates) -> grouped folds cannot be class-stratified"), len(lab)
    return True, f"{len(uniq)} groups, {mixed} mirrored", len(uniq)


if _groups and not DRY:
    _ok, _why, _ng = grouping_is_informative(_groups, y.cpu())
    print(f"[grouping] {_why}")
    if _ok:
        # Canonical CV from probe_cv.py -- an importable library with no side
        # effects, so importing it is safe (unlike step6_steer_all.py, which
        # executes the whole experiment on import). Do NOT reimplement the folds
        # here: a local copy already silently produced 3-of-5 empty folds.
        # GATE_HIDDEN/GATE_REPS match `make_probe` and step5_robust.py, so this
        # number is the SAME instrument that measures tone below (see config block).
        import probe_cv as _pc
        _m, _se2 = _pc.cv_report(Xa, y.cpu(), hidden=GATE_HIDDEN, reps=GATE_REPS,
                                 groups=_groups)
        _cv = _m
        print(f"probe HELD-OUT (template-grouped) accuracy = {_m:.2f} "
              f"+/- {_se2:.2f} (2*SE)  "
              f"<-- honest instrument quality; gate {MIN_PROBE_ACC}")
        print(f"       (gate protocol: hidden={GATE_HIDDEN}, reps={GATE_REPS}, "
              f"same as the measuring probe)")
        print("       (in-sample below is optimistic: `net` was fit on exactly this data.)")
        _gate = _cv
    else:
        _cv = None
        print("[grouping] grouped CV is NOT meaningful here -> in-sample only.")
        print(f"           {MIN_PROBE_ACC}-style held-out numbers cannot be trusted for")
        print("           this axis. step8 CANNOT honestly certify the instrument,")
        print("           so any tone shift it reports is exploratory only. Fix the")
        print("           group key (give each mirrored template a per-polarity id)")
        print("           before treating a dose effect as confirmable.")
        _gate = acc
elif DRY:
    _gate = acc
    print("[DRY RUN] skipping the grouped held-out gate: the stub has no template")
    print("          structure, so it cannot be honestly evaluated. In-sample only.")
else:
    _gate = acc
    print("[warn] no template groups -> cannot compute honest held-out accuracy.")
    print("        Gate uses in-sample, so treat any verdict as PRELIMINARY.")
print(f"probe in-sample accuracy = {acc:.2f}")
if _cv is not None:
    # When the grouped gate ran above, this value was already printed there WITH its
    # 2*SE and its protocol; do not print a second, differently-formatted copy (it
    # reads as two different numbers in a saved log). This is the one-line summary
    # the decision block below actually gates on.
    print(f"[gate] instrument = template-grouped held-out {_cv:.2f}  "
          f"(gate {MIN_PROBE_ACC})")
    _gate = _cv
else:
    _gate = acc
if not (_gate >= MIN_PROBE_ACC):
    raise SystemExit("Axis not measurable under the honest (grouped held-out) "
                     "protocol -> no steering claim possible.")

keys = all_keys(Xa, y, net)
_c = float(torch.nn.functional.cosine_similarity(keys['mean'], keys['random'], dim=0))
print(f"cos(mean, random) = {_c:.3f}  (the sham must point somewhere unrelated)")

n_empty = 0
RESULTS = {}     # direction -> {scale: (tone, fluency, [per-sample tones])}
for name in DIRECTIONS:
    delta = keys[name].to(model.dtype)
    RESULTS[name] = {}
    print(f"\n--- direction: {name} " + "-" * 40)
    print(f"{'scale':>7} {'tone':>8} {'fluency':>9} {'d_flu':>8}  status")
    for scale in SCALES:
        ts, fs = [], []
        for p in PROMPTS:
            for _ in range(SAMPLES):
                seed_cell(name)             # same stream for every scale -> paired
                t = STEER.generate(p, delta, scale, LAYER, generate)
                if not t.strip():
                    n_empty += 1
                    continue
                ts.append(tone(net, capture_vec(t)))
                fs.append(fluency(t))
        tone_m = statistics.mean(ts) if ts else float("nan")
        flu_m = statistics.mean(fs) if fs else float("nan")
        RESULTS[name][scale] = (tone_m, flu_m, ts)
        base_f = RESULTS[name].get(0.0, (float('nan'), float('nan'), None))[1]
        d_flu = flu_m - base_f if base_f == base_f else float("nan")
        status = ""
        if d_flu == d_flu and abs(d_flu) > BREAK_FL:
            status = "MODEL DEGRADING (too strong)"
        print(f"{scale:>7g} {tone_m:>+8.3f} {flu_m:>9.3f} {d_flu:>8.3f}  {status}")

n_steered, n_gen = STEER.check()
print(f"\n[steering audit] {n_steered}/{n_gen} generations had the hook LIVE "
      f"({STEER.n_should_steer} should have been).")
if n_empty:
    print(f"[steering audit] {n_empty} empty generation(s) skipped.")

h_ref, doses = STEER.report_dose()
if h_ref:
    print(f"\n[dose] median ||h|| at layer {LAYER} = {h_ref:.2f} (pre-addition)")
    for s, r in doses.items():
        print(f"[dose] scale {s:>6g}  -> {r * 100:6.2f}% of ||h||")

# ---- sham DISTRIBUTION, not one sham draw ------------------------------------
# WHY THIS EXISTS. One post-hoc sham draw is not a control, it is a lottery. In the
# first dose run the single sham moved tone by -0.958 while the real `mean` key moved
# +0.560, which looks like "the sham beats the real direction" but actually only
# says one draw from a wide null distribution happened to be large. A real claim
# needs the whole null: several independent sham directions, then compare the real
# slope against THAT DISTRIBUTION.
N_SHAM = int(os.environ.get("MESE_SHAM", "6"))
if N_SHAM > 1:
    print(f"\n=== sham distribution: {N_SHAM} independent random directions ===")
    shams = []
    for i in range(N_SHAM):
        v = sham_vector(keys["mean"].shape, stable_seed(AXIS, "sham", i),
                        keys["mean"].device)
        shams.append(unit(v))
    # each sham gets the SAME treatment as the real key: paired slope across scales.
    # Stored as (tone, fluency, [per-sample tones]) -- the SAME 3-tuple shape as
    # RESULTS, so paired_slope can consume a sham and a real key identically.
    SHAM_SLOPES = {}
    for i, v in enumerate(shams):
        delta = v.to(model.dtype)
        per_scale = {}
        for scale in SCALES:
            ts, fs = [], []
            for p in PROMPTS:
                for _ in range(SAMPLES):
                    seed_cell(f"sham{i}")     # its own stream, shared across scales
                    t = STEER.generate(p, delta, scale, LAYER, generate)
                    if not t.strip():
                        n_empty += 1
                        continue
                    ts.append(tone(net, capture_vec(t)))
                    fs.append(fluency(t))
            per_scale[scale] = (statistics.mean(ts) if ts else float("nan"),
                                statistics.mean(fs) if fs else float("nan"), ts)
        SHAM_SLOPES[i] = per_scale
        print(f"  sham {i}: " + "  ".join(f"{s:g}={per_scale[s][0]:+.3f}" for s in SCALES))
    n_steered2, n_gen2 = STEER.check()

# ---- PAIRED slope + decision -------------------------------------------------
print("\n" + "=" * 78)
print("DECISION")
print("=" * 78)


def paired_slope(per_scale, hi=None, at_scale=None):
    """(slope, se) paired against scale 0, at a chosen scale.

    at_scale pins the exact scale; hi picks the largest fluency-safe one. Fixing the
    scale matters: comparing a real slope taken at one scale against sham slopes
    taken at another is not a like-for-like test, and it produced two different
    numbers for the same key in two blocks of this script before it was pinned.
    """
    lo_ts = per_scale.get(0.0, (float('nan'), float('nan'), None))[2]
    if not lo_ts:
        return float("nan"), float("nan"), None
    if at_scale is not None:
        cand = [at_scale] if at_scale in per_scale else []
    else:
        cand = [s for s in sorted(per_scale) if s != 0.0
                and (hi is None or s <= hi)]
    for s in reversed(cand):                      # prefer the largest safe scale
        ts = per_scale[s][2]
        if not ts:
            continue
        n = min(len(ts), len(lo_ts))
        d = [ts[i] - lo_ts[i] for i in range(n)]   # per-SAMPLE paired difference
        m = statistics.mean(d)
        se = (statistics.stdev(d) / (n ** 0.5)) if n > 1 else float("nan")
        return m, se, s
    return float("nan"), float("nan"), None


for name in DIRECTIONS:
    r = RESULTS[name]
    base = r.get(0.0)
    if base is None:
        continue
    # Restrict to the fluency-INTACT window: a slope measured where the model is
    # already degrading measures breakage, not steering.
    safe = [s for s in SCALES if s == 0.0 or abs(r[s][1] - base[1]) <= BREAK_FL]
    hi = max(safe) if safe else None
    mag, se, at = paired_slope(r, hi)
    tag = "SHAM" if name == "random" else "real"
    z = mag / se if se == se and se > 1e-9 else float("nan")
    # `hi`/`at` are None when EVERY nonzero scale already broke fluency (happens in
    # the dry stub, where even scale 4 degrades). Format defensively rather than
    # crashing on a `:g` of None -- and say WHY the window is empty.
    hi_s = "none" if hi is None else f"{hi:g}"
    at_s = "n/a" if at is None else f"{at:g}"
    win = "fluency-intact window <= " + hi_s
    if hi is None or at is None:
        win += "  [NO intact scale: all broke fluency -- nothing to interpret]"
    print(f"{tag:>5} {name:>16}: paired slope {mag:+.3f} +/- {se:.3f} "
          f"(z={z:+.2f}) at scale {at_s} [{win}]")

if N_SHAM > 1:
    # Pin BOTH sides to the same scale. The real key's scale is chosen from ITS OWN
    # fluency trace (the largest scale at which the model is still intact), then the
    # shams are re-read at exactly that scale. Without this the verdict compared a
    # real slope at scale S against sham slopes at scale max, which is not a test.
    _b = RESULTS["mean"].get(0.0)
    _safe = [s for s in SCALES
             if s == 0.0 or abs(RESULTS["mean"][s][1] - _b[1]) <= BREAK_FL]
    _hi = max(_safe) if _safe else None
    real_m, real_se, TARGET = paired_slope(RESULTS["mean"], _hi)
    slopes = []
    for i, per_scale in SHAM_SLOPES.items():
        m, se, at = paired_slope(per_scale, at_scale=TARGET)   # SAME scale
        if m == m:
            slopes.append(m)
    if TARGET is None:
        print("\n  No fluency-intact scale exists for the real key, so there is nothing")
        print("  to compare: every dose broke the model. Bisect the ladder DOWN first.")
        slopes = []
    if slopes:
        mu, sd = statistics.mean(slopes), statistics.stdev(slopes)
        print(f"\n  all slopes below are PAIRED, at scale {TARGET:g} "
              f"(fluency-intact for the real key)")
        print(f"  sham slope distribution: mean={mu:+.3f} sd={sd:.3f} "
              f"(n={len(slopes)})")
        print(f"  range: [{min(slopes):+.3f}, {max(slopes):+.3f}]")
        if sd > 1e-9:
            z = (real_m - mu) / sd
            # BOTH tests, because with n=6 the parametric z is optimistic. The
            # empirical test asks the question that actually matters -- how often does
            # a RANDOM direction beat the real key -- and needs no distributional
            # assumption. Report both; if they disagree, quote the weaker one.
            n_ge = sum(1 for v in slopes if v >= real_m)
            p_emp = (1 + n_ge) / (1 + len(slopes))     # add-one, never claims p=0
            print(f"\n  real `mean` key: {real_m:+.3f} +/- {real_se:.3f} "
                  f"->  z vs sham dist = {z:+.2f}")
            print(f"  empirical: {n_ge}/{len(slopes)} sham directions reach it "
                  f"(add-one p = {p_emp:.3f})")
            _sig = min(abs(z) >= 2.0, p_emp < 0.05)
            # A z-test only means something if the instrument can actually measure
            # the thing. With degenerate grouping the probe cannot be certified, so
            # a "significant" tone shift may be template memorization inside the
            # measuring probe. Never let a statistical verdict stand alone.
            if _cv is None:
                print("\n  VERDICT: NOT CONFIRMABLE.")
                print("           The shift is statistically suggestive, but the measuring")
                print("           instrument could NOT be validated (degenerate template")
                print("           grouping -> no honest held-out accuracy exists). Tone here")
                print("           may be template memorization rather than a real change.")
                print("           Fix dataset grouping before this means anything.")
                _sig = False
            elif not _sig:
                print("\n  VERDICT: NOT SEPARABLE from random directions.")
                print("           The two tests disagree or are both marginal, so this")
                print("           is a PRELIMINARY candidate effect at best. Treat as")
                print("           underpowered -- NOT as a steering result, and NOT as")
                print("           evidence that the model cannot be steered.")
                print("           Confirm with more shams (MESE_SHAM=15) and a second")
                print("           axis before writing it up.")
            else:
                print("\n  VERDICT: separable from the sham distribution on BOTH tests.")
                print("           Promising -- but n=6 shams is a thin null. Re-run to")
                print("           confirm, and check the dose-response is monotone")
                print("           before writing it up.")
    print(f"\n[steering audit incl. shams] {n_steered2}/{n_gen2} hook LIVE "
          f"({STEER.n_should_steer} should have been).")
print()
print("Read it this way:")
print("  * fluency BROKEN before tone moves  -> ladder too strong; bisect DOWN.")
print("  * tone moves with fluency INTACT    -> REAL steering effect; the old null was")
print("                                       mis-scaled, not a null.")
print("  * no tone shift, and the sham is the same size -> likely a null, but only")
print("    meaningful if the [dose] percentages reached a level where fluency was at")
print("    risk. A sub-5% dose cannot support a no-effect claim.")
