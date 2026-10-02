"""
MESE-CR — step7: steering SEARCH (layer sweep + last-token), to attack the step6 null
====================================================================================
step6 (`step6_steer_all.py`) established a clean, GATED null at layer 14 with mean
pooling: tone is readable (grouped probe 0.66-0.98) but no direction -- not even the
probe's own supervised gradient (probe-grad) -- moves tone beyond a random sham.
That null is an absence of effect under a NARROW method. step7 tests the two
remaining levers from paper §8b:

  (b) LAYER SWEEP  : steer at every layer in MESE_LAYERS, not just 14.
                     (the pressure may live at a different depth)
  (c) LAST-TOKEN   : steer/measure the final-token residual instead of the mean.
                     (mean pooling may wash out the tone signal -- the same reason
                      grouped LINEAR probes sit at chance)

For each (layer, pooling) cell and each axis we:
  - train the measuring probe AT THAT layer/pooling,
  - report its template-grouped held-out accuracy (gate >= MIN_PROBE_ACC),
  - steer with mean / probe-grad / sham(random) at scales {0, 1.5, 4},
  - measure tone with the SAME-layer probe (steer hook OFF) and report fluency.

Then a GRID summary across layers/poolings/axes, so the paper can state exactly how
far the search went and whether the null held.

Config (env):
  MESE_LAYERS=14,20,26     layers to sweep        (default 14,20,26)
  MESE_POOL=mean,last      poolings to test       (default mean,last)
  MESE_AXES=0,2            axis indices          (default all 4)
  MESE_SHAM=n              random sham directions per cell (default 1).
                           n=1  -> compare to a single draw (step6-style, noise-prone);
                           n>=20 -> build a null DISTRIBUTION and score each cell by a
                                    z-score vs it (use this before claiming any effect).
  MESE_QUICK=1             smoke test (tiny grid, ~3-5 min)
Run in Google Colab (GPU). Requires dataset.py in the same folder.

IMPORTANT (what step7 found): with 5 prompts x 4 samples = 20 generations/cell the
per-cell noise floor is ~ +/-0.2, i.e. larger than a +0.10 verdict threshold. A
24-cell grid with one sham draw therefore produces ~half "wins" by chance. Always
re-check flags with MESE_SHAM=20; `analyze_step7.py` computes the noise statistics.
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
AXES_ALL = ["Valence", "Arousal", "Care/Harm", "Fairness"]
SCALES = [0.0, 1.5, 4.0]
DIRECTIONS = ["mean", "probe-grad", "random"]   # search keeps only the best + sham
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

# ---- search space (env-controlled) ----
LAYERS = [int(x) for x in os.environ.get("MESE_LAYERS", "14,20,26").split(",") if x.strip()]
POOLS  = [p.strip() for p in os.environ.get("MESE_POOL", "mean,last").split(",") if p.strip()]
AXES   = [AXES_ALL[int(i)] for i in os.environ.get("MESE_AXES", "0,1,2,3").split(",") if i.strip()]
N_SHAM = int(os.environ.get("MESE_SHAM", "1"))   # 1 = single sham draw (step6-style);
                                                 # >=20 = sham DISTRIBUTION -> per-cell z-score

# ---- QUICK mode: MESE_QUICK=1 python step7_sweep.py  (smoke test first!) ----
QUICK = os.environ.get("MESE_QUICK", "") not in ("", "0", "false", "False")
if QUICK:
    PROMPTS = PROMPTS[:2]
    SAMPLES = 1
    SCALES = [0.0, 4.0]
    DIRECTIONS = ["mean", "random"]
    LAYERS = LAYERS[:1]
    POOLS = POOLS[:1]
    AXES = AXES[:1]

# ---- tee everything to a log file: a dropped Colab session must not lose the run ----
LOG = os.environ.get("MESE_LOG", "results_step7.txt")
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
_log.write(f"\n===== step7 run {datetime.datetime.now():%Y-%m-%d %H:%M:%S} "
           f"quick={QUICK} =====\n")
sys.stdout = _Tee(sys.__stdout__, _log)
print(f"[log] writing to {LOG}  |  QUICK={QUICK}")
print(f"layers={LAYERS}  pools={POOLS}  axes={AXES}")
print(f"(prompts={len(PROMPTS)}, samples={SAMPLES}, scales={SCALES}, directions={DIRECTIONS})")
print("Full sweep: ~45-90 min on a T4 (trim with MESE_LAYERS / MESE_POOL). QUICK: ~3-5 min.")


torch.manual_seed(0); random.seed(0)
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map="auto")
model.eval()
N_LAYERS = len(model.model.layers)
print(f"model={MODEL_ID}  depth={N_LAYERS}")

# ---- capture activations at ALL sweep layers in ONE forward pass ----
def capture_layers(text, layers):
    """Return {layer: hidden_states (1, T, d)} for the given layers, one pass."""
    caps = {}
    handles = []
    for L in layers:
        def mk(L):
            def h(mod, args):
                hs = args[0]
                if isinstance(hs, tuple): hs = hs[0]
                caps[L] = hs.detach().float()      # keep (1, T, d) on-device
                return args
            return h
        handles.append(model.model.layers[L].register_forward_pre_hook(mk(L)))
    ids = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad(): model(**ids)
    for h in handles: h.remove()
    return caps

def pool_vec(hs, pool):
    """(1,T,d) -> (d,) : mean over tokens, or the LAST token's residual."""
    if pool == "last":
        return hs[0, -1, :]
    return hs[0].mean(dim=0)

# ---- build the (layer, pool) -> (N, d) activation matrices once ----
LAYERS = sorted(set(L for L in LAYERS if 0 <= L < N_LAYERS))
print(f"\ncapturing activations at layers {LAYERS} ...")
acc = {(L, p): [] for L in LAYERS for p in POOLS}
for si, (sentence, _, _) in enumerate(DATA):
    caps = capture_layers(sentence, LAYERS)
    for L in LAYERS:
        for p in POOLS:
            acc[(L, p)].append(pool_vec(caps[L], p).cpu())
    if (si + 1) % 200 == 0:
        print(f"  {si+1}/{len(DATA)} sentences captured")

XMAT = {}
for key, rows in acc.items():
    M = torch.stack(rows).to(DEVICE)
    XMAT[key] = M / (M.norm(dim=1, keepdim=True) + 1e-8)   # L2-normalize per row
print(f"  built {len(XMAT)} activation matrices of shape "
      f"{next(iter(XMAT.values())).shape}")

def make_probe(Xa, y, hidden=32, iters=500):
    """Small MLP measuring probe on the active samples for this axis."""
    xs, ys = Xa, (y > 0).float().to(Xa.device)
    net = torch.nn.Sequential(torch.nn.Linear(Xa.size(1), hidden), torch.nn.SiLU(),
                              torch.nn.Linear(hidden, 1)).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(iters):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(xs).view(-1), ys)
        loss.backward(); opt.step()
    net.eval(); return net

# ---- is this axis MEASURABLE at this (layer, pool)? grouped held-out accuracy ----
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
    for te in group_folds(y, groups, k):
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

def unit(v):
    return v / (v.norm() + 1e-8)

# ---- CONTROLS for the READING claim (cheap, no generation) -------------------
# A high grouped held-out accuracy only speaks about TONE if two things hold:
#  (a) label-permutation: the identical CV on SHUFFLED labels collapses to ~0.50.
#      Anything above chance here means the split/training leaks, not that tone
#      is decodable.
#  (b) length-only: the identical CV on sequence LENGTH alone must NOT already
#      reach that accuracy. This one matters specifically for `last` pooling --
#      the last token's residual co-encodes WHERE the sentence ended, i.e. its
#      length -- so a near-perfect last-token probe could be a positional/length
#      confound rather than tone. NOTE: the one-hot(template) control used in
#      step5 CANNOT serve this purpose here, because template identity is held
#      out wholly by grouped CV and therefore sits at chance by construction for
#      every pooling; it is not a control for a grouped last-token number.
SEQLEN = torch.tensor([len(tok(s).input_ids) for s, _, _ in DATA], dtype=torch.float32)

def control_acc(Xa, y, groups, se_act, kind, seeds=(0, 1, 2)):
    """Probe accuracy under a leak/confound control, averaged over `seeds`.

    Averaging matters: a SINGLE CV pass on shuffled labels already wanders to
    ~0.64 on a 100-row set (see _smoke_controls.py), so a one-seed control has a
    noise floor of roughly +-0.10 and would flag real cells as leaking. Chance
    plus that noise floor is what the leak bar (0.60) is set against.
    `se_act` must be aligned with the rows of `Xa` (subset to active samples).
    """
    accs = []
    if kind == "length":
        seeds = seeds[:1]                      # length feature is fixed -> one pass
    for s in seeds:
        if kind == "length":                   # 1-feature probe: standardized length
            s_v = se_act.float()
            z = (s_v - s_v.mean()) / (s_v.std() + 1e-8)
            accs.append(probe_cv_acc(z.view(-1, 1).to(Xa.device), y, groups))
        else:                                  # labels shuffled, features untouched
            g = torch.Generator().manual_seed(s)
            perm = torch.randperm(y.numel(), generator=g)
            accs.append(probe_cv_acc(Xa, y[perm.to(y.device)], groups))
    return statistics.fmean(accs)

# ---- ways to estimate the steering direction (at a given layer/pool) ----
def dir_mean(Xa, y):
    return Xa[y > 0].mean(dim=0) - Xa[y < 0].mean(dim=0)

def dir_probe_grad(net, Xa, y):
    """Supervised, nonlinear key: the measuring probe's input gradient at the
    positive-pole mean (the steepest ascent of the READABLE structure)."""
    dev = next(net.parameters()).device
    pos = Xa.to(dev)[y > 0]
    x0 = pos.mean(dim=0).detach().clone().requires_grad_(True)
    logit = net(x0.unsqueeze(0)).squeeze(0)
    (logit.sum()).backward()
    return unit(x0.grad.detach())


# ---- steering hook, parametrized by layer ----
@dataclass
class Ctx: dirn=None; handle=None; layer=None
steer_ctx = Ctx()

def attach_steer(vec, scale, layer):
    """Add scale * vec to layer `layer`'s output. scale=0 removes the hook."""
    if steer_ctx.handle:
        steer_ctx.handle.remove(); steer_ctx.handle = None
    if scale:
        steer_ctx.dirn = vec
        steer_ctx.layer = layer
        s = scale
        def _h(mod, inp, out):
            # decoder layer returns (hidden_states, past_kv); weights may sit on cpu
            if isinstance(out, tuple):
                h, rest = out[0], out[1:]
                return ((h + s * steer_ctx.dirn.to(h.device)),) + rest
            return out + s * steer_ctx.dirn.to(out.device)
        steer_ctx.handle = model.model.layers[layer].register_forward_hook(_h)

def capture_text_vec(text, layer, pool):
    """Pooled activation of `text` at (layer, pool), steer OFF for an honest tone."""
    caps = capture_layers(text, [layer])
    v = pool_vec(caps[layer], pool).detach().float().cpu()
    return v / (v.norm() + 1e-8)

def tone(net, v):
    dev = next(net.parameters()).device
    with torch.no_grad():
        p = torch.sigmoid(net(v.to(dev).unsqueeze(0))).item()
    return 2.0 * p - 1.0

def generate(prompt):
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=GEN_LEN, do_sample=True, temperature=0.8,
                         top_p=0.9, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids['input_ids'].shape[1]:], skip_special_tokens=True)

def fluency(text):
    """Mean per-token log-likelihood (higher = more fluent English)."""
    ids = tok(text, return_tensors="pt").to(model.device)['input_ids']
    with torch.no_grad():
        logits = model(ids).logits
    logp = torch.nn.functional.log_softmax(logits[:, :-1], dim=-1)
    return logp.gather(-1, ids[:, 1:].unsqueeze(-1)).mean().item()

LAB = torch.tensor([lbl for _, lbl, _ in DATA], dtype=torch.float32)

print("\n" + "=" * 78)
print("step7 SEARCH: layer sweep x token-pooling x axis")
print("tone in -1..1 (measured with the steer hook OFF); fluency = mean per-token logp.")
print(f"Measuring-probe gate: template-grouped held-out accuracy must be >= {MIN_PROBE_ACC}.")
print("A 'real steer' = a real direction's slope clearly ABOVE the SHAM (random) slope.")
print("=" * 78)

CELLS = defaultdict(dict)   # (L, pool, axis) -> {direction: (slope, fluency)}
INFO  = {}                  # (L, pool, axis) -> (n_str, acc)
CONTROLS = {}               # (L, pool, axis) -> (label-perm acc, length-only acc)
for (L, p) in sorted(XMAT):
    Xfull = XMAT[(L, p)]
    print(f"\n################ LAYER {L}  |  POOL {p} ################")
    for ax_name in AXES:
        axis = AXES_ALL.index(ax_name)
        act = LAB[:, axis].ne(0)
        act_idx = torch.nonzero(act).view(-1).tolist()
        y = LAB[:, axis][act].to(DEVICE)
        Xa = Xfull[act]
        groups = ([GROUPS[i] for i in act_idx] if GROUPS else None)
        se_act = SEQLEN[act_idx]
        n_pos, n_neg = int((y > 0).sum()), int((y < 0).sum())
        acc = probe_cv_acc(Xa, y, groups)
        INFO[(L, p, ax_name)] = (f"{n_pos}/{n_neg}", acc)
        print(f"\n### L{L} {p} {ax_name}: {int(act.sum())} sentences "
              f"({n_pos} pos / {n_neg} neg), {len(set(groups)) if groups else int(act.sum())} groups")
        print(f"    measuring probe held-out accuracy = {acc:.2f}   (gate {MIN_PROBE_ACC})")
        # ---- reading CONTROLS: the accuracy must beat both or it means nothing ----
        a_perm = control_acc(Xa, y, groups, se_act, "perm")
        a_len = control_acc(Xa, y, groups, se_act, "length")
        CONTROLS[(L, p, ax_name)] = (a_perm, a_len)
        print(f"    CONTROLS: label-permutation = {a_perm:.2f} (want ~0.50, noise floor ~0.55)   "
              f"length-only = {a_len:.2f} (want << {acc:.2f})")
        if a_perm >= 0.65:
            print("    -> LEAK WARNING: shuffled labels still score high; this cell's")
            print("       accuracy cannot be interpreted as decodability. Do not cite it.")
        if a_len >= acc - 0.05:
            print("    -> CONFOUND WARNING: sentence LENGTH alone already matches this")
            print("       accuracy, so the reading may be a length/positional artifact")
            print("       (expected risk for `last` pooling). Do not cite it as tone.")
        if not (acc >= MIN_PROBE_ACC):
            print("    -> AXIS NOT MEASURABLE here: skipping (an 'effect' would be indistinguishable")
            print("       from probe noise, so reporting one would be dishonest).")
            CELLS[(L, p, ax_name)]["unmeasurable"] = ("-", "-")
            continue
        net = make_probe(Xa, y)
        base = dir_mean(Xa, y)
        real_dirs = {"mean": base, "probe-grad": dir_probe_grad(net, Xa, y)}
        # SHAM: n random directions. n=1 reproduces step6's single-draw control;
        # n>=20 gives a null DISTRIBUTION so a cell can be scored (z-score) rather
        # than compared to one noisy draw.
        sham_keys = ["random"] if N_SHAM == 1 else [f"random{i}" for i in range(N_SHAM)]
        sham_dirs = {k: torch.randn_like(base) for k in sham_keys}
        all_dirs = {k: unit(v) for k, v in {**real_dirs, **sham_dirs}.items()}
        print(f"    cos(probe-grad, mean) = "
              f"{torch.nn.functional.cosine_similarity(all_dirs['probe-grad'].view(1,-1), all_dirs['mean'].view(1,-1)).item():+.3f}")
        if N_SHAM > 1:
            print(f"    [sham distribution: {N_SHAM} random directions]")
        run_names = [d for d in DIRECTIONS if d != "random"] + sham_keys
        for name in run_names:
            delta = all_dirs[name].to(model.dtype)
            tones, flu = {}, {}
            for scale in SCALES:
                attach_steer(delta, scale, L)
                ts, fs = [], []
                for pr in PROMPTS:
                    for _ in range(SAMPLES):
                        t = generate(pr)
                        attach_steer(delta, 0.0, L)          # clear steer to measure honestly
                        ts.append(tone(net, capture_text_vec(t, L, p)))
                        fs.append(fluency(t))
                attach_steer(delta, 0.0, L)
                tones[scale], flu[scale] = statistics.mean(ts), statistics.mean(fs)
            slope = tones[SCALES[-1]] - tones[SCALES[0]]
            fl = statistics.mean(list(flu.values()))
            line = "  ".join(f"{s:>3}={tones[s]:+.2f}" for s in SCALES)
            print(f"  [{name:>10}] tone  {line}   slope={slope:+.2f}   fluency={fl:.2f}")
            CELLS[(L, p, ax_name)][name] = (slope, fl)

print("\n" + "=" * 78)
print("GRID SUMMARY   (slope = tone@max - tone@0; 'random' is the SHAM control)")
print(f"{'layer':>5} {'pool':>5} {'axis':<10} {'probe':>5} "
      f"{'mean':>6} {'probe-grad':>10} {'SHAM':>6}   verdict")
if N_SHAM == 1:
    print("!" * 78)
    print("!! WARNING: single sham DRAW (MESE_SHAM=1). A one-draw control has a noise")
    print("!! floor ~ +/-0.2 at 20 generations/cell, so a fixed +0.10 threshold flags")
    print("!! cells by chance (measured: 11/24 and 6/8 in two runs). The rows below are")
    print("!! NOT findings. Re-run with MESE_SHAM=20 for a z-scored verdict.")
    print("!" * 78)
wins = 0
for (L, p, ax_name) in sorted(CELLS):
    c = CELLS[(L, p, ax_name)]
    if "unmeasurable" in c:
        print(f"{L:>5} {p:>5} {ax_name:<10} {INFO[(L,p,ax_name)][1]:>5.2f} "
              f"{'-':>6} {'-':>10} {'-':>6}   unmeasurable")
        continue
    sham_keys = [k for k in c if k.startswith("random")]
    sham_slopes = [c[k][0] for k in sham_keys]
    sham_mu = statistics.fmean(sham_slopes)
    sham_sd = statistics.pstdev(sham_slopes) if len(sham_slopes) > 1 else float("nan")
    real = {k: v[0] for k, v in c.items() if not k.startswith("random")}
    best_k = max(real, key=lambda k: real[k]) if real else "-"
    best = real[best_k] if real else float("nan")
    if N_SHAM > 1 and sham_sd and sham_sd > 1e-9:
        z = (best - sham_mu) / sham_sd
        win = (z > 3.0)                       # 3-sigma vs the null DISTRIBUTION
        tag = f"z={z:+.1f} {'*** EFFECT' if win else 'null'}"
    else:
        win = best > sham_mu + 0.10           # single-draw control (step6-style)
        tag = ('flag? best=' + best_k + ' BUT UNCONFIRMED (single sham draw; '
               're-run MESE_SHAM=20)') if win else 'null'
    wins += int(win)
    pg = f"{real['probe-grad']:+.2f}" if "probe-grad" in real else "-"
    sham_txt = f"{sham_mu:>+6.2f}"
    if N_SHAM > 1:
        sham_txt += f"±{sham_sd:.2f}"
    print(f"{L:>5} {p:>5} {ax_name:<10} {INFO[(L,p,ax_name)][1]:>5.2f} "
          f"{real.get('mean', float('nan')):>+6.2f} {pg:>10} {sham_txt}   {tag}")

print(f"\nCells beyond the sham: {wins} / "
      f"{sum(1 for c in CELLS.values() if 'unmeasurable' not in c)} measurable cells."
      + (f"   (criterion: z > 3 over {N_SHAM} sham draws)" if N_SHAM > 1
         else "   (criterion: best real slope > single sham draw + 0.10)"))
if N_SHAM == 1:
    print("=> These are FLAGS, not findings: a single sham draw cannot distinguish an")
    print("   effect from generation noise. Re-run the same grid with MESE_SHAM=20 for a")
    print("   per-cell z-score against a null DISTRIBUTION (that run CAN be quoted).")
elif wins == 0:
    print("=> The step6 null HOLDS across every layer/pooling searched: readable but")
    print("   not steerable by residual-stream addition, even with a supervised key.")
else:
    print("=> At least one cell is > 3 sigma over the sham distribution: inspect those")
    print("   rows and re-run at the winning (layer, pooling) with more generations.")

# ---- recap: did the READING claim survive its own controls? ----
if CONTROLS:
    worst_perm = max(v[0] for v in CONTROLS.values())
    conf = [(k, v[1], INFO[k][1]) for k, v in CONTROLS.items()
            if v[1] >= INFO[k][1] - 0.05]
    print(f"\nREADING CONTROLS recap ({len(CONTROLS)} cells):")
    print(f"  worst label-permutation accuracy = {worst_perm:.2f}  "
          f"({'LEAK: do not cite these accuracies' if worst_perm >= 0.65 else 'no leakage: CV is clean'})")
    if conf:
        print(f"  {len(conf)} cell(s) where LENGTH alone matches the probe accuracy:")
        for (L, p, ax), la, acc in sorted(conf):
            print(f"    L{L} {p} {ax}: length-only {la:.2f} vs probe {acc:.2f}")
        print("  -> for those cells the 'readable' result may be a length/positional")
        print("     artifact, NOT tone. Qualify the pooling-comparison claim accordingly.")
    else:
        print("  no cell is matched by length alone -> the reading result is not a")
        print("     length artifact, so the last-token > mean pooling finding stands.")


