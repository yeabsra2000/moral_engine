"""
MESE-CR — M1: Robust statistics (repeated CV + template-grouped CV)
===================================================================
Upgrades Step 3 so the Observer results carry statistical weight:
  - REPEATED 5-fold CV (10 reps, fresh shuffles) -> mean held-out accuracy +- spread
  - PAIRED comparison of MLP ("bending fabric") vs linear ("rigid")
    -> how often does MLP beat linear, and the mean paired improvement.
  - TWO split protocols: RANDOM vs GROUPED (template/motif held out). The gap
    between them is the template-memorization inflation of the earlier numbers.

A result becomes publication-defensible when MLP's GROUPED mean is clearly above
chance (0.50) with a small standard error, and it consistently beats linear.

Run in Google Colab (GPU). Requires dataset.py AND probe_cv.py in the same folder.
"""
import os, sys, datetime

# ---- dependency bootstrap ---------------------------------------------------
# Inlined on purpose: a shared helper module would itself be one more file a Colab
# user can forget to upload, which is the exact failure this guards against.
# Without it a missing file surfaces as a bare "ModuleNotFoundError: No module
# named 'dataset'" (twice, from the fallback import below) that never says WHICH
# file is missing or where Python looked.
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:                              # pasted into a notebook cell
    _HERE = os.getcwd()
for _p in (_HERE, os.getcwd()):
    if _p not in sys.path:
        sys.path.insert(0, _p)
_REQUIRED = ["dataset.py", "probe_cv.py"]
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

import torch, statistics
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from dataset import DATA, GROUPS          # GROUPS = template/motif key per row
except ModuleNotFoundError:                    # dataset.py truly missing -> real error
    raise
except ImportError:                            # older dataset.py without GROUPS
    from dataset import DATA
    GROUPS = None
from probe_cv import cv_rep, template_only_features  # shared, unit-tested CV logic

print("CUDA:", torch.cuda.is_available())
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if DEVICE == "cuda" else torch.float32
MODEL_ID = os.environ.get("MESE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
AXES = ["Valence", "Arousal", "Care/Harm", "Fairness"]
KFOLD = 5

# ---- QUICK mode: MESE_QUICK=1 python step5_robust.py  (smoke test first!) ----
QUICK = os.environ.get("MESE_QUICK", "") not in ("", "0", "false", "False")
# MESE_DRY=1 runs the whole analysis pipeline WITHOUT loading the model (activations
# are seeded noise). Use it to wiring-check the script on a laptop CPU (~5 min in
# QUICK mode, measured 305 s) with no model download; its numbers are meaningless,
# and it logs to its own file so it cannot contaminate a real result log.
DRY = os.environ.get("MESE_DRY", "") not in ("", "0", "false", "False")
REPS = 3 if QUICK else 10
# layers to read; override with MESE_LAYERS for a model with a different depth
TARGET_LAYERS = ([int(x) for x in os.environ["MESE_LAYERS"].split(",") if x.strip()]
                 if os.environ.get("MESE_LAYERS") else ([14] if QUICK else [12, 14, 16]))

# ---- tee everything to a log file: a dropped Colab session must not lose the run ----
# A DRY run gets its OWN log by default, so a wiring check can never mix into (or be
# mistaken for) real model output. Override with MESE_LOG if you want it elsewhere.
LOG = os.environ.get("MESE_LOG",
                     "results/step5_dry.txt" if DRY else "results/step5.txt")
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
os.makedirs(os.path.dirname(LOG) or ".", exist_ok=True)
_log = open(LOG, "a", encoding="utf-8")
_log.write(f"\n===== step5 run {datetime.datetime.now():%Y-%m-%d %H:%M:%S} "
           f"quick={QUICK} dry={DRY} =====\n")
sys.stdout = _Tee(sys.__stdout__, _log)
print(f"[log] writing to {LOG}  |  QUICK={QUICK} DRY={DRY} "
      f"(REPS={REPS}, layers={TARGET_LAYERS})")
if DRY:
    print("*** DRY RUN: model not loaded, activations are noise. Results are MEANINGLESS.")
else:
    print("Full run: ~15-25 min on a T4. QUICK run: ~2-4 min.")

# ---- capture hidden states (mean-pooled per sentence) ----
# DRY (see the flag near QUICK above) substitutes seeded noise for activations.
SEQLEN = []                      # token count per sentence, for the length CONTROL
if DRY:
    print("[DRY RUN] model NOT loaded; activations = seeded Gaussian noise.")
    print("          Validates plumbing (probes/controls/logging) only.")
    print("          Every accuracy printed below is meaningless.\n")
    _g = torch.Generator().manual_seed(0)
    _dim = 64
    layer_means = {idx: [torch.randn(_dim, generator=_g) for _ in DATA]
                   for idx in TARGET_LAYERS}
    _gl = torch.Generator().manual_seed(1)
    SEQLEN = torch.randint(8, 30, (len(DATA),), generator=_gl).tolist()
else:
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map="auto")
    model.eval()
    layer_means = {idx: [] for idx in TARGET_LAYERS}
    def make_capture_hook(idx):
        def hook_fn(module, args):
            hs = args[0]
            if isinstance(hs, tuple): hs = hs[0]
            layer_means[idx].append(hs.detach().float().mean(dim=1).squeeze(0).to("cpu"))
            return args
        return hook_fn
    handles = [model.model.layers[i].register_forward_pre_hook(make_capture_hook(i)) for i in TARGET_LAYERS]
    for sentence, _, _ in DATA:
        ids = tok(sentence, return_tensors="pt").to(model.device)
        SEQLEN.append(ids.input_ids.size(1))
        with torch.no_grad(): model(**ids)
    for h in handles: h.remove()
LABELS = torch.tensor([lbl for _, lbl, _ in DATA], dtype=torch.float32)
_se = torch.tensor(SEQLEN, dtype=torch.float32)
LENFEAT = ((_se - _se.mean()) / (_se.std() + 1e-8)).view(-1, 1)   # 1-column control

HAS_GROUPS = GROUPS is not None
print("=" * 78)
print(f"Repeated {REPS}x{KFOLD}-fold CV. Chance = 0.50. Accuracy = mean +- 2*SE.")
print("  RANDOM : sentences split independently (near-duplicate templates can")
print("           land in both train and test -> optimistic)")
print("  GROUPED: a whole template/motif is held out (honest generalization)")
print("=" * 78)
for layer_idx in TARGET_LAYERS:
    X = torch.stack(layer_means[layer_idx]); X = X / (X.norm(dim=1, keepdim=True) + 1e-8); X = X.to(DEVICE)
    print(f"--- layer {layer_idx} ---")
    for ax in range(4):
        active = LABELS[:, ax].ne(0)
        if active.sum() < 20:
            print(f"  {AXES[ax]:<10}: too few active samples ({active.sum().item()})"); continue
        y = (LABELS[active, ax] > 0).float(); Xa = X[active]
        gs = ([GROUPS[i] for i in torch.nonzero(active).view(-1).tolist()]
              if HAS_GROUPS else None)
        got = {}
        for mode, groups in (("random", None), ("grouped", gs)):
            if mode == "grouped" and not HAS_GROUPS:
                continue
            lin = [cv_rep(Xa, y, 0, s, groups) for s in range(REPS)]
            mlp = [cv_rep(Xa, y, 64, s, groups) for s in range(REPS)]
            ml = statistics.fmean(mlp)
            se = statistics.stdev(mlp) / (REPS ** 0.5) if REPS > 1 else 0.0
            diff = [a - b for a, b in zip(mlp, lin)]
            wins = sum(1 for d in diff if d > 0) / len(diff)
            got[mode] = (ml, statistics.fmean(lin))
            print(f"  {AXES[ax]:<10} [{mode:>7}] n={int(active.sum()):>3}  "
                  f"MLP={ml:.2f} +/- {2*se:.2f}   linear={got[mode][1]:.2f}   "
                  f"MLP wins {100*wins:.0f}%")
        if len(got) == 2:
            print(f"  {AXES[ax]:<10} {'':>9}  template-memorization gap = "
                  f"{got['random'][0] - got['grouped'][0]:+.2f}")
        # CONTROL: identical CV on one-hot(template) features, which carry NO tone
        # information. Any accuracy here is pure template memorization, so this is
        # the baseline the random-split number must be compared against.
        if HAS_GROUPS:
            ctrl = statistics.fmean([cv_rep(template_only_features(gs), y, 64, s)
                                     for s in range(REPS)])
            print(f"  {AXES[ax]:<10} {'':>9}  template-identity CONTROL (random split) = "
                  f"{ctrl:.2f}  ->  real signal above control = {got['random'][0] - ctrl:+.2f}")
            # LENGTH control, under the HONEST grouped split: if sentence length alone
            # predicts this well, the reading may be a length/positional artifact and
            # not tone. Template identity cannot test this: it is held out WHOLLY by
            # grouped CV, so it sits at chance there by construction.
            # NB: hidden=64 (SAME capacity as the measured MLP), not hidden=0. A
            # linear 1-feature probe underfits badly -- 400 Adam steps at lr=1e-3
            # cannot always escape a wrong-sign weight init on a 1-D input, so with
            # hidden=0 this control scores ~0.6 on data where length FULLY determines
            # the label, i.e. it would vacuously "pass" and prove nothing.
            _lns = [cv_rep(LENFEAT[active].to(DEVICE), y, 64, s, gs)
                    for s in range(REPS)]
            ln = statistics.fmean(_lns)
            # Report the SPREAD as well as the mean: one mean cannot separate "length
            # really carries signal on this axis" from "one noisy fold". This control is
            # the one that decides whether a reading survives, so it must be falsifiable.
            ln_se = statistics.stdev(_lns) / (REPS ** 0.5) if REPS > 1 else 0.0
            # LEAK control: identical grouped CV, labels shuffled -> must sit at ~0.50.
            gp = torch.Generator().manual_seed(0)
            yp = y[torch.randperm(y.numel(), generator=gp)]
            _ps = [cv_rep(Xa, yp, 64, s, gs) for s in range(min(REPS, 3))]
            perm = statistics.fmean(_ps)
            perm_se = statistics.stdev(_ps) / (len(_ps) ** 0.5) if len(_ps) > 1 else 0.0
            print(f"  {AXES[ax]:<10} {'':>9}  length-only CONTROL (grouped) = "
                  f"{ln:.2f} +/- {2*ln_se:.2f}   "
                  f"label-permutation CONTROL (grouped) = {perm:.2f} +/- {2*perm_se:.2f}")
            # LEAK verdict: same 0.65 bar as step7_sweep.py, chosen in
            # test_probe_calibration.py to sit ABOVE the 3-seed noise floor (a single
            # seed alone wandered to 0.64 on a 100-row set and would have faked a leak).
            if perm >= 0.65:
                print(f"  {AXES[ax]:<10} {'':>9}  -> LEAK WARNING: shuffled labels still "
                      f"score {perm:.2f}; this reading cannot be interpreted as decodability.")
            if got['grouped'][0] > 0.55 and ln >= got['grouped'][0] - 0.05:
                print(f"  {AXES[ax]:<10} {'':>9}  -> CONFOUND WARNING: length alone matches the "
                      f"grouped MLP ({got['grouped'][0]:.2f}); do not cite this as tone.")
print("=" * 78)
print("Read the GROUPED row as the honest accuracy, and the CONTROL row as the")
print("accuracy that pure template identity already buys under a random split.")
print("Defensible = GROUPED mean clearly above 0.50 with a small spread AND the")
print("random-split number clearly above its own template-identity control.")
