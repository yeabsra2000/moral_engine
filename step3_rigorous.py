"""
MESE-CR — Step 3: RIGOROUS probe evaluation (the publishable version)
=====================================================================
Fixes the two methodological problems from Step 2:
  1. PROPER generalization: 5-fold cross-validation -> accuracy is measured
     on sentences the probe NEVER trained on (held out).
  2. HARD test: separate sentences whose tone is only knowable from the
     SITUATION (no trigger keyword), vs easy keyword-revealed ones. We report
     held-out accuracy on the hard subset specifically — the impressive metric.

If hard held-out accuracy stays clearly above chance (50%), the claim is strong:
   "A small probe can infer context-driven moral/emotional tone straight from
    the model's internal activations."

Run in Google Colab (GPU).
"""
import torch
import torch.nn as nn
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataset import DATA

print("CUDA:", torch.cuda.is_available())
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if DEVICE == "cuda" else torch.float32
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
TARGET_LAYERS = [12, 14, 16]
AXES = ["Valence", "Arousal", "Care/Harm", "Fairness"]

# --------------------------------------------------------------------------
# 1. Capture mean-pooled hidden states per sentence at each target layer
# --------------------------------------------------------------------------
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map="auto")
model.eval()

layer_means = {idx: [] for idx in TARGET_LAYERS}
def make_capture_hook(idx):
    def hook_fn(module, args):
        hs = args[0]
        if isinstance(hs, tuple):
            hs = hs[0]
        layer_means[idx].append(hs.detach().float().mean(dim=1).squeeze(0).to("cpu"))
        return args
    return hook_fn
handles = [model.model.layers[idx].register_forward_pre_hook(make_capture_hook(idx)) for idx in TARGET_LAYERS]
for sentence, _, _ in DATA:
    ids = tok(sentence, return_tensors="pt").to(model.device)
    with torch.no_grad():
        model(**ids)
for h in handles:
    h.remove()

LABELS   = torch.tensor([lbl for _, lbl, _ in DATA], dtype=torch.float32)   # (n,4)
HARDMASK = torch.tensor([1.0 if d == "hard" else 0.0 for _, _, d in DATA])

# --------------------------------------------------------------------------
# 2. 5-fold cross-validated training + evaluation
# --------------------------------------------------------------------------
torch.manual_seed(0)
K = 5

def make_folds(y):
    """Stratified 5-fold indices."""
    idxs = list(range(y.numel()))
    classes = defaultdict(list)
    for i, v in enumerate(y.tolist()):
        classes[v].append(i)
    folds = [[] for _ in range(K)]
    for c, members in classes.items():
        for f in range(K):
            folds[f] += members[f::K]          # round-robin spread each class
    return folds

def build_probe(n_in, hidden):
    if hidden:
        return nn.Sequential(nn.Linear(n_in, hidden), nn.SiLU(), nn.Linear(hidden, 1))
    return nn.Sequential(nn.Linear(n_in, 1))

def train_and_predict(Xtr, ytr, Xte, hidden, epochs=800):
    net = build_probe(Xtr.size(1), hidden).to(Xtr.device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    ytr = (ytr > 0).float().to(Xtr.device)   # keep labels on the SAME device as features
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(Xtr).view(-1), ytr)
        loss.backward(); opt.step()
    with torch.no_grad():
        return (torch.sigmoid(net(Xte).view(-1)) > 0.5).int().cpu()

def evaluate(X, y, hard):
    """5-fold CV; returns (held-out acc, held-out acc on hard-only)."""
    folds = make_folds(y)
    all_idx, all_pred, all_hard = [], [], []
    for f in range(K):
        te = folds[f]; tr = [i for i in range(y.numel()) if i not in set(te)]
        xtr, xte = X[tr], X[te]
        ytr = y[tr]
        p = train_and_predict(xtr, ytr, xte, hidden=64)
        all_idx += te; all_pred += p.tolist(); all_hard += hard[te].tolist()
    order = sorted(range(len(all_idx)), key=lambda i: all_idx[i])
    pred = torch.tensor([all_pred[i] for i in order])
    tgt  = (y > 0).int()
    acc_all   = (pred == tgt).float().mean().item()
    hard_ok = (hard == 1) & (pred == tgt)
    hard_tot = (hard == 1).sum().item()
    acc_hard = hard_ok.sum().item() / hard_tot if hard_tot else float("nan")
    return acc_all, acc_hard

print("=" * 82)
print(f"{'AXIS':<10} {'n_active':<9} {'#hard':<6} {'linear CV':<12} {'MLP CV':<12} {'MLP hard-only'}")
print("=" * 82)
for layer_idx in TARGET_LAYERS:
    X = torch.stack(layer_means[layer_idx])
    X = X / (X.norm(dim=1, keepdim=True) + 1e-8)
    X = X.to(DEVICE)
    print(f"--- layer {layer_idx} (hidden={X.size(1)}d) ---")
    for ax in range(4):
        active = LABELS[:, ax].ne(0)
        if active.sum() < K * 2:
            print(f"  {AXES[ax]:<8} not enough active samples (n={active.sum().item()})")
            continue
        Xa = X[active]; ya = (LABELS[active, ax] > 0).float(); ha = HARDMASK[active]
        # linear probe CV (same fold splits as MLP)
        folds = make_folds(ya)
        alln, allp = [], []
        for f in range(K):
            te = folds[f]; tr = [i for i in range(ya.numel()) if i not in set(te)]
            p = train_and_predict(Xa[tr], ya[tr], Xa[te], hidden=0)
            alln += te; allp += p.tolist()
        order = sorted(range(len(alln)), key=lambda i: alln[i])
        pred_lin = torch.tensor([allp[i] for i in order])
        acc_lin = (pred_lin == (ya > 0).int()).float().mean().item()

        acc_mlp, acc_hard = evaluate(Xa, ya, ha)
        print(f"  {AXES[ax]:<8} {active.sum().item():<9} {ha.sum().item():<6} {acc_lin:<12.3f} {acc_mlp:<12.3f} {100*acc_hard:.1f}%")
print("=" * 82)
print("Chance = 50%.  'MLP hard-only' is the crucial number: strictly above ~65% = strong.")
