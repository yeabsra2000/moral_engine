"""
MESE-CR — Step 2: Train the "Observer" probe (the heart of the idea)
====================================================================
Research question this answers:
   Can a small network read the emotional/moral tone of a sentence OUT OF
   the model's internal activations?

It also tests the paper's headline claim head-to-head:
   LINEAR probe (rigid flat hyperplane)  vs  NONLINEAR MLP probe (bending fabric)

We use a small hand-labeled dataset where every sentence is about ONE of the
4 axes (valence / arousal / care / fairness) with a known sign (+/-).
For each axis we measure the probe's accuracy only on sentences where that
axis is "active" (non-neutral). Chance = 50%. Well above 50% = the signal is
REAL and recoverable -> strong evidence for the paper.

Run in Google Colab (GPU). Paste the file or upload it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

print("CUDA:", torch.cuda.is_available())
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if DEVICE == "cuda" else torch.float32

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
TARGET_LAYERS = [12, 14, 16]

# ------------------------------------------------------------------------
# 1. Curated dataset: (sentence, labels[valence, arousal, care, fairness])
#    labels are +1 (high pole) / -1 (low pole) / 0 (not this sentence's axis)
# ------------------------------------------------------------------------
# Axis order: [Valence, Arousal, Care/Harm, Fairness/Cheating]
def S(txt, v=0, a=0, c=0, f=0):
    return (txt, [v, a, c, f])

DATA = [
    # ---- VALENCE (v) : -1 negative, +1 positive ----
    S("She finally received the scholarship she had worked all year for.", v=1),
    S("The unexpected news about his dad being healthy again lifted everyone.", v=1),
    S("A tiny flower bloomed through the crack in the pavement.", v=1),
    S("After months of practice the kid finally nailed the solo.", v=1),
    S("They gathered for a warm dinner and laughed the whole night.", v=1),
    S("His family celebrated when he came home safe from the war.", v=1),
    S("Her dog lost in the storm for weeks was never found.", v=-1),
    S("The diagnosis the doctors gave him was worse than anyone feared.", v=-1),
    S("He watched his childhood home burn to the ground.", v=-1),
    S("The letter said she had been rejected from the only program she wanted.", v=-1),
    S("They buried their friend on a gray morning with heavy rain.", v=-1),
    S("Her favorite pet had been sick and no medicine could save it.", v=-1),
    # ---- AROUSAL (a) : -1 calm/low energy, +1 intense/high energy ----
    S("Suddenly sirens blared and everyone dove for cover at once.", a=1),
    S("The crowd erupted screaming as the final goal went in.", a=1),
    S("A frantic race to the station began when the alarm hit midnight.", a=1),
    S("Emergency lights flashed and people shouted over the roar.", a=1),
    S("The library was silent except for the slow tick of one clock.", a=-1),
    S("He dozed off in the armchair while the rain tapped the window.", a=-1),
    S("They sat quietly on the porch watching the sun set slowly.", a=-1),
    S("A gentle breeze moved the curtain and nothing else stirred.", a=-1),
    # ---- CARE / HARM (c) : -1 harmful, +1 protective/nurturing ----
    S("The nurse stayed by the patient's bed all night to comfort them.", c=1),
    S("A stranger stopped to help the injured cyclist into shelter.", c=1),
    S("The teacher quietly gave extra time to the struggling student.", c=1),
    S("They adopted the old street dog and gave it a warm home.", c=1),
    S("He left the wounded animal on the road to suffer alone.", c=-1),
    S("The gang targeted the weakest kids and mocked their fears.", c=-1),
    S("She spread cruel rumors about her friend to ruin her standing.", c=-1),
    S("They trapped the bird and starved it for entertainment.", c=-1),
    # ---- FAIRNESS / CHEATING (f) : -1 unjust/deceptive, +1 fair/transparent ----
    S("Both teams received the exact same equipment, rules and referees.", f=1),
    S("The judge recused herself because she knew one of the parties.", f=1),
    S("The company published its full pricing for every customer alike.", f=1),
    S("They counted every ballot twice to make sure nothing was missed.", f=1),
    S("The official secretly changed the results to favor his own friend.", f=-1),
    S("She copied her neighbor's answers and claimed the work as hers.", f=-1),
    S("The boss promoted his relative over clearly better qualified staff.", f=-1),
    S("They rigged the auction so no outsider could actually win.", f=-1),
]

AXES = ["Valence", "Arousal", "Care/Harm", "Fairness"]
print(f"Dataset: {len(DATA)} labeled sentences across 4 axes\n")

# ------------------------------------------------------------------------
# 3. Capture mean-pooled hidden states per sentence at each target layer
# ------------------------------------------------------------------------
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map="auto")
model.eval()

# per-layer list of sentence-mean vectors
layer_means = {idx: [] for idx in TARGET_LAYERS}

def make_capture_hook(idx):
    def hook_fn(module, args):
        hs = args[0]
        if isinstance(hs, tuple):
            hs = hs[0]
        mean = hs.detach().float().mean(dim=1).squeeze(0)  # (hidden,) , drop batch dim
        layer_means[idx].append(mean.to("cpu"))
        return args
    return hook_fn

handles = [model.model.layers[idx].register_forward_pre_hook(make_capture_hook(idx)) for idx in TARGET_LAYERS]

for sentence, _ in DATA:
    ids = tok(sentence, return_tensors="pt").to(model.device)
    with torch.no_grad():
        model(**ids)

for h in handles:
    h.remove()

# Build feature matrix and label matrix
LABELS = torch.tensor([lbl for _, lbl in DATA], dtype=torch.float32)   # (n, 4), values in {-1,0,1}



# ------------------------------------------------------------------------
# 4. Train LINEAR probe vs NONLINEAR (MLP / "bending fabric") probe
# ------------------------------------------------------------------------
torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

def make_features(layer_idx):
    X = torch.stack(layer_means[layer_idx])          # (n, 1536)
    X = X / (X.norm(dim=1, keepdim=True) + 1e-8)     # normalize rows
    return X.to(device)

def train_probe(X, y, hidden_dim=0, epochs=400):
    """hidden_dim=0 -> linear probe; else an MLP (bending fabric)."""
    X = X.reshape(X.size(0), -1)          # defensive: flatten any extra dims
    y = y.to(X.device)                    # keep labels on the same device as features
    n_in = X.size(1)
    if hidden_dim:
        net = nn.Sequential(nn.Linear(n_in, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
    else:
        net = nn.Sequential(nn.Linear(n_in, 1))
    net = net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = torch.nn.functional.binary_cross_entropy_with_logits
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(net(X).view(-1), y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = (torch.sigmoid(net(X).view(-1)) > 0.5).float()
        return (pred.eq(y > 0.5).float().mean().item())

print("=" * 78)
print(f"{'LAYER':<6} {'AXIS':<10} {'linear(flat)':<15} {'nonlinear(bending)':<20} {'n_active'}")
print("=" * 78)
summary = {}
for layer_idx in TARGET_LAYERS:
    X = make_features(layer_idx)
    for ax in range(4):
        active = LABELS[:, ax].ne(0)
        if active.sum() < 8:
            print(f"{layer_idx:<6} {AXES[ax]:<10} {'--':<15} {'--':<20} {active.sum().item():<8}")
            continue
        y = (LABELS[active, ax] > 0).float()
        Xa = X[active]
        acc_lin = train_probe(Xa, y, hidden_dim=0)
        acc_mlp = train_probe(Xa, y, hidden_dim=64)
        summary.setdefault(ax, []).append((acc_lin, acc_mlp))
        print(f"{layer_idx:<6} {AXES[ax]:<10} {acc_lin:<15.3f} {acc_mlp:<20.3f} {active.sum().item():<8}")

print("=" * 78)
print("Per-axis mean accuracy across layers (chance = 50%):")
for ax in range(4):
    lin = sum(p[0] for p in summary[ax]) / len(summary[ax])
    mlp = sum(p[1] for p in summary[ax]) / len(summary[ax])
    verdict = "SIGNAL FOUND" if max(lin, mlp) > 0.65 else "ambiguous (need more data)"
    print(f"  {AXES[ax]:<10} linear={lin*100:.1f}%  nonlinear={mlp*100:.1f}%  -> {verdict}")
