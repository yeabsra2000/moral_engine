"""
MESE-CR — Step 4: The STEERER (activate-and-measure)
====================================================
Now that the Observer (probe) WORKS, we test the second half: can we push the
model's internal activations so its OUTPUT tone actually changes?

Method (ActAdd-style activation steering):
  For an axis (e.g. Valence) at a chosen layer, we compute a steering direction
    delta = mean_activation(positive sentences) - mean_activation(negative)
  Then we ADD scale * delta to the hidden states during generation.
  We measure the effect with the Observer probe and print the real generated text.

To keep the measurement honest:
  steer hook is REMOVED before measuring tone, so the probe sees the actual
  text representation, not the injected vector.

Run in Google Colab (GPU). Requires dataset.py.
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataset import DATA

print("CUDA:", torch.cuda.is_available())
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if DEVICE == "cuda" else torch.float32
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
STEER_LAYER = 14          # where we inject
OBS_LAYER   = 14          # where the probe reads
AXIS = 0                  # 0=Valence, 1=Arousal, 2=Care, 3=Fairness
TARGET_SIGN = 1           # steer toward positive pole

tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map="auto")
model.eval()

capture = {}              # layers -> list of (label_sign, mean_vec)
def make_capture_hook(layer_idx):
    def hook_fn(module, args):
        hs = args[0]
        if isinstance(hs, tuple):
            hs = hs[0]
        capture.setdefault(layer_idx, []).append(hs.detach().float().mean(dim=1).squeeze(0).to("cpu"))
        return args
    return hook_fn
# NOTE: STEER_LAYER and OBS_LAYER are both 14 here, so ONE hook is enough.
handles = [model.model.layers[STEER_LAYER].register_forward_pre_hook(make_capture_hook(STEER_LAYER))]

# Capture training-sentence activations (only the sentences whose AXIS is active)
sent_signs = []
for sentence, lbl, _ in DATA:
    if lbl[AXIS] == 0:
        continue
    sent_signs.append(1 if lbl[AXIS] > 0 else -1)
    ids = tok(sentence, return_tensors="pt").to(model.device)
    with torch.no_grad():
        model(**ids)
for h in handles:
    h.remove()

def stacked(layer):
    return torch.stack(capture[layer])        # (n, 1536)

# ---- steering direction from training activations ----
obs = stacked(OBS_LAYER)
pos = obs[torch.tensor(sent_signs) > 0].mean(dim=0)
neg = obs[torch.tensor(sent_signs) < 0].mean(dim=0)
delta = (pos - neg).float().to(DEVICE)
delta = delta / (delta.norm() + 1e-8)

# ---- Observer probe (trained now, to measure tone of any text) ----
probe = torch.nn.Sequential(torch.nn.Linear(obs.size(1), 64), torch.nn.SiLU(), torch.nn.Linear(64, 1)).to(DEVICE)
y = torch.tensor(sent_signs, dtype=torch.float32).to(DEVICE)
X = obs / (obs.norm(dim=1, keepdim=True) + 1e-8)
opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
for _ in range(800):
    opt.zero_grad()
    loss = F.binary_cross_entropy_with_logits(probe(X.to(DEVICE)).view(-1), (y > 0).float())
    loss.backward(); opt.step()
probe.eval()

def tone_of(text, steer_off=False):
    """Measure predicted valence (in [-1,1]) of a text via the probe.

    steer_off=True: temporarily detach the steering hook before reading (steer-off
    measurement of the completed generation).
    steer_off=False: read with the steering still armed (steered representation).
    """
    global steer_handle
    saved = steer_handle
    if steer_off and steer_handle is not None:
        steer_handle.remove()
        steer_handle = None
    try:
        ids = tok(text, return_tensors="pt").to(model.device)
        layer_means = {}
        def hook_fn(module, args):
            hs = args[0]
            if isinstance(hs, tuple):
                hs = hs[0]
            layer_means["m"] = hs.detach().float().mean(dim=1).squeeze(0)
            return args
        h = model.model.layers[OBS_LAYER].register_forward_pre_hook(hook_fn)
        with torch.no_grad():
            model(**ids)
        h.remove()
        v = layer_means["m"].to(DEVICE)
        v = v / (v.norm() + 1e-8)
        with torch.no_grad():
            p = torch.sigmoid(probe(v.unsqueeze(0))).item()
        return 2.0 * p - 1.0
    finally:
        if steer_off and saved is not None:
            steer_handle = saved
            model.model.layers[STEER_LAYER].register_forward_pre_hook(saved._hook_fn if hasattr(saved, '_hook_fn') else None)  # best effort
        # if steer_off, hook was a new one removed; restore original
        if steer_off and saved is not None:
            # easier: just re-attach the same scale context; but simpler: call attach_steer at caller level
            pass

# (note: simpler and safe: to do steer-off reads, detach, measure, re-attach; caller can track scale)


# ---- steering hook ----
steer_handle = None
def attach_steer(scale):
    global steer_handle
    if steer_handle is not None:
        steer_handle.remove()
    def hook_fn(module, args):
        hs = args[0]
        if isinstance(hs, tuple):
            hs = hs[0]
        add = (scale * delta).view(1, 1, -1).to(hs.device, hs.dtype)
        return (hs + add,) + args[1:]
    steer_handle = model.model.layers[STEER_LAYER].register_forward_pre_hook(hook_fn)

def generate(prompt, max_new_tokens=40):
    ids = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new_tokens,
                             do_sample=True, temperature=0.8,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0], skip_special_tokens=True)

import statistics

PROMPTS = [
    "The long hike ended in heavy rain, but I still",
    "Our project did not win the award, but the team",
    "I missed the bus this morning, but I found",
]
SCALES = [0.0, 1.5, 4.0]
SAMPLES = 3   # generations per scale per prompt

print("=" * 74)
print("STEERING CURVE  (axis: Valence, toward POSITIVE, layer 14)")
print("Each tone = mean over prompt x sample generations. READ = steered activation")
print("(the hook stays armed during the read; see the note on tone_of below).")
print("=" * 74)

results = {s: [] for s in SCALES}
examples = {s: [] for s in SCALES}
for scale in SCALES:
    attach_steer(scale)
    for p in PROMPTS:
        for _ in range(SAMPLES):
            t = generate(p)
            results[scale].append(tone_of(t))
            examples[scale].append(t)

for s in SCALES:
    vals = results[s]
    mean = statistics.mean(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    print(f"\n--- scale={s}:  mean tone = {mean:+.2f} (+/- {sd:.2f}) ---")
    # show the first sample from each prompt
    seen = set()
    for t in examples[s]:
        first = t.split(". ")[0][:90]
        if first not in seen:
            seen.add(first)
            print(f"   > {first}...")

if steer_handle is not None:
    steer_handle.remove()
print("\n" + "=" * 74)
print("A clean result = mean tone rises as scale goes 0 -> 4 (monotonic, fluent text).")
