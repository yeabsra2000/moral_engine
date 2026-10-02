"""
MESE-CR Pilot — Step 1: Proof of life on Google Colab (free GPU)
===============================================================
What this does:
  1. Detects GPU (Colab free T4) vs CPU.
  2. Loads a NON-GATED 1B model (no HuggingFace login needed).
  3. Runs a 20-token smoke-test generation (baseline).
  4. Installs a clean "Observer" hook that captures hidden-state vectors
     at your target layers, so the NEXT step can train a probe on them.

Bugs fixed vs your original idea.md script:
  - Probes/steerer are NOT randomly initialized and labelled as functional
    (we leave steering OFF here; we first capture activations to train on).
  - register_forward_pre_hook: uses args[0] (the hidden tensor), not the tuple.
  - No float16-on-CPU (we pick a dtype that works on the device automatically).
  - Non-gated model so no account/token needed to start.

To run in Colab: File > Upload > choose this .py, or paste contents into a cell.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# ------------------------------------------------------------------------
# 0. Environment & device detection
# ------------------------------------------------------------------------
print("CUDA available:", torch.cuda.is_available())
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
print("Using dtype:", DTYPE)

# Non-gated 1B model (no login). Swap model id here if you prefer another.
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"   # ~1.5B, open, no gating
TARGET_LAYERS = [12, 14, 16]              # mid-to-late transformer layers

# ------------------------------------------------------------------------
# 1. Load model + tokenizer
# ------------------------------------------------------------------------
print(f"\nLoading {MODEL_ID} ...")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    device_map="auto",
)
model.eval()
print("Model loaded.")

# ------------------------------------------------------------------------
# 2. Observer: clean hook to CAPTURE intermediate hidden states
#    (the training material for the future probe). Steering is off here.
# ------------------------------------------------------------------------
hidden_states_cache = {}   # layer_index -> list of (batch, seq, hidden) tensors

def make_capture_hook(layer_idx):
    def hook_fn(module, args):
        # args is a tuple; the hidden states are the FIRST element (args[0])
        hs = args[0]
        if isinstance(hs, tuple):
            hs = hs[0]
        hidden_states_cache.setdefault(layer_idx, []).append(hs.detach().float().cpu())
        return args            # return unchanged -> pure observer, no steering
    return hook_fn

handles = []
for idx in TARGET_LAYERS:
    layer = model.model.layers[idx]
    h = layer.register_forward_pre_hook(make_capture_hook(idx))
    handles.append(h)
print(f"Observer hooks attached to layers {TARGET_LAYERS}")

# ------------------------------------------------------------------------
# 3. Smoke-test generation (baseline)
# ------------------------------------------------------------------------
def generate(prompt: str, max_new_tokens: int = 20) -> str:
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0], skip_special_tokens=True)

prompt = "A student caught cheating on an exam should"
print("\n--- Baseline generation (no steering) ---")
print(generate(prompt, max_new_tokens=30))

# ------------------------------------------------------------------------
# 4. Demonstrate the observer captured vectors, and report shapes
#    We use a FIXED-length input so all layers give matching shapes
#    (this is what a probe training set will use).
# ------------------------------------------------------------------------
print("\n--- Observer capture check (fixed-length input) ---")
fixed_prompt = "Cheating on an exam is wrong because"
inputs = tok(fixed_prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    model(**inputs)   # one forward pass -> each hook logs exactly one tensor

CLEAN = [h for h in handles]
for idx in TARGET_LAYERS:
    layer_tensors = hidden_states_cache[idx]
    # Merge tokens across all captured batches (rows are tokens; hdim matches)
    cat = torch.cat([t.view(-1, t.size(-1)) for t in layer_tensors], dim=0)
    print(f"layer {idx}: {cat.shape} (tokens x {cat.size(-1)}-dim) "
          f"min={cat.min():.3f} max={cat.max():.3f} mean={cat.mean():.3f}")
print("Capture OK: real activation vectors collected at target layers.")

# ------------------------------------------------------------------------
# Cleanup: detach hooks so they don't slow future runs
# ------------------------------------------------------------------------
for h in handles:
    h.remove()
print("\nHooks detached. Pilot passed: model loads, generates, and we can capture activations.")
