"""
dry_model.py — a tiny stand-in for {tokenizer, model} so step6/step7 can be
wiring-checked on a laptop CPU with NO model download.
=============================================================================
`MESE_DRY=1` makes those scripts use `build_dry()` instead of Transformers.

It is a real (if minimal) causal LM -- char-level vocab, 28 decoder blocks, an
LM head -- and it honours the exact hook API the scripts use
(`register_forward_pre_hook(hook(module, args))` and
`register_forward_hook(hook(module, inp, out))`, with the decoder block
returning a TUPLE like HF does under `use_cache`). That matters: it means
steering and reading flow end-to-end exactly as with a real model -- the steer
perturbs a block's output, which shifts the sampled continuation, which changes
the activation the probe then reads. A stub that returned a constant would test
the plumbing but not this coupling.

The NUMBERS ARE MEANINGLESS. The "language" is not English and the "activations"
are not language features; nothing here speaks to any real model or dataset.
It validates plumbing only, and it is deliberately NOT imported by the
experiment scripts unless MESE_DRY is set, so it can never touch a real run.
"""
import torch

CHAR_LO, CHAR_HI = 32, 127
VOCAB = 1 + (CHAR_HI - CHAR_LO)          # 96 tokens; id 0 is <eos>
D_MODEL = 64                             # deliberately tiny: CPU-cheap
N_LAYERS = 28                            # >= 26 so the step7 sweep layers are valid


class _Batch(dict):
    """Minimal stand-in for a HF BatchEncoding: dict access + attribute access + .to()."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:              # so tok(x).input_ids works like HF
            raise AttributeError(name) from exc

    def to(self, device):                    # already on CPU; nothing to move
        return self


class _Handle:
    """Removable hook handle, matching `h = layer.register_..._hook(fn); h.remove()`."""

    def __init__(self, store, fn):
        self.store, self.fn = store, fn

    def remove(self):
        if self.fn in self.store:
            self.store.remove(self.fn)


class _Layer:
    """One decoder block: honours both hook registrations the scripts rely on."""

    def __init__(self, d, gen):
        self.W = torch.randn(d, d, generator=gen) / d ** 0.5
        self._pre, self._post = [], []

    def register_forward_pre_hook(self, fn):
        self._pre.append(fn)
        return _Handle(self._pre, fn)

    def register_forward_hook(self, fn):
        self._post.append(fn)
        return _Handle(self._post, fn)

    def __call__(self, h, *rest):
        args = (h,) + rest
        for fn in list(self._pre):           # same signature as HF: fn(module, args)
            out = fn(self, args)
            if out is not None:
                args = tuple(out)
        h = args[0]
        h = h + torch.tanh(h @ self.W)       # bounded, so 28 blocks cannot diverge
        out = (h, None)                      # TUPLE, like HF with use_cache=True
        for fn in list(self._post):          # fn(module, inputs, output)
            r = fn(self, args, out)
            if r is not None:
                out = r
        return out


class _Tokenizer:
    """Char-level encode/decode with the HF call surface the scripts use."""

    def __init__(self):
        self.eos_token_id = 0

    def __call__(self, text, return_tensors="pt"):
        ids = [1 + (ord(c) - CHAR_LO) % (VOCAB - 1) for c in text] or [1]
        return _Batch(input_ids=torch.tensor([ids], dtype=torch.long))

    def decode(self, ids, skip_special_tokens=True):
        vals = ids.tolist() if torch.is_tensor(ids) else list(ids)
        chars = []
        for i in vals:
            if i == self.eos_token_id and skip_special_tokens:
                continue
            chars.append(chr(CHAR_LO + (i - 1) % (CHAR_HI - CHAR_LO)))
        return "".join(chars)


class DryModel:
    """Toy causal LM: char embeddings -> N_LAYERS blocks -> LM head."""

    def __init__(self, seed=0):
        gen = torch.Generator().manual_seed(seed)
        self.dtype = torch.float32
        self._device = torch.device("cpu")
        self.embed = torch.randn(VOCAB, D_MODEL, generator=gen) / D_MODEL ** 0.5
        self.layers = [_Layer(D_MODEL, gen) for _ in range(N_LAYERS)]
        self.head = torch.randn(D_MODEL, VOCAB, generator=gen) / D_MODEL ** 0.5
        self.model = self                    # so `model.model.layers` resolves, as in HF

    @property
    def device(self):
        return self._device

    def eval(self):
        return self

    def __call__(self, input_ids=None, **kw):
        h = self.embed[input_ids]            # (1, T) -> (1, T, d)
        for layer in self.layers:
            out = layer(h)
            h = out[0] if isinstance(out, tuple) else out
        return _Batch(logits=h @ self.head)  # (1, T, V)

    def generate(self, input_ids=None, max_new_tokens=16, do_sample=True,
                 temperature=1.0, top_p=1.0, pad_token_id=None, **kw):
        out = input_ids
        for _ in range(int(max_new_tokens)):
            with torch.no_grad():
                logits = self(input_ids=out).logits[0, -1]
            if do_sample:
                probs = torch.softmax(logits / max(float(temperature), 1e-6), dim=-1)
                nxt = int(torch.multinomial(probs, 1))
            else:
                nxt = int(logits.argmax())
            out = torch.cat([out, torch.tensor([[nxt]], dtype=torch.long)], dim=1)
            if pad_token_id is not None and nxt == pad_token_id:
                break
        return out


def build_dry(seed=0):
    """Return (tokenizer, model) with the call surface step6/step7 use."""
    torch.manual_seed(seed)
    return _Tokenizer(), DryModel(seed)