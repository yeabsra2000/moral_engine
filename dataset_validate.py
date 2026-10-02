"""
MESE-CR — M4 (part 3): HARD-set review flagger
==============================================
Loads dataset_draft.csv (LLM drafts) and flags candidates that need your
attention BEFORE you finalize labels, using the ONE shared definition of
"leaky" in leak_lexicon.py:
  LEAK     - the "surface-neutral" sentence still contains an emotion/moral
             keyword (or a cross-axis one), so it is really an EASY item.
  ABSTRACT - it is an abstract proposition ("a fair distribution is needed"),
             not a concrete event, so there is no situation to read tone from.
  FRAGMENT - not a full sentence / no concrete actor (too short, no determiner).
  DUP      - near-duplicate of another candidate (the drafts repeated a lot).
  PROBE    - optional: probe predicts the OPPOSITE of the proposed label (only
             when run on GPU with the model). Strong signal to re-check.

Runs offline using only the CSV; adds --probe on Colab with model.
Output: dataset_review.csv (all candidates + flags) for you to review & relabel.
NOTE: since these drafts proved mostly unusable, the released dataset uses the
curated set in dataset_hard.py; this tool stays for auditing future drafts.
"""
import csv, sys, argparse
from leak_lexicon import AXES, reason_to_reject, find_leaks


def flags_for(axis_label, sentence):
    return reason_to_reject(sentence)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="run probe-disagreement check (needs GPU/model)")
    ap.add_argument("--draft", default="dataset_draft.csv")
    ap.add_argument("--out", default="dataset_review.csv")
    args = ap.parse_args()

    with open(args.draft, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    AXIS = AXES
    probe_nets, probe_X = (None, None)
    if args.probe:
        probe_nets, probe_X = load_probe([r["sentence"] for r in rows])  # bulk capture

    out, flagged = [], []
    seen = {}
    for i, r in enumerate(rows):
        s = r["sentence"]
        lids = [AXIS[j] for j in range(4) if r[AXIS[j][0]] != "0"]
        axis_label = lids[0] if lids else "Valence"
        reasons = flags_for(axis_label, s)
        akey = {"Valence": "V", "Arousal": "A", "Care/Harm": "C", "Fairness": "F"}[axis_label]
        pole = int(r["proposed"]); ax_idx = AXIS.index(axis_label)
        norm = " ".join(s.lower().split())
        if norm in seen:
            reasons.append("DUP-of-row-" + str(seen[norm]))
        seen.setdefault(norm, i)
        disagree = ""
        if probe_nets is not None and pole != 0:
            pred = predict_one(probe_nets, probe_X[i], ax_idx)   # bulk-captured row
            if pred != pole:
                disagree = "PROBE-DISAGREES"
        flag = list(reasons)
        if disagree: flag.append(disagree)
        out.append({**r, "axis": axis_label, "k": akey, "flag": " | ".join(flag)})
        if flag: flagged.append(len(out) - 1)

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)

    print(f"Reviewed {len(out)} candidates; wrote {args.out}")
    print(f"  flagged for review: {len(flagged)}")
    for i in flagged[:40]:
        r = out[i]; print(f"  [{r['axis']:9} {r['proposed']:>2}] {r['flag']}  | {r['sentence'][:80]}")
    print("\nRule of thumb: LEAK items are actually EASY (reassign their label to the")
    print("polarity the keyword implies, or drop them from the HARD set). ABSTRACT and")
    print("FRAGMENT items cannot be used at all — rewrite them as concrete events.")

# ---- optional GPU probe-disagreement path (linear probe trained on EASY set) ----
def load_probe(candidate_sentences):
    """Train quick linear probes on the easy set; bulk-capture candidate activations.
    Returns (nets list, X_candidates) both on CPU."""
    import torch as T
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from dataset_templates import build_easy
    DTYPE = T.bfloat16 if T.cuda.is_available() else T.float32
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct",
                                                 torch_dtype=DTYPE, device_map="auto")
    model.eval()
    LAYER = 14

    def capture(sentences):
        cap = []
        h = model.model.layers[LAYER].register_forward_pre_hook(
            lambda m, a: (cap.append(a[0].detach().float().mean(dim=1).squeeze(0).to("cpu")), a)[1]
            if a and a[0].ndim == 3 else a)
        for s in sentences:
            ids = tok(s, return_tensors="pt").to(model.device)
            with T.no_grad(): model(**ids)
        h.remove()
        return T.stack(cap)

    easy = build_easy()                     # list of [sentence, V,A,C,F, kind]
    X = capture([r[0] for r in easy]); X = X / X.norm(dim=1, keepdim=True).clamp_min(1e-8)
    LAB = T.tensor([r[1:5] for r in easy], dtype=T.float32)
    nets = []
    for ax in range(4):
        act = LAB[:, ax].ne(0)
        xs, ys = X[act], (LAB[act, ax] > 0).float()
        net = T.nn.Sequential(T.nn.Linear(X.size(1), 16), T.nn.SiLU(), T.nn.Linear(16, 1))
        opt = T.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(400):
            opt.zero_grad()
            loss = T.nn.functional.binary_cross_entropy_with_logits(net(xs).view(-1), ys)
            loss.backward(); opt.step()
        net.eval(); nets.append(net)

    Xc = capture(candidate_sentences); Xc = Xc / Xc.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return nets, Xc

def predict_one(nets, v, ax_idx):
    """classify one captured CPU activation vector with the given axis net."""
    import torch as T
    n = nets[ax_idx]
    p = T.sigmoid(n(v.unsqueeze(0))).item()
    return 1 if p > 0.5 else -1

if __name__ == "__main__":
    main()

