"""
MESE-CR — M4 (part 2): HARD-context candidate generator
========================================================
Loads the local Qwen-1.5B and DRAFTS "hard-context" sentences: surface-neutral
scenario descriptions whose tone/morality must be inferred from the situation
(no emotion/moral keywords). Proposed labels are assigned by construction because
each pole is generated with a directional scenario hint.
Writes dataset_draft.csv  ->  columns: sentence,V,A,C,F,proposed,scenario
Review these with dataset_validate.py; you finalize the hard labels yourself.
"""
import csv, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# scenario hint per axis/pole (tone set by situation, not by emotion words)
CELLS = {
    "Valence":   {"pos": "receiving unexpectedly good news about one's future",
                  "neg": "receiving unexpectedly bad news about one's future"},
    "Arousal":   {"pos": "a sudden alert that instantly energies someone",
                  "neg": "a calm, uneventful, drowsy moment"},
    "Care/Harm": {"pos": "gently helping or comforting another person who is hurt",
                  "neg": "leaving another hurt person completely without help"},
    "Fairness":  {"pos": "dividing resources fairly and equally among everyone",
                  "neg": "quietly taking an unfair extra share for oneself"},
}
TARGET_PER_CELL = 16          # sentences per pole
CHUNK = 8                     # ask per model call
MAX_TRIES = 6

INST = (
    "Write exactly {n} standalone English sentences. CRITICAL RULES:\n"
    "1. Use only neutral, factual words. Do NOT use emotion or moral words (no "
    "happy, sad, good, bad, kind, cruel, unfair, joy, fear, relieved, panic...).\n"
    "2. Each sentence must describe ONE concrete event with a named or definite "
    "actor and object (a specific person doing a specific thing in a specific "
    "place). Example of the required style: 'The landlord called to say the "
    "eviction notice had been withdrawn.'\n"
    "3. NEVER write abstract statements about resources, money, society, values, "
    "fairness or what people 'should' do. BAD: 'A fair distribution is needed.'\n"
    "4. The situation must be: {scenario}\n"
    "Output one sentence per line. Nothing else."
)

print("CUDA:", torch.cuda.is_available())
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct",
                                             torch_dtype=DTYPE, device_map="auto")
model.eval()

def gen_lines(scenario, n):
    prompt = INST.format(n=n, scenario=scenario.replace("\n", " ")) + "\n"
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=40 * n, do_sample=True,
                         temperature=0.9, top_p=0.95, pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0][ids['input_ids'].shape[1]:], skip_special_tokens=True)
    lines = []
    for ln in text.splitlines():
        s = ln.strip(" .-1234567890()[]\"").strip()
        if len(s) > 15 and s.lower() != s.upper() and not any(x in s.lower() for x in
            ["sentence", "output one", "here are", "###", "write", "please"]):
            if s not in lines:
                lines.append(s + ".")
    return lines

rows = []
axis_map = {"Valence": 0, "Arousal": 1, "Care/Harm": 2, "Fairness": 3}
from leak_lexicon import reason_to_reject      # same gate the validator uses
rejected = 0
for axis, poles in CELLS.items():
    for pole, scenario in poles.items():
        lbl = 1 if pole == "pos" else -1
        got, tries = [], 0
        while len(got) < TARGET_PER_CELL and tries < MAX_TRIES:
            got += gen_lines(scenario, min(CHUNK, TARGET_PER_CELL - len(got)))
            # keep only drafts that already pass the leakage/concreteness gate,
            # so what lands in the CSV is worth reading (abstract propositions and
            # keyword-leaked sentences are dropped here, not by you)
            keep, seen = [], set()
            for g in got:
                if g in seen:
                    continue
                seen.add(g)
                if reason_to_reject(g):
                    rejected += 1
                    continue
                keep.append(g)
            got = keep; tries += 1
        got = got[:TARGET_PER_CELL]
        for s in got:
            ax = axis_map[axis]
            vals = [0, 0, 0, 0]; vals[ax] = lbl
            rows.append([s] + vals + [lbl, scenario])
        print(f"{axis:>10} {pole:>3}: kept {len(got)}/{TARGET_PER_CELL}")

os.makedirs("data", exist_ok=True)
with open("data/dataset_draft.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["sentence", "V", "A", "C", "F", "proposed", "scenario"]); w.writerows(rows)
print(f"\nWROTE data/dataset_draft.csv: {len(rows)} hard-context candidates for review "
      f"({rejected} drafts rejected by the leakage/concreteness gate).")
