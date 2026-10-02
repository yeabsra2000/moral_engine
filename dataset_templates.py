"""
MESE-CR — M4 (part 1): EASY-set sentence generator
===================================================
Builds a large, TRUSTWORTHY "easy" set for all 4 axes using curated filler-slot
templates. Polarity is guaranteed BY CONSTRUCTION (sentiment/moral keywords drive
the label), so these need no human review. Path = easy.csv
Format per record: sentence | V | A | C | F | easy
  +1 = pos/arousing/care/fair   -1 = neg/calm/harm/cheat   0 = neutral for that axis
"""
import itertools, csv

AXES = ["Valence", "Arousal", "Care/Harm", "Fairness"]

# ---- filler slots (subjects / verbal cores), guaranteed-polar on the target axis ----
SUBS = ["I", "My friend", "My brother", "My sister", "Our neighbor", "The student",
        "My teammate", "The volunteer", "My cousin", "The nurse"]

VAL_POS = ["felt absolutely ecstatic about", "was overjoyed by", "was thrilled about",
           "felt pure joy at", "was delighted with", "celebrated", "felt wonderful after"]
VAL_NEG = ["felt absolutely devastated by", "was heartbroken about", "was furious about",
           "was crushed by", "felt miserable after", "was in despair over", "regretted"]

ARO_POS = ["adrenaline surged and I could not calm down", "felt my heart race wildly",
           "was so excited I could not sleep", "felt a rush of intense energy",
           "was wound up and bursting with energy"]
ARO_NEG = ["felt totally calm and relaxed", "was completely soothed and at peace",
           "felt slow and drowsy the whole day", "was deeply restful and quiet",
           "felt serene and unhurried"]

CARE_POS = ["gently comforted", "carefully tended to", "made a warm meal for",
            "wrapped a blanket around", "rushed to help"]
HARM_NEG = ["roughly shoved", "cruelly ignored the cries of", "kicked",
            "yelled harshly at", "left helplessly stranded"]

FAIR_POS = ["split the money exactly equally among", "let everyone take a fair turn",
            "returned the lost wallet to", "ended the contest and shared the credit with",
            "gave each person the same portion to"]
FAIR_NEG = ["quietly took the far larger share away from", "cut ahead of",
            "kept the prize to himself and gave none to", "changed the rules to favor",
            "withheld the promised reward from"]

OBJ_NEUTRAL = ["ordinary things", "the daily routine", "this week's events",
               "a quiet afternoon", "the usual tasks", "the small ceremony"]
OBJ_ANIM = ["the child", "the older lady", "the tired worker", "the injured animal",
            "the new student", "the neighbor", "the junior member", "the lost visitor"]

def _rec(sentence, vals, group, with_group=False):
    """One dataset row: [sentence, V, A, C, F, 'easy'] (+ group key if asked).

    group = the TEMPLATE the sentence came from. Sentences sharing a group are
    near-paraphrases of each other (only the subject slot differs), so grouped
    CV must keep a whole group on one side of the split -- otherwise random CV
    leaks near-duplicates across train/test and inflates probe accuracy.
    """
    row = [sentence] + vals + ["easy"]
    if with_group:
        row.append(group)
    return row

def build_easy(with_group=False):
    rows = []
    # --- Valence (V = +-1, others 0) ---
    for subj in SUBS:
        for ph in VAL_POS:
            rows.append(_rec(f"{subj} {ph} the results.", [1, 0, 0, 0], f"valence|{ph}", with_group))
        for ph in VAL_NEG:
            rows.append(_rec(f"{subj} {ph} the results.", [-1, 0, 0, 0], f"valence|{ph}", with_group))
    # --- Arousal (A = +-1, others 0) ---
    for subj in SUBS:
        for ph in ARO_POS:
            rows.append(_rec(f"{subj} said {ph} during the event.", [0, 1, 0, 0], f"arousal|{ph}", with_group))
        for ph in ARO_NEG:
            rows.append(_rec(f"{subj} said {ph} during the event.", [0, -1, 0, 0], f"arousal|{ph}", with_group))
    # --- Care/Harm (C = +-1) ---
    for subj in SUBS:
        for i, ph in enumerate(CARE_POS):
            for o in OBJ_ANIM[:4]:
                rows.append(_rec(f"{subj} {ph} {o} on the roadside.", [0, 0, 1, 0],
                                 f"care|{ph}|{o}", with_group))
        for i, ph in enumerate(HARM_NEG):
            for o in OBJ_ANIM[:4]:
                rows.append(_rec(f"{subj} {ph} {o} in front of everyone.", [0, 0, -1, 0],
                                 f"harm|{ph}|{o}", with_group))
    # --- Fairness (F = +-1) ---
    for subj in SUBS:
        for ph in FAIR_POS:
            rows.append(_rec(f"{subj} {ph} the whole team.", [0, 0, 0, 1], f"fair|{ph}", with_group))
        for ph in FAIR_NEG:
            rows.append(_rec(f"{subj} {ph} the rest of the team.", [0, 0, 0, -1], f"unfair|{ph}", with_group))
    return rows

def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["sentence", "V", "A", "C", "F", "kind", "group"])
        w.writerows(rows)

if __name__ == "__main__":
    rows = build_easy(with_group=True)
    write_csv("easy.csv", rows)
    from collections import Counter
    print(f"WROTE easy.csv: {len(rows)} sentences")
    for ax in range(4):
        c = Counter(r[1 + ax] for r in rows)
        print(f"  {AXES[ax]}: {dict(c)}")
    print(f"  template groups: {len({r[6] for r in rows})} (each holds ~{len(rows)//len({r[6] for r in rows})} sentences)")
