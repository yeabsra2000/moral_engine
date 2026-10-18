"""
MESE-CR — M4 (part 1): EASY-set sentence generator
===================================================
Builds a large, TRUSTWORTHY "easy" set for all 4 axes using curated filler-slot
templates. Polarity is guaranteed BY CONSTRUCTION (sentiment/moral keywords drive
the label), so these need no human review. Path = easy.csv
Format per record: sentence | V | A | C | F | easy
  +1 = pos/arousing/care/fair   -1 = neg/calm/harm/cheat   0 = neutral for that axis
"""
import itertools, csv, os

AXES = ["Valence", "Arousal", "Care/Harm", "Fairness"]

# ---- filler slots (subjects / verbal cores), guaranteed-polar on the target axis ----
#
# FRAME-NEUTRALITY (2026-10-18). The first version ended every Care/Harm sentence
# with a fixed locative that tracked polarity ("... on the roadside." for +1,
# "... in front of everyone." for -1) and every Fairness sentence with a fixed
# object ("... the whole team." / "... the rest of the team."). Those frames were
# SHARED across the template groups, so grouped CV -- which holds out the verb --
# could not remove them, and a model-free rule on the frame alone decoded
# Care/Harm at ~0.96 and Fairness at ~0.87. The frames are now gone (Care/Harm)
# or polarity-neutral and embedded (Fairness), so the only label-carrying signal
# is the verb, which grouped CV does hold out. `shared_context_features` in
# probe_cv.py is the control that would catch a regression.
SUBS = ["I", "My friend", "My brother", "My sister", "Our neighbor", "The student",
        "My teammate", "The volunteer", "My cousin", "The nurse"]

VAL_POS = ["felt absolutely ecstatic about", "was overjoyed by", "was thrilled about",
           "felt pure joy at", "was delighted with", "celebrated", "felt wonderful after"]
VAL_NEG = ["felt absolutely devastated by", "was heartbroken about", "was furious about",
           "was crushed by", "felt miserable after", "was in despair over", "regretted"]

ARO_POS = ["felt my heart race wildly", "was so excited I could not sleep",
           "felt a rush of intense energy", "felt wound up with restlessness",
           "felt a sudden surge of thrill"]
ARO_NEG = ["felt totally calm", "was deeply restful", "felt completely soothed",
           "felt serene the whole time", "was drowsy the whole afternoon"]

CARE_POS = ["gently comforted", "calmly reassured", "warmly welcomed",
            "carefully bandaged", "patiently fed"]
HARM_NEG = ["roughly shoved", "coldly ignored", "harshly scolded",
            "recklessly pushed", "openly mocked"]

FAIR_POS = ["split the money exactly equally among the whole team",
            "let everyone take a fair turn",
            "returned the lost wallet to its owner",
            "ended the contest and shared the credit with the runner-up",
            "gave each person the same portion"]
FAIR_NEG = ["quietly took the far larger share for themselves",
            "cut ahead of the line",
            "kept the prize and gave none to the others",
            "changed the rules to favor one side",
            "withheld the promised reward from the helpers"]

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
            rows.append(_rec(f"{subj} {ph} during the event.", [0, 1, 0, 0], f"arousal|{ph}", with_group))
        for ph in ARO_NEG:
            rows.append(_rec(f"{subj} {ph} during the event.", [0, -1, 0, 0], f"arousal|{ph}", with_group))
    # --- Care/Harm (C = +-1). NB: the sentence ENDS at the object -- no shared
    #     locative. A fixed suffix that tracked polarity (the old "on the roadside"
    #     / "in front of everyone") is cross-group surface and survives grouped CV.
    for subj in SUBS:
        for i, ph in enumerate(CARE_POS):
            for o in OBJ_ANIM[:4]:
                rows.append(_rec(f"{subj} {ph} {o}.", [0, 0, 1, 0],
                                 f"care|{ph}|{o}", with_group))
        for i, ph in enumerate(HARM_NEG):
            for o in OBJ_ANIM[:4]:
                rows.append(_rec(f"{subj} {ph} {o}.", [0, 0, -1, 0],
                                 f"harm|{ph}|{o}", with_group))
    # --- Fairness (F = +-1). The old polarity-locked object ("the whole team." /
    #     "the rest of the team.") is gone; each verb now carries its own neutral
    #     object so no cross-group frame separates the poles.
    for subj in SUBS:
        for ph in FAIR_POS:
            rows.append(_rec(f"{subj} {ph}.", [0, 0, 0, 1], f"fair|{ph}", with_group))
        for ph in FAIR_NEG:
            rows.append(_rec(f"{subj} {ph}.", [0, 0, 0, -1], f"unfair|{ph}", with_group))
    return rows

def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["sentence", "V", "A", "C", "F", "kind", "group"])
        w.writerows(rows)

if __name__ == "__main__":
    rows = build_easy(with_group=True)
    os.makedirs("data", exist_ok=True)
    write_csv("data/easy.csv", rows)
    from collections import Counter
    print(f"WROTE data/easy.csv: {len(rows)} sentences")
    for ax in range(4):
        c = Counter(r[1 + ax] for r in rows)
        print(f"  {AXES[ax]}: {dict(c)}")
    print(f"  template groups: {len({r[6] for r in rows})} (each holds ~{len(rows)//len({r[6] for r in rows})} sentences)")
