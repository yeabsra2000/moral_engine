"""
MESE-CR — leakage lexicon and concreteness checks (single source of truth)
==========================================================================
A sentence only counts as HARD-context if its tone is NOT readable from a
surface keyword. This module holds:

  LEAK[axis]   : words that make that axis's polarity explicit
                 (= the sentence is really an EASY item, not a hard one)
  FLAT_LEAK    : tone words that leak ANY axis (checked for every sentence)
  ABSTRACT_RE  : shapes of abstract propositions ("a fair distribution is
                 needed") that carry no concrete event -> unusable
  CONCRETE_RE  : markers of a concrete situation (determiners / names)

Used by dataset_hard.py (audit of the curated set), dataset_validate.py
(flagging LLM drafts) and dataset_merge.py (final gate before writing
dataset.py), so all three agree on what "leaky" means.
"""
import re

# ---- per-axis leakage lexicons (word stems; substring match, lowercased) ----
LEAK = {
    "Valence": [
        "happy", "sad", "great", "good", "bad", "terrible", "wonderful", "awful",
        "joy", "fear", "delight", "miserable", "ecstatic", "upset", "excited",
        "excite", "disappoint", "amazing", "lovely", "horrible", "angry", "glad",
        "fortun", "luck", "thank", "proud", "devastat", "heartbrok", "relie",
        "regret", "hopeful", "hope", "despair", "furi", "thrill", "overjoy",
    ],
    "Arousal": [
        "excited", "excite", "calm", "relax", "boring", "drowsy", "energetic",
        "alert", "peaceful", "restless", "nervous", "tired", "adrenaline",
        "serene", "sudden", "loud", "panic", "rush", "quiet", "silen",
        "still", "slow", "frantic", "urgent", "soothe", "unhurried",
    ],
    "Care/Harm": [
        "kind", "cruel", "hurt", "help", "comfort", "neglect", "abandon",
        "care", "harm", "gentl", "save", "injur", "danger", "rescue",
        "suffer", "sorrow", "warm", "compassion", "protect", "cry", "cried",
        "tears", "alone", "lonely", "violence", "abuse", "victim",
    ],
    "Fairness": [
        "fair", "unfair", "equal", "cheat", "honest", "steal", "stole", "share",
        "greed", "reward", "deserve", "just", "unjust", "equit", "bias",
        "discriminat", "favorit", "rigged", "corrupt", "bribe",
    ],
}
AXES = ["Valence", "Arousal", "Care/Harm", "Fairness"]

# tone words that are leaky regardless of which axis the sentence belongs to
FLAT_LEAK = ["happy", "sad", "cruel", "kind", "unfair", "joy", "delight",
             "miserable", "comfort", "panic", "relie", "disappoint", "awkward",
             "heartbrok", "tragic", "beautiful", "ugly", "generous", "selfish"]

# "abstract proposition" shapes: no concrete actor/event -> not a usable stimulus
ABSTRACT_RE = [
    r"^\s*[a-z]*\s*\b(resources|money|wealth|power|justice|fairness|morality|"
    r"morals|values|trust|honesty|respect|responsibility|cooperation|conflict|"
    r"society|distribution|allocation)\b",
    r"\b(is|are)\s+(needed|required|necessary|important|essential|possible)\b",
    r"\b(should|could|might|may)\s+be\b",
    r"\bthere\s+(is|are)\s+only\b",
    # "<abstract noun> <copula>" = a claim about a category, not an event
    r"^\s*(the\s+|a\s+|an\s+)?(decision|situation|choice|issue|matter|question|"
    r"principle|lesson|idea|concept|feeling|emotion|relationship|action|act|"
    r"deed|thing|problem|dilemma|case|example|experience|thought|belief)\b[^.]*?"
    r"\b(is|are|was|were)\b",
]

# a usable stimulus mentions some concrete thing: a determiner phrase, or a
# capitalised name anywhere EXCEPT at position 0 (sentence-initial capital is
# not evidence of a concrete entity)
DET_RE = re.compile(
    r"\b(the|a|an|his|her|their|our|my|your|its|this|that|these|those|each|every)"
    r"\s+\w+", re.I
)
NAME_RE = re.compile(r"\b[A-Z][a-z]{3,}\b")


def has_concrete_marker(sentence):
    if DET_RE.search(sentence):
        return True
    for m in NAME_RE.finditer(sentence):
        if m.start() > 0:
            return True
    return False


def find_leaks(sentence, axis=None):
    """Return leak words found in `sentence` (word-boundary stem match).

    axis=None  -> check FLAT_LEAK + every axis lexicon (strictest; used for the
                  hard set and for drafts whose axis is unknown).
    axis=<str> -> check FLAT_LEAK + that axis's lexicon only.

    Matching is `\\b<stem>\\w*`, so 'gentl' catches gentle/gently but 'just' does
    NOT fire inside 'adjusted'.
    """
    low = sentence.lower()
    hits = []
    lex = {axis: LEAK[axis]} if axis is not None else dict(LEAK)
    pats = [(f"{ax}:{w}", re.compile(r"\b" + re.escape(w.strip()) + r"\w*"))
            for ax, words in lex.items() for w in words]
    pats += [(f"flat:{w}", re.compile(r"\b" + re.escape(w.strip()) + r"\w*"))
             for w in FLAT_LEAK]
    for label, pat in pats:
        if pat.search(low):
            hits.append(label)
    # de-duplicate, keep order
    seen, out = set(), []
    for h in hits:
        if h not in seen:
            seen.add(h); out.append(h)
    return out


def reason_to_reject(sentence):
    """Return a list of reasons this sentence cannot be a hard-context item
    (empty list = it passes). Reasons: LEAK, ABSTRACT, NO-CONCRETE, LENGTH."""
    reasons = []
    s = sentence.strip()
    if not s:
        return ["EMPTY"]
    leaks = find_leaks(s, axis=None)
    if leaks:
        reasons.append("LEAK:" + ",".join(leaks))
    low = s.lower()
    if any(re.search(p, low) for p in ABSTRACT_RE):
        reasons.append("ABSTRACT")
    if not has_concrete_marker(s):
        reasons.append("NO-CONCRETE")
    n = len(s.split())
    if n < 6:
        reasons.append("TOO-SHORT")
    if n > 32:
        reasons.append("TOO-LONG")
    return reasons
