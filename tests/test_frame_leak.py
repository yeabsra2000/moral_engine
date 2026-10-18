"""
MESE-CR -- frame-leak regression test (model-free, fast)
========================================================
Grouped CV holds out the VERB but not any surface token that is SHARED across
template groups. The first dataset build had exactly such a token: every
Care/Harm sentence ended in a polarity-locked locative ("... on the roadside."
for +1, "... in front of everyone." for -1) and every Fairness sentence in a
polarity-locked object ("... the whole team." / "... the rest of the team.").
Those frames are shared by dozens of groups, so a one-token rule with NO model
decoded Care/Harm and Fairness perfectly -- while the activation probe it was
supposed to support claimed only ~0.5-0.8.

This test finds, per axis, the best single token that appears in MORE THAN ONE
group (i.e. a token grouped CV cannot remove) and asks how well the rule
"token present -> one pole, else the other pole" predicts the WHOLE axis. It
then re-injects the OLD frames and checks the metric spikes, so the test is
guarded against silently passing.

Metric note: accuracy is over every sentence of the axis, not only the ones
containing the token -- otherwise any phrase-unique word trivially scores 1.0.
"""
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import DATA, GROUPS  # noqa: E402

AXES = ["Valence", "Arousal", "Care/Harm", "Fairness"]
THRESH = 0.80   # no shared token may predict polarity better than this


def best_shared_token(sents, labels, groups):
    """Best full-population accuracy of a single cross-group surface token."""
    n = len(labels)
    n_pos = sum(1 for l in labels if l > 0)
    n_neg = n - n_pos
    cnt = defaultdict(lambda: [0, 0])       # token -> [n_pos, n_neg]
    seen_in = defaultdict(set)              # token -> set(group ids)
    for s, l, g in zip(sents, labels, groups):
        for w in set(s.rstrip(".").lower().split()):
            cnt[w][0 if l > 0 else 1] += 1
            seen_in[w].add(g)
    best, best_w = 0.0, None
    for w, (p, neg) in cnt.items():
        if not w or len(seen_in[w]) <= 1:   # unique to one group -> held out
            continue
        rule_a = (p + (n_neg - neg)) / n    # present -> +1, else -1
        rule_b = (neg + (n_pos - p)) / n    # present -> -1, else +1
        a = max(rule_a, rule_b)
        if a > best:
            best, best_w = a, w
    return best, best_w


def _axis_rows(ax):
    sents, labels, groups = [], [], []
    for i, (s, yv, _k) in enumerate(DATA):
        if yv[ax] != 0:
            sents.append(s); labels.append(yv[ax]); groups.append(GROUPS[i])
    return sents, labels, groups


def _reinject(ax, suf_pos, suf_neg):
    sents, labels, groups = [], [], []
    for i, (s, yv, _k) in enumerate(DATA):
        if yv[ax] != 0:
            sents.append(s.rstrip(".") + (suf_pos if yv[ax] > 0 else suf_neg))
            labels.append(yv[ax]); groups.append(GROUPS[i])
    return sents, labels, groups


def run():
    fails = 0
    print("frame-leak audit (best single surface token shared across >1 group;")
    print(f"rule must stay < {THRESH:.2f} on every axis)")
    for ax, name in enumerate(AXES):
        sents, labels, groups = _axis_rows(ax)
        acc, tok = best_shared_token(sents, labels, groups)
        ok = acc < THRESH
        fails += (not ok)
        print(f"  {name:<10} n={len(labels):>3}  best shared token = {acc:.2f} "
              f"('{tok}')   {'ok' if ok else 'LEAK'}")

    print("  control: re-injected OLD frames must still spike (proves the test bites):")
    for suf_pos, suf_neg, name, ax in (
            (" on the roadside.", " in front of everyone.", "care", 2),
            (" the whole team.", " the rest of the team.", "fair", 3)):
        sents, labels, groups = _reinject(ax, suf_pos, suf_neg)
        acc, tok = best_shared_token(sents, labels, groups)
        fired = acc >= 0.90
        fails += (not fired)
        print(f"    {name:<10} best shared token = {acc:.2f} ('{tok}')   "
              f"{'fires as expected' if fired else 'CONTROL FAILED'}")
    return fails


if __name__ == "__main__":
    n = run()
    print("PASS" if n == 0 else f"FAIL ({n} problem(s))")
    sys.exit(1 if n else 0)
