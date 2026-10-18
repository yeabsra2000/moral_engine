"""
MESE-CR — CV harness (no model dependency, so it can be unit-tested offline)
===========================================================================
Holds the split + training logic that step5_robust.py and step6_steer_all.py
share:

  make_folds(y, seed, groups)  -> K stratified folds, either RANDOM (sentences
                                  independent) or GROUPED (whole template/motif
                                  held out together, so no paraphrase leaks
                                  from train into test)
  build_net(n_in, hidden)      -> linear (hidden=0) or 2-layer SiLU MLP
  cv_rep(...)                  -> one repeated-CV pass, returns held-out accuracy
  cv_report(...)               -> mean +- 2*SE over REPS repeats
  shared_context_features(...) -> the FRAME-LEAK control features

Why this matters for the paper: the easy set is built from templates, so
"...felt ecstatic about the results." and "My friend felt ecstatic about the
results." are near-duplicates. Splitting them randomly across train/test makes
a probe look far better than it is (test_grouped_cv.py measures that gap with a
template-identity-only feature space).

FRAME LEAK (the bug this harness now guards against). Grouped CV removes the
*verb* leak, but any surface fragment that is CONSTANT ACROSS GROUPS survives
it. The first version of dataset_templates.py ended every Care/Harm sentence
with a fixed locative that tracked polarity ("... on the roadside." for +1,
"... in front of everyone." for -1) and every Fairness sentence with a fixed
object ("... the whole team." / "... the rest of the team."). Those frames were
shared by MANY groups, so a frame-only rule is invisible to both random and
grouped CV on the activation probe -- and it decodes Care/Harm at ~0.96 and
Fairness at ~0.87 with no model at all. shared_context_features() builds exactly
that (model-free) feature space so the reading numbers can be checked against
the best purely-surface rule.
"""
import statistics
from collections import defaultdict

import torch

KFOLD = 5


def build_net(n_in, hidden, device="cpu"):
    net = (torch.nn.Sequential(torch.nn.Linear(n_in, hidden), torch.nn.SiLU(),
                               torch.nn.Linear(hidden, 1))
           if hidden else torch.nn.Sequential(torch.nn.Linear(n_in, 1)))
    return net.to(device)


def template_only_features(groups, noise=0.05, seed=0):
    """CONTROL feature space: one-hot(template id) + noise.

    It contains no tone information whatsoever, so any accuracy a probe reaches
    on it is pure template memorization. Running the same CV on these features
    gives the baseline the random-split probe number must be compared against
    (see test_grouped_cv.py). Requires torch, no model.
    """
    ids = sorted(set(groups))
    idx = {g: i for i, g in enumerate(ids)}
    onehot = torch.nn.functional.one_hot(
        torch.tensor([idx[g] for g in groups]), num_classes=len(ids)).float()
    if noise:
        g = torch.Generator().manual_seed(seed)
        onehot = onehot + noise * torch.randn(onehot.shape, generator=g)
    return onehot


def shared_context_features(sentences, groups):
    """FRAME-LEAK control: one-hot over tokens that appear in MORE THAN ONE group.

    A grouped split holds out whole template/motif groups, so a token that is
    unique to a single group cannot leak; but a token (or fixed suffix) shared
    across groups is still visible to a probe trained on the held-in groups.
    This feature space keeps ONLY those cross-group surface tokens, so it is a
    model-free lower bound on how much a reading claim could be explained by a
    shared FRAME rather than by tone.

    Run the SAME grouped CV on these features. If the resulting accuracy is at
    or above the activation probe's, the reading is a surface artifact, not
    activation-level tone (this is exactly how the Care/Harm locative and the
    Fairness object frames were caught). Requires torch, no model.
    """
    toks = [[w.strip(".,;:!?\"'()").lower() for w in s.split()] for s in sentences]
    groups_per_tok = defaultdict(set)
    for tl, g in zip(toks, groups):
        for w in set(tl):
            groups_per_tok[w].add(g)
    vocab = sorted(w for w, gs in groups_per_tok.items() if len(gs) > 1 and w)
    idx = {w: i for i, w in enumerate(vocab)}
    feats = torch.zeros(len(sentences), max(len(vocab), 1))
    for i, tl in enumerate(toks):
        for w in tl:
            if w in idx:
                feats[i, idx[w]] = 1.0
    return feats[:, :len(vocab)] if vocab else feats


def make_folds(y, seed, groups=None, k=KFOLD):
    """Return k stratified index folds.

    groups=None -> sentences split independently (optimistic).
    groups=list -> every member of a template/motif goes to ONE fold (honest).

    The GROUPED branch assigns whole groups to folds so that (a) no group is
    ever split across folds, and (b) the per-fold POSITIVE RATE stays close to
    the global rate even when a group straddles both polarities. The previous
    version kept a single last-write-wins label per group, so a mixed-polarity
    (hard) motif was treated as pure and the folds came out badly imbalanced
    (measured positive-rate spread up to 0.74). Here each group contributes its
    own (n_pos, n_total) and is placed in the fold that keeps the running
    positive rate nearest the global one; ties are broken by fold size and,
    within a size bucket, by `seed`, so repeated CV still varies across seeds.
    """
    import random
    rnd = random.Random(seed)
    if groups is not None:
        members = defaultdict(list)
        for i, g in enumerate(groups):
            members[g].append(i)
        yl = y.tolist()
        g_pos = {g: sum(1 for i in idxs if yl[i] > 0) for g, idxs in members.items()}
        g_neg = {g: len(idxs) - g_pos[g] for g, idxs in members.items()}
        g_tot = {g: len(idxs) for g, idxs in members.items()}
        # Separate pure-polarity, mixed-polarity and label-free groups. Pure
        # groups are assigned to balance that class's count across folds; mixed
        # (hard mirror-pair) groups carry one row of each pole, so they only need
        # size balancing. The old code kept a single last-write-wins label per
        # group, which treated a mixed pair as pure and left folds badly
        # imbalanced (positive-rate spread up to 0.74).
        pure_pos = [g for g in members if g_pos[g] and not g_neg[g]]
        pure_neg = [g for g in members if g_neg[g] and not g_pos[g]]
        mixed = [g for g in members if g_pos[g] and g_neg[g]]
        zero = [g for g in members if not g_pos[g] and not g_neg[g]]
        for bucket in (pure_pos, pure_neg, mixed, zero):
            rnd.shuffle(bucket)          # seed-dependent -> repeated CV varies
        folds = [[] for _ in range(k)]
        f_pos, f_neg, f_tot = [0] * k, [0] * k, [0] * k
        tie = [rnd.random() for _ in range(k)]   # seed-dependent fold permutation

        def assign(g, key):
            f = min(range(k), key=key)
            folds[f].extend(members[g])
            f_pos[f] += g_pos[g]
            f_neg[f] += g_neg[g]
            f_tot[f] += g_tot[g]

        # largest groups first so they cannot be stranded in a single fold.
        # Pure groups are keyed on their OWN class count only (no size term in the
        # tie-break): mixing size into the key would pull one class into the folds
        # the other class left small and re-break the balance it is meant to fix.
        # `tie` breaks equal-count ties differently per seed, so repeated CV still
        # varies (otherwise a deterministic balanced assignment would give every
        # repeat the same folds and an SE of 0).
        for g in sorted(pure_pos, key=lambda g: -g_tot[g]):
            assign(g, lambda f: (f_pos[f], tie[f]))
        for g in sorted(pure_neg, key=lambda g: -g_tot[g]):
            assign(g, lambda f: (f_neg[f], tie[f]))
        for g in sorted(mixed, key=lambda g: -g_tot[g]):
            assign(g, lambda f: (abs(f_pos[f] - f_neg[f]), f_tot[f], tie[f]))
        for g in sorted(zero, key=lambda g: -g_tot[g]):
            assign(g, lambda f: (f_tot[f], tie[f]))
        return folds
    cl = defaultdict(list)
    for i, v in enumerate(y.tolist()):
        cl[v].append(i)
    for c in cl:
        rnd.shuffle(cl[c])
    folds = [[] for _ in range(k)]
    for c, mem in cl.items():
        for f in range(k):
            folds[f] += mem[f::k]
    return folds


def train_probe(X, y, hidden=64, epochs=400, lr=1e-3, wd=1e-4, device=None):
    """Fit one probe on (X, y) with labels +-1 -> returns the trained net."""
    device = device or X.device
    net = build_net(X.size(1), hidden, device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    yt = (y > 0).float().to(device)
    Xd = X.to(device)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(Xd).view(-1), yt)
        loss.backward()
        opt.step()
    net.eval()
    return net


def cv_rep(X, y, hidden, seed, groups=None, k=KFOLD, epochs=400):
    """One K-fold pass; returns held-out accuracy (chance = 0.5)."""
    folds = make_folds(y, seed, groups, k)
    Xd = X.to(X.device)
    preds = torch.zeros(y.numel(), dtype=torch.long)
    for f in range(k):
        te = folds[f]
        tr = [i for i in range(y.numel()) if i not in set(te)]
        if not tr or not te:
            continue
        net = train_probe(Xd[tr], y[tr], hidden, epochs=epochs)
        with torch.no_grad():
            pred = (torch.sigmoid(net(Xd[te]).view(-1)) > 0.5).long().cpu()
        preds[te] = pred
    return (preds == (y > 0).long()).float().mean().item()


def cv_report(X, y, hidden, seed=0, groups=None, k=KFOLD, epochs=400, reps=10):
    """Mean +- 2*SE over `reps` repeated K-fold passes."""
    accs = [cv_rep(X, y, hidden, seed + r, groups, k, epochs) for r in range(reps)]
    m = statistics.fmean(accs)
    se = statistics.stdev(accs) / (reps ** 0.5) if reps > 1 else 0.0
    return m, 2.0 * se
