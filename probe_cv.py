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

Why this matters for the paper: the easy set is built from templates, so
"...felt ecstatic about the results." and "My friend felt ecstatic about the
results." are near-duplicates. Splitting them randomly across train/test makes
a probe look far better than it is (test_grouped_cv.py measures that gap with a
template-identity-only feature space).
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


def make_folds(y, seed, groups=None, k=KFOLD):
    """Return k stratified index folds.

    groups=None -> sentences split independently (optimistic).
    groups=list -> every member of a template/motif goes to ONE fold (honest).
    """
    import random
    rnd = random.Random(seed)
    if groups is not None:
        lab_of = {}
        for g, lab in zip(groups, y.tolist()):
            lab_of[g] = lab
        by_class = defaultdict(list)
        for g, lab in lab_of.items():
            by_class[lab].append(g)
        fold_groups = [[] for _ in range(k)]
        for lab in sorted(by_class):
            gs = by_class[lab][:]
            rnd.shuffle(gs)
            for j, g in enumerate(gs):
                fold_groups[j % k].append(g)
        sets = [set(f) for f in fold_groups]
        folds = [[] for _ in range(k)]
        for i, g in enumerate(groups):
            for f, s in enumerate(sets):
                if g in s:
                    folds[f].append(i)
                    break
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
            preds[te] = (torch.sigmoid(net(Xd[te]).view(-1)) > 0.5).long().cpu()
    return (preds == (y > 0).long()).float().mean().item()


def cv_report(X, y, hidden, reps=10, groups=None, k=KFOLD, epochs=400):
    """Repeated CV: (mean accuracy, 2*SE)."""
    accs = [cv_rep(X, y, hidden, s, groups, k, epochs) for s in range(reps)]
    se = statistics.stdev(accs) / (reps ** 0.5) if reps > 1 else 0.0
    return statistics.fmean(accs), 2 * se
