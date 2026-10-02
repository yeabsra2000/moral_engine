"""Analyse a step7 log: are the flagged cells real, or generation noise?

Usage:  python analyze_step7.py [results_step7.txt]

Reports:
  1. per-pooling slope distributions for real directions vs the SHAM control
  2. a GLOBAL permutation test on (all real slopes) vs (all sham slopes) -> one
     quotable p-value for "does steering move tone anywhere in the search?"
  3. a sign-consistency check (a real steerable direction must not flip sign
     between adjacent layers)
  4. the noise floor implied by the sham spread vs the grid's flag threshold

Works on the concatenated log the script writes (all runs appended).
"""
import re, random, statistics as st, sys

path = sys.argv[1] if len(sys.argv) > 1 else "results_step7.txt"
txt = open(path, encoding="utf-8").read()

# keep only FULL runs (quick=False); drop the smoke grid
blocks = txt.split("===== step7 run")
full = [b for b in blocks if "quick=False" in b.split("\n")[0]]
if not full:                      # allow a plain excerpt (no run header) as input
    full = [txt]
body = "\n".join(full)
print(f"[{path}] analysing {len(full)} full run(s)")

pat = re.compile(r"^\s*(\d+)\s+(mean|last)\s+(\S+)\s+([\d.]+|nan)\s+"
                 r"([+-][\d.]+|nan|-)\s+([+-][\d.]+|nan|-)\s+([+-][\d.]+|nan|-)\s+(\S+.*)$")
rows = []
in_grid = False
for line in body.splitlines():
    if line.startswith("layer") and "probe" in line:
        in_grid = True; continue
    if in_grid:
        m = pat.match(line)
        if not m:
            if line.strip().startswith("Cells") or line.strip().startswith("=>"):
                in_grid = False
            continue
        L, pool, axis, probe, mean_s, pg_s, sham_s, verdict = m.groups()
        rows.append(dict(L=int(L), pool=pool, axis=axis, probe=float(probe),
                         mean=(None if mean_s == "-" else float(mean_s)),
                         pg=(None if pg_s == "-" else float(pg_s)),
                         sham=float(sham_s)))
print(f"parsed {len(rows)} grid cells\n")
if not rows:
    raise SystemExit("no GRID SUMMARY rows found - check the log path")

# ---- 1) per-pooling distributions -----------------------------------------
for pool in ("mean", "last"):
    sub = [r for r in rows if r["pool"] == pool]
    if not sub:
        continue
    shams = [r["sham"] for r in sub]
    reals = [v for r in sub for v in (r["mean"], r["pg"]) if v is not None]
    print(f"--- pool = {pool}  ({len(sub)} cells) ---")
    print(f"  SHAM slopes : mean={st.fmean(shams):+.3f}  sd={st.pstdev(shams):.3f}  "
          f"range=[{min(shams):+.2f}, {max(shams):+.2f}]")
    print(f"  REAL slopes : mean={st.fmean(reals):+.3f}  sd={st.pstdev(reals):.3f}  "
          f"range=[{min(reals):+.2f}, {max(reals):+.2f}]  (n={len(reals)})")

# ---- 2) global permutation test -------------------------------------------
real_all = [v for r in rows for v in (r["mean"], r["pg"]) if v is not None]
sham_all = [r["sham"] for r in rows]
obs = st.fmean(real_all) - st.fmean(sham_all)
pool = real_all + sham_all
na = len(real_all)
rng = random.Random(0)
N_PERM = 20000
count = 0
for _ in range(N_PERM):
    rng.shuffle(pool)
    if (st.fmean(pool[:na]) - st.fmean(pool[na:])) >= obs:
        count += 1
p = (count + 1) / (N_PERM + 1)
print("\n--- GLOBAL test (all real slopes vs all sham slopes) ---")
print(f"  n_real={na}  n_sham={len(sham_all)}")
print(f"  observed mean(real) - mean(sham) = {obs:+.4f}")
print(f"  permutation p (one-sided, {N_PERM} shuffles) = {p:.3f}")
print("  => " + ("NO detectable steering effect anywhere in the grid" if p > 0.05
                 else "POSSIBLE effect -- inspect cells"))

# ---- 3) sign consistency ---------------------------------------------------
print("\n--- sign consistency of the SAME direction across cells ---")
for k, name in (("mean", "mean"), ("pg", "probe-grad")):
    v = [r[k] for r in rows if r[k] is not None]
    pos = sum(1 for x in v if x > 0); neg = sum(1 for x in v if x < 0)
    print(f"  {name:>10}: {pos} positive, {neg} negative -> "
          f"{'SIGN-CONSISTENT' if not (pos and neg) else 'SIGN-FLIPS (noise-like)'}")

# ---- 4) noise floor vs threshold ------------------------------------------
sham_sd = st.pstdev(sham_all)
print("\n--- noise floor ---")
print(f"  sham sd (across cells)      = {sham_sd:.3f}")
print(f"  real sd (across cells)      = {st.pstdev(real_all):.3f}")
print("  grid flag threshold         = +0.10")
print("  <- a threshold below the noise sd flags cells by chance; that is why the")
print("     naive grid reported 'effects'. Use MESE_SHAM=20 + z>3 for real verdicts.")
