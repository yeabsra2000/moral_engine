"""Doc <-> transcript audit for the step8 dose runs (no model, CPU, <1 s).

WHY THIS TEST EXISTS. The paper quotes ~35 numbers from two committed GPU
transcripts (`results/step8_care_n15.txt`, `results/step8_arousal_n25.txt`).
Every number in the docs was pasted by hand, and this repo's recurring bug
class is "a control that reads as protective but is inert" -- a doc quoting a
stale or transposed figure reads authoritative and is never re-checked by
eye. This test closes that loop mechanically:

  A. each transcript re-derives its own DECISION block from the raw per-scale
     rows (slope, sham mean/sd, z, add-one p, count, range, audits, verdict);
  B. the dose-response ladders quoted in the docs match the transcript rows
     (Care/Harm non-monotone, Arousal monotone -- both are load-bearing
     claims in the paper);
  C. every headline value in README.md / paper.md / results.md matches a
     value parsed from the transcripts (tolerance only where the docs quote
     2 dp of a 3 dp quantity);
  D. the stale-claim phrases retired in M13e are absent from the docs;
  E. the n=15 Arousal sub-distribution quoted historically in results.md
     re-derives from the first 15 rows of the persisted n=25 transcript.

Stdlib only -- deliberately runnable on any machine with the repo checkout.

Run from the repo root:  python tests/test_docs_match_transcripts.py
"""
import os
import re
import statistics
import sys

# ---- missing-file guard: actionable message instead of a bare traceback ----
# (same rationale as the guard in test_steering_collapse.py)
_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

REQUIRED = [
    "results/step8_care_n15.txt",
    "results/step8_arousal_n25.txt",
    "README.md",
    "paper.md",
    "results.md",
]
# The transcript paths above are repo-relative, so also search the repo root
# (the parent of this file's folder) — the suite must work from anywhere.
_ROOT = os.path.dirname(_HERE)
missing = [f for f in REQUIRED
           if not any(os.path.exists(os.path.join(p, f))
                      for p in (_HERE, _ROOT, os.getcwd()))]
if missing:
    raise SystemExit(
        "\nMISSING FILE(S): %s\n"
        "  This test audits the committed docs against the committed transcripts,\n"
        "  so it must run from the repo root with those files present.\n"
        "    script folder : %s\n"
        "    working dir   : %s\n" % (", ".join(missing), _HERE, os.getcwd()))


def _path(name):
    for p in (_HERE, _ROOT, os.getcwd()):
        if os.path.exists(os.path.join(p, name)):
            return os.path.join(p, name)
    return name


def read(name):
    # utf-8-sig: tolerate a BOM if the file was written by Windows tooling
    with open(_path(name), encoding="utf-8-sig") as fh:
        return fh.read()


def norm(text):
    "Unicode minus -> ASCII, so docs written with U+2212 match ASCII transcripts."
    return text.replace("\u2212", "-")


fails = []


def check(name, got, lo, hi):
    ok = lo <= got <= hi
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<66} {got:+.3f}  (want {lo} to {hi})")
    if not ok:
        fails.append(name)


# --------------------------------------------------------------------------
# transcript parsing
# --------------------------------------------------------------------------
_NUM = re.compile(r"[-+]?\d+\.\d+")


def parse_transcript(text):
    """Pull every headline quantity out of one step8 dose transcript."""
    t = norm(text)
    d = {}
    m = re.search(r"\[gate\] instrument = template-grouped held-out ([0-9.]+)", t)
    d["gate"] = float(m.group(1)) if m else None

    # dose-phase and full-run hook audits: live count must equal EXPECTED count
    m = re.search(r"\[steering audit\] (\d+)/(\d+) generations had the hook LIVE "
                  r"\((\d+) should have been\)", t)
    d["audit_dose"] = (int(m.group(1)), int(m.group(3))) if m else None
    m = re.search(r"\[steering audit incl\. shams\] (\d+)/(\d+) hook LIVE "
                  r"\((\d+) should have been\)", t)
    d["audit_full"] = (int(m.group(1)), int(m.group(3))) if m else None

    d["dose_pct"] = {int(s): float(p)
                     for s, p in re.findall(r"\[dose\]\s+scale\s+(\d+)\s+->\s+([\d.]+)%", t)}

    # per-scale rows inside a `--- direction: X ---` block
    dirs = {}
    for name, body in re.findall(
            r"--- direction: (\w+) -+\n(.*?)(?=\n--- |\n\[steering audit\])", t, re.S):
        rows = {}
        for sc, tone in re.findall(
                r"^\s*(\d+)\s+([-+]?\d+\.\d{3})\s+[-+]?\d+\.\d{3}\s+[-+]?\d+\.\d{3}", body, re.M):
            rows[int(sc)] = float(tone)
        dirs[name] = rows
    d["dirs"] = dirs

    # sham rows: `  sham N: 0=...  4=...  10=...  25=...  50=...`
    shams = {}
    for idx, body in re.findall(r"^\s*sham (\d+):\s+(.+)$", t, re.M):
        shams[int(idx)] = {int(sc): float(v)
                           for sc, v in re.findall(r"(\d+)=([-+]?\d+\.\d+)", body)}
    d["shams"] = shams

    # DECISION block
    m = re.search(r"real\s+mean: paired slope ([-+]\d+\.\d+) \+/- (\d+\.\d+) "
                  r"\(z=([-+]\d+\.\d+)\) at scale (\d+)", t)
    d["real_slope"] = float(m.group(1))
    d["real_se"] = float(m.group(2))
    d["decision_scale"] = int(m.group(4))

    m = re.search(r"sham slope distribution: mean=([-+]\d+\.\d+) sd=(\d+\.\d+) \(n=(\d+)\)", t)
    d["sham_mean"] = float(m.group(1))
    d["sham_sd"] = float(m.group(2))
    d["n"] = int(m.group(3))

    m = re.search(r"range: \[([-+]\d+\.\d+), ([-+]\d+\.\d+)\]", t)
    d["range_lo"] = float(m.group(1))
    d["range_hi"] = float(m.group(2))

    m = re.search(r"real `mean` key: ([-+]\d+\.\d+) \+/- \d+\.\d+ ->  z vs sham dist = ([-+]\d+\.\d+)", t)
    d["key_slope"] = float(m.group(1))
    d["z"] = float(m.group(2))

    m = re.search(r"SHAM\s+random: paired slope ([-+]\d+\.\d+)", t)
    d["sham_slope"] = float(m.group(1))

    m = re.search(r"empirical: (\d+)/(\d+) sham directions reach it "
                  r"\(add-one p = (\d+\.\d+)(?:; floor at this n = \d+\.\d+)?\)", t)
    d["n_ge"] = int(m.group(1))
    d["n_ge_total"] = int(m.group(2))
    d["p"] = float(m.group(3))

    m = re.search(r"VERDICT: (.+)", t)
    d["verdict"] = m.group(1).strip()
    return d


def sham_slopes(d):
    "Paired sham slopes at the DECISION scale, from the raw rows."
    s0, s1 = 0, d["decision_scale"]
    return [row[s1] - row[s0] for _, row in sorted(d["shams"].items())]


def mean_ladder(d):
    "The real (`mean` direction) tone ladder keyed by scale."
    return d["dirs"]["mean"]


print("=" * 78)
print("SECTION A -- each transcript re-derives its own DECISION block from raw rows")
print("=" * 78)

RUNS = {}
for tag, fname in (("care", "results/step8_care_n15.txt"),
                   ("arousal", "results/step8_arousal_n25.txt")):
    d = RUNS[tag] = parse_transcript(read(fname))
    slopes = sham_slopes(d)
    lad = mean_ladder(d)
    real_from_rows = lad[d["decision_scale"]] - lad[0]
    m_rec, sd_rec = statistics.mean(slopes), statistics.stdev(slopes)
    n_ge_rec = sum(1 for s in slopes if s >= real_from_rows)
    z_rec = (real_from_rows - m_rec) / sd_rec
    p_rec = (1 + n_ge_rec) / (1 + len(slopes))

    print(f"-- {tag}: n={d['n']} shams, decision scale {d['decision_scale']}")
    check(f"{tag}: real slope re-derives from mean-direction rows",
          real_from_rows, d["real_slope"] - 0.002, d["real_slope"] + 0.002)
    check(f"{tag}: sham mean re-derives from sham rows",
          m_rec, d["sham_mean"] - 0.002, d["sham_mean"] + 0.002)
    check(f"{tag}: sham sd re-derives from sham rows (sample sd)",
          sd_rec, d["sham_sd"] - 0.005, d["sham_sd"] + 0.005)
    check(f"{tag}: z re-derives from raw rows",
          z_rec, d["z"] - 0.06, d["z"] + 0.06)
    check(f"{tag}: #shams reaching the real key re-derives",
          float(n_ge_rec), float(d["n_ge"]), float(d["n_ge"]))
    check(f"{tag}: add-one p = (1+n_ge)/(1+n) matches logged",
          p_rec, d["p"] - 0.001, d["p"] + 0.001)
    check(f"{tag}: range LOW re-derives from sham rows",
          min(slopes), d["range_lo"] - 0.002, d["range_lo"] + 0.002)
    check(f"{tag}: range HIGH re-derives from sham rows",
          max(slopes), d["range_hi"] - 0.002, d["range_hi"] + 0.002)
    check(f"{tag}: gate parsed and above the 0.6 gate",
          d["gate"], 0.6, 1.001)
    check(f"{tag}: dose-phase hook audit live == expected",
          float(d["audit_dose"][0] == d["audit_dose"][1]), 1.0, 1.001)
    check(f"{tag}: full-run hook audit live == expected",
          float(d["audit_full"][0] == d["audit_full"][1]), 1.0, 1.001)
    check(f"{tag}: sham count in DECISION matches raw rows",
          float(d["n"] == len(slopes) == d["n_ge_total"]), 1.0, 1.001)
    check(f"{tag}: decision scale has a dose % in the transcript",
          float(d["decision_scale"] in d["dose_pct"]), 1.0, 1.001)

check("care verdict polarity is NOT SEPARABLE",
      float("NOT SEPARABLE" in RUNS["care"]["verdict"]), 1.0, 1.001)
check("arousal verdict polarity is (un-negated) SEPARABLE",
      float(RUNS["arousal"]["verdict"].lower().startswith("separable")
            and "NOT" not in RUNS["arousal"]["verdict"]), 1.0, 1.001)

print()
print("=" * 78)
print("SECTION B -- dose-response ladders (docs quote them; both are load-bearing)")
print("=" * 78)

care_ladder = mean_ladder(RUNS["care"])
ar_ladder = mean_ladder(RUNS["arousal"])
care_slopes = {sc: care_ladder[sc] - care_ladder[0] for sc in (4, 10, 25, 50)}
ar_tones = [ar_ladder[sc] for sc in (0, 4, 10, 25, 50)]

# Arousal: monotone increasing tone with dose (docs claim "monotone")
check("arousal ladder is strictly monotone increasing (as docs claim)",
      float(all(b > a for a, b in zip(ar_tones, ar_tones[1:]))), 1.0, 1.001)
# Care/Harm: mixed-sign slopes (docs claim "non-monotone")
signs = {+1 if v > 0 else -1 for v in care_slopes.values()}
check("care ladder slopes change sign (docs claim 'non-monotone')",
      float(len(signs) == 2), 1.0, 1.001)

care_tuple = ", ".join(f"{care_slopes[sc]:+.2f}" for sc in (4, 10, 25, 50))
ar_pattern = r"\s*(?:\u2192|,)\s*".join(re.escape(f"{v:+.3f}") for v in ar_tones)

DOCS = {name: norm(read(name)) for name in ("README.md", "paper.md", "results.md")}

for doc in ("README.md", "results.md"):
    check(f"{doc}: quotes the care slope tuple ({care_tuple})",
          float(care_tuple in DOCS[doc]), 1.0, 1.001)
    check(f"{doc}: quotes the arousal tone ladder in order",
          float(bool(re.search(ar_pattern, DOCS[doc]))), 1.0, 1.001)
    check(f"{doc}: labels the care ladder non-monotone",
          float("non-monotone" in DOCS[doc]), 1.0, 1.001)

print()
print("=" * 78)
print("SECTION C -- docs quote values that parse out of the transcripts")
print("=" * 78)


def has_num(doc_text, v, tol=1e-9):
    return any(abs(float(m.group(0)) - v) <= tol
               for m in _NUM.finditer(doc_text))


def req_num(doc, label, v, tol=1e-9):
    if tol == "either":                      # signed in one doc, |magnitude| in another
        ok = has_num(DOCS[doc], v) or has_num(DOCS[doc], abs(v))
        check(f"{doc}: quotes {label} ({v})", float(ok), 1.0, 1.001)
        return
    check(f"{doc}: quotes {label} ({v})", float(has_num(DOCS[doc], v, tol)), 1.0, 1.001)


def req_str(doc, label, s):
    check(f"{doc}: quotes {label} ({s})", float(DOCS[doc].count(s) >= 1), 1.0, 1.001)


care_vals = [
    ("care gate (held-out probe)", RUNS["care"]["gate"], 1e-9),
    ("care real slope", RUNS["care"]["real_slope"], 1e-9),
    ("care real SE", RUNS["care"]["real_se"], 1e-9),
    ("care z vs sham dist", RUNS["care"]["z"], 1e-9),
    ("care add-one p", RUNS["care"]["p"], 1e-9),
    ("care sham mean", RUNS["care"]["sham_mean"], 1e-9),
    ("care sham sd", RUNS["care"]["sham_sd"], 1e-9),
    ("care range low", RUNS["care"]["range_lo"], 1e-9),
    ("care range high", RUNS["care"]["range_hi"], 1e-9),
    ("care SHAM key slope (docs quote it signed or as |magnitude|)",
     RUNS["care"]["sham_slope"], "either"),
    ("care dose % at decision scale",
     RUNS["care"]["dose_pct"][RUNS["care"]["decision_scale"]], 1e-9),
]
ar_vals = [
    ("arousal gate (held-out probe)", RUNS["arousal"]["gate"], 1e-9),
    ("arousal real slope", RUNS["arousal"]["real_slope"], 1e-9),
    ("arousal real SE", RUNS["arousal"]["real_se"], 1e-9),
    ("arousal z vs sham dist", RUNS["arousal"]["z"], 1e-9),
    ("arousal add-one p", RUNS["arousal"]["p"], 1e-9),
    ("arousal sham mean", RUNS["arousal"]["sham_mean"], 1e-9),
    ("arousal sham sd", RUNS["arousal"]["sham_sd"], 1e-9),
    # docs quote -0.411 (recomputed from the printed rows) vs runner's -0.412:
    ("arousal range low", RUNS["arousal"]["range_lo"], 0.002),
    ("arousal range high", RUNS["arousal"]["range_hi"], 1e-9),
    ("arousal dose % at decision scale",
     RUNS["arousal"]["dose_pct"][RUNS["arousal"]["decision_scale"]], 1e-9),
]

for doc in ("README.md", "results.md"):
    for label, v, tol in care_vals + ar_vals:
        req_num(doc, label, v, tol)

# paper.md quotes a thinner set (abstract/RQ style): z's, counts, p floor, gates
for label, v in (("care gate", care_vals[0][1]), ("arousal gate", ar_vals[0][1]),
                 ("care z", care_vals[3][1]), ("arousal z", ar_vals[3][1]),
                 ("arousal add-one p (floor)", ar_vals[4][1])):
    req_num("paper.md", label, v)

for doc in ("README.md", "paper.md", "results.md"):
    req_str(doc, "arousal empirical count", "0/25")
    req_str(doc, "care empirical count", "2/15")
    check(f"{doc}: un-negated SEPARABLE verdict present",
          float(bool(re.search(r"(?<!NOT )SEPARABLE", DOCS[doc]))), 1.0, 1.001)
req_str("results.md", "full-run hook audit", "3240/4050")
check("results.md: care NOT SEPARABLE verdict quoted",
      float("NOT SEPARABLE" in DOCS["results.md"]), 1.0, 1.001)

print()
print("=" * 78)
print("SECTION D -- stale-claim guard: phrases retired in M13e stay retired")
print("=" * 78)

FORBIDDEN = [
    # (phrase, docs it must not appear in)
    ("not currently a result",             ("README.md", "paper.md", "results.md")),
    ("no measurement of what the mechanism", ("README.md", "paper.md", "results.md")),
    ("this is the blocking step",          ("README.md", "paper.md", "results.md")),
    ("BLOCKING \u2014 RE-RUN",             ("README.md", "paper.md", "results.md")),
    ("remains an open empirical question", ("README.md", "paper.md", "results.md")),
    ("still a null",                       ("README.md", "paper.md", "results.md")),
    ("cannot yet be reported",             ("README.md", "paper.md", "results.md")),
    ("citing any steering number",         ("README.md",)),
    ("transcript was never persisted",     ("README.md", "paper.md")),
    ("axis remains open",                  ("README.md", "paper.md")),
]
for phrase, docs in FORBIDDEN:
    for doc in docs:
        check(f"{doc}: forbidden stale phrase absent ({phrase!r})",
              float(phrase not in DOCS[doc]), 1.0, 1.001)

print()
print("=" * 78)
print("SECTION E -- historical n=15 Arousal numbers re-derive from the n=25 rows")
print("=" * 78)
# results.md's M13 section quotes the n=15 Arousal distribution (that run's
# transcript was never persisted). The n=25 run is a deterministic extension:
# its first 15 sham rows ARE those shams, so the quoted numbers must re-derive.
ar_dec = RUNS["arousal"]["decision_scale"]
first15 = [row[ar_dec] - row[0]
           for _, row in sorted(RUNS["arousal"]["shams"].items())[:15]]
sub_mean, sub_sd = statistics.mean(first15), statistics.stdev(first15)
check("n=15 subset mean re-derives (+0.136)", sub_mean, 0.136 - 0.002, 0.136 + 0.002)
check("n=15 subset sd re-derives (0.263)", sub_sd, 0.263 - 0.005, 0.263 + 0.005)
check("n=15 subset range low re-derives (-0.253)",
      min(first15), -0.253 - 0.002, -0.253 + 0.002)
check("n=15 subset range high re-derives (+0.697)",
      max(first15), 0.697 - 0.002, 0.697 + 0.002)
check("results.md quotes the n=15 subset mean",
      float(has_num(DOCS["results.md"], sub_mean, 0.001)), 1.0, 1.001)
check("results.md quotes the n=15 subset sd",
      float(has_num(DOCS["results.md"], sub_sd, 0.001)), 1.0, 1.001)

print()
print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
raise SystemExit(1 if fails else 0)




