MESE-CR — 1–2 Page Brief

**What:** Audit of activation reading and steering for moral/emotional tone (Valence, Arousal, Care/Harm, Fairness) in Qwen2.5-1.5B-Instruct.

**Key findings (methodological):**
1. Frame leakage: polarity-locked template endings survive grouped CV. Introduced cross-group token control (shared_context_features) + regression test.
2. Pseudoreplication: seeding (axis,direction) only made samples per prompt identical (SEs understated). Fixed to (prompt,sample) while keeping slope paired.
3. Geometry: probe-derived steering keys lie in 
owspace(W1) and collapse onto mean on real activations (machine-checked).

**Empirical (audited):** Care/Harm steering = NULL under audited sham distribution. Arousal = CANDIDATE (1 of 2 seeds), with corrected seeding and larger honest SEs. step7 layer/pooling sweep with MESE_SHAM>=20 + z>3 shows no detectable steering effect globally.

**Takeaway:** Reading generalizes under grouped CV with frame/leak controls; residual-stream addition of probe-derived directions does not produce reliable steering effects at this scale after removing confounds. The work is an audit that fixes confounds rather than overclaiming.

Repo: https://github.com/yeabsra2000/moral_engine
