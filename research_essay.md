# Research Essay: Multi-Dimensional Morality & Emotional Steering in Large Language Models

## Introduction

This project investigates whether large language models can genuinely understand and manipulate moral and emotional dimensions of language. Building on prior work in concept probing and activation steering, we developed a novel framework — the Multi-Dimensional Morality & Emotional Steering Engine (MESE-CR) — that treats moral alignment, emotional state, and situational intent as structured geometric pathways within the model's high-dimensional activation space. Using held-out validation sets spanning valence, arousal, and fairness axes, we evaluated the ability of small probe networks to read emotional/moral tone directly from model activations, offering a potential pathway for interpretability-driven alignment.

## Methodology

Our experimental pipeline consisted of three phases:

1. **Probe Development** — We constructed small, focused probe networks that map specific activation subspaces to target dimensions (valence, arousal, fairness). These probes were designed to be fast, memory-efficient, and capable of operating on intermediate layers of the residual stream.

2. **Steering Experiments** — Using the MESE-CR architecture, we tested whether targeted interventions (steering hooks) could move model behavior along desired moral/emotional axes. The system included a concept bottleneck engine embedded directly in the residual stream, enabling direct manipulation of activation patterns associated with specific ethical dimensions.

3. **Auditing & Validation** — To ensure robustness, we implemented a per-generation steering audit that tracks whether the steering hook was actually active during inference. This audit revealed previously hidden failures: the original implementation armed the hook only once per scale and disarmed it on the first sample, causing any treatment effect to be diluted approximately 20× into the noise floor. Additionally, we identified multiple undetectable bugs — such as scaling violations and seed inconsistencies — that standard measurement controls could not uncover.

## Key Findings

### Robust Performance on Some Axes
On the Care/Harm dimension, the steering null remained stable across independent seeds, confirming that moral tone can indeed be read from model activations with high accuracy (0.75–1.00). The Arousal axis showed promising but less conclusive results: the steering effect was observed on one of two independent seeds, indicating a candidate finding that requires further replication.

### Critical Flaws Exposed
Several methodological weaknesses undermined confidence in the initial results:

- **Treatment Dilution Bug** — The original steering loop armed the hook only once per scale and disarmed it on the first sample. This caused any treatment effect to be diluted roughly 20× into the sham noise floor, rendering many null results indistinguishable from chance.

- **Undetectable Bugs** — Multiple control failures (scaling violations, seed inconsistency, improper seeding) were present but invisible to conventional measurement protocols. Only a dedicated treatment audit could reveal these issues.

- **Seed-Inconsistency** — The Care/Harm null was not fully reliable; one of two independent seeds failed to produce the expected separation, suggesting the result could not be consistently reproduced.

### Ethical Dual-Use Concern
Beyond technical performance, our work highlights a broader concern: activation steering can be leveraged both for beneficial alignment and for malicious purposes (e.g., forcing AI to generate harmful content). This dual-use nature demands careful governance and responsible deployment.

## Implications

The exposure of the 20× dilution bug and other hidden flaws has important consequences for AI alignment research. It demonstrates that naive implementations of activation steering can produce misleading conclusions, and that rigorous auditing (including per-generation hook tracking) is essential. Our findings underscore the need for transparent reporting of implementation details and for independent verification of sensitive alignment methods.

## Conclusion

This research advances the understanding of how large language models represent moral and emotional concepts in their internal activations. While some steering effects are robust (particularly on Care/Harm), others remain uncertain due to implementation artifacts. Most importantly, the discovery of systematic biases in the original pipeline — particularly the treatment dilution bug — serves as a cautionary tale for future work in activation-based alignment. Rigorous auditing, open code, and reproducible experiments are indispensable for building trustworthy AI systems.