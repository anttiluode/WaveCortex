# Protocol: WaveCortex

Written 17 November 2026 by Claude Opus 5.5, **before** the batch. Committed and pushed before `experiments/run.py` is run; the receipt holds this file's hash.

## The question

Three repositories each did one thing:
- [WaveComputer](https://github.com/anttiluode/WaveComputer): one body perceives, screens, and imagines through A → B → C. Its "two-form" unit keeps an imagined event on time and as unsure as its own learned links.
- [WaveComputerAstra](https://github.com/anttiluode/WaveComputerAstra): two imagined hops, which need writer addresses and supplied clocks. It also insists on a scalar baseline beside every wave result.
- [ApicalRuleReversalMachine](https://github.com/anttiluode/ApicalRuleReversalMachine): a multiplying apical gate. Branches write pulse · (W + g·V), and slow weights with a fast surprise-driven state infer a hidden, uncued rule.

**The fusion asks:** when a hidden rule governs the **first** link of a chain, can one body keep the rule in its apical state and carry it through an **imagined** event to the next link? Can it do that with the imagined event as unsure as the body is, and with a cut that separates the gated link from the fixed one?

## The machine

`cortex/cortex.py` is WaveComputer's machine with ApicalRuleReversalMachine's branches. Nothing is new except that the two meet:
- **Geometry and units:** sites A = 0, B = 4, C = 7; WaveComputer's 72 calibrated latched fade clocks (AB, AC and B units, + and − halves).
- **Kept from WaveComputer:** two fields, residual learning, and the B-unit kinds for imagination (content, regen, two-form).
- **Kept from ApicalRuleReversalMachine:** every branch has W and V; one shared g; three-factor learning; normalized pooled g step (η = 0.5, κ = 0.05); slow weights (rate 0.006, 1500 training trials); g starts at +1.
- `tests/test_engine.py` checks that with the gate off it reproduces WaveComputer's verified machine exactly, in training and in every eyes-closed kind.

## The world

- A = ±1.
- True B = rule · A, flipped with probability 0.1.
- C = ρ₂ · true B, flipped with probability 0.1.
- ρ₂ = ±1, fixed per body (alternating by seed).
- The rule flips without a cue after 20–40 trials.
- Training is eyes open. Testing is 400 trials with weights frozen. Every test trial is first read, then lived (A, B and C seen; g updates).

**Reads**, all taken along A·ρ₂, where the exact answer is $`(2Q_2-1)(2Q_1-1)(2P(\text{rule}=+1)-1)`$:
- eyes closed, A alone → imagined B → C, with each B-unit kind;
- the same with the apical compartment cut (g = 0);
- cut, eyes open with the real B: the fixed link B → C.

**Comparisons on the same sequences:**
- the exact ideal observer over rule and block position;
- a per-A-sign plastic learner on C, rate chosen on its own training;
- the scalar no-wave copy of the fused algorithm.

**Transfer trial:** the first trial of an A sign after a rule switch, when the other sign was lived since the switch.

## Development, disclosed

Seeds 981–986 ([`experiments/dev_results.json`](experiments/dev_results.json)). Nothing was tuned: every constant is inherited from the two parent repos. Every rule below held in development:
- sorted 6/6;
- chain accuracy 0.87 (ideal 0.93);
- transfer 0.76 (ideal 0.77, plastic 0.07);
- the cut: chain 0.49, fixed link 1.00;
- own-belief correlation 0.86;
- error to ideal 0.091, against 0.129 for content and 0.244 for plastic.

What development also showed:
- **The ideal observer shrinks its prediction when unsure far more than any wave version does** (0.74 against about 0.90). Much of the ideal's uncertainty comes from knowing that blocks last 20–40 trials, so it grows unsure as a block ages. A body that only feels surprise cannot know that. So the confidence test is set against the body's **own** belief |g|, as WaveComputer set it against its own links.
- **Two-form beat regen** in 4 of 6 bodies on error and on own-belief tracking: too close for a rule. It is reported, not ruled.

## Decision rules (n = 32; "≥ 29" is 90%)

- **P1. Learning sorts the two links.** |V| > |W| on the A → B branches, and |W| > |V| on the B → C branches, in ≥ 29 bodies.
- **P2. The imagined chain follows the hidden rule.** Eyes-closed two-form C sign accuracy, median ≥ 0.80.
- **P3. Transfer.**
  - Two-form median ≥ 0.65 on transfer trials;
  - plastic median ≤ 0.25;
  - two-form above plastic in ≥ 29 bodies.
- **P4. A double dissociation under the cut.**
  - The imagined chain falls to chance: median accuracy 0.35–0.65;
  - the fixed link with the real B stays right: median ≥ 0.95.
- **P5. Imagination is as unsure as the body.** The correlation between |g| and |eyes-closed two-form C| has median ≥ 0.70.
- **P6. Closer to the ideal than relay content and than the plastic learner.** Squared error to the ideal is smaller than both in ≥ 29 bodies each.

**Reported without a rule:**
- two-form against regen;
- the scalar copy;
- slopes against the ideal;
- shrinkage;
- the gap to the ideal.

## Predictions, recorded before the run

- **P1–P6 hold**, near the development numbers.
- **Two-form against regen:** better in about 2/3 of bodies, on error and on own-belief tracking. A small difference: in this world, regen's threshold already silences a weak imagined B.
- **The scalar copy is closer to the ideal than the wave body in nearly all bodies** (development: 6/6).
- **All wave versions stay overconfident against the exact ideal near block ends,** for the reason above.
