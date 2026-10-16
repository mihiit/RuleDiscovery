# Development log (honesty record)

Two degenerate behaviours of the hypothesis agent were found in the first full run and fixed **before** the reported final runs:

1. **Refuted-hypothesis retest loop.** A refuted key/door pair (probability ~0) still had the highest, least-negative utility
   (cheap, nearby) when useful experiments were far away; one episode re-tested it 163 times.
   Fix: macros with R < 0.1 and IG < 0.2 are discarded when any useful macro exists.
2. **Key ping-pong.** The agent swapped keys back and forth to collect an information-gain bonus for doors it could not reach.
   Fix: only doors with a reachable adjacent cell count as test targets.

Intermediate pre-fix numbers (difficulty-5 success ~0.62, then ~0.71) are not used anywhere in the paper's tables or figures.
Also corrected during review: the paper draft claimed information-gain weight reduced redundant tests; the data show it does not
(redundant tests constant, differences in success not significant after Holm correction). Text and guards were updated.

## Added for the conference version
- PSRL (posterior sampling) baseline; its first version looped (key ping-pong, unreachable sample, set-order nondeterminism) - fixed with sticky samples, resampling on no progress, sorted candidates.
- Primitive-action PPO needed three fixes (terminal potential, key-pickup penalty, stability) and still learned almost nothing on one CPU core; reported as under-trained.
- Macro-level PPO (FF/GRU) over the same macro-actions with no belief features: strong learned baseline.
- Factorial ablation (reward form x information bonus), master-key assumption-violation study, Proposition 1 with an empirical check (the check initially used nominal n and had to use the number of reachable keys).
- A RL evaluation stage once overwrote generalization.csv with an empty file; save() now refuses empty partial writes.
