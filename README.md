# Is It the Beliefs? A Matched-Information Study of Hidden-Rule Discovery by Hypothesis-Testing, Bayesian and Learned Agents

When does explicit hypothesis testing help an agent discover hidden environment rules? This repository contains two procedural benchmark
families, a hypothesis-driven agent, a posterior-sampling (PSRL) agent, learned baselines (PPO over testable macro-actions, over generic skills,
and over primitive actions), the experiments, the statistics, and the Springer LNCS paper generated from the result files.

> Independent controlled study. It does **not** reproduce ARC-AGI-3 or any external benchmark, uses no language models, and makes no claim about
> general intelligence.

## Key results (30 evaluation seeds x 30 environments; the trained model is the statistical unit for learned agents)

Learned agents: 15 trained models for Macro-FF/GRU and 15 for Macro-GRU+belief on the mixed-difficulty distribution and regime D; 10 on regimes B, C, E; 3 on A and C2; 10 per vault architecture; 2 for generic-skill agents; 1 for primitive PPO.

| Success at difficulty | 3 | 4 | 5 |
|---|---|---|---|
| Reactive heuristic | 0.397 | 0.080 | 0.074 |
| PSRL (posterior sampling) | 0.982 | 0.879 | 0.656 |
| **Hypothesis agent** | 0.981 | 0.930 | 0.736 |
| Hypothesis-adaptive (soft exclusivity) | 0.981 | 0.930 | 0.733 |
| Macro-GRU (learned, testable macro-actions) | 0.980 | 0.767 | 0.546 |
| Macro-GRU+belief (learned, given tracker beliefs) | 0.973 | 0.919 | 0.740 |
| Generic-GRU (generic skills only; 2 models) | 0.940 | 0.561 | 0.352 |
| PPO-GRU (primitive; under-trained) | 0.000 | 0.000 | 0.000 |

* The comparison is asymmetric: the hypothesis agent is given hand-built rule templates; learned agents are not. The matched-information control (Macro-GRU+belief) removes part of that asymmetry and closes most of the gap (difficulty 5: 0.74; unseen colors: 0.98). Primitive PPO is under-trained and is not evidence about deep RL in general.
* Learned macro agents without beliefs generalize across recombined rules but not to unseen color symbols (regime D: 0.65 vs 0.98).
* Combination vault: elimination matches the closed-form failed-press prediction (3.30 vs 3.33; 8.37 vs 8.32); PPO-GRU solves small codes (1.00) but not larger ones (0.14).
* Master key: the hard-exclusivity agent drops to 0.23; the adaptive agent recovers 1.00, for detectable violations only.

See `paper/main.pdf` (LNCS) and `paper/main_ieee.pdf` (IEEE conference) for the full analysis and limitations.

## Quick start
```bash
git clone <this-repo> && cd interactive-rule-discovery
pip install -r requirements.txt
pytest -q                                   # 40 tests
python scripts/run_all.py --skip-training   # re-runs everything using the shipped trained models (~40 min, one CPU core)
```
`python scripts/run_all.py` additionally retrains all learned agents (several CPU hours; finished models carry a `.done` marker and are skipped,
delete `data/models/` to retrain from scratch). `--skip-rl` omits learned agents entirely (~25 min). The paper needs `pdflatex` and `bibtex`
with Springer's `llncs.cls` (included in `paper/`).

## Layout
| Path | Content |
|---|---|
| `environment/` | grid world (`world.py`, `rules.py`, `generator.py`, solvable-by-construction), combination vault (`vault.py`), renderer |
| `agents/` | random, reactive, hypothesis (+ manager, memory, selector), PSRL, macro/generic engines and PPO evaluation wrappers, vault agents |
| `rl/` | JAX PPO: primitive (`train_ppo.py`), macro (`train_macro.py`), generic skills (`train_generic.py`), vault (`train_vault.py`) |
| `experiments/` | baselines, scaling, generalization (regimes A-E, C2), ablations, factorial, master key, theory check, rule types, recovery, failure analysis, vault |
| `evaluation/` | metrics, bootstrap intervals, Welch / Mann-Whitney / Cohen's d, Holm correction |
| `scripts/` | `run_all.py`, stage runners, training schedules, `analyze.py` (tables) |
| `visualization/`, `paper/` | figures; Springer LNCS paper (`template.tex`, `build_paper.py` fills it from `data/results/`) |
| `data/results/`, `data/models/` | CSV/JSON results of every experiment; trained models and learning curves |

## Reproducibility notes
* Everything is deterministic given seeds; train, evaluation and test environments use disjoint seed ranges (checked by tests).
* Every number and every comparative word in the paper (higher / indistinguishable / lower) is generated from `data/results/*.csv`; a few claims are guarded by assertions that fail if the data contradict them.
* Learned agents: macro 12k (mixed) / 5k (per regime) episodes x 3 seeds; generic ~6k episodes x 2 seeds; vault 30k episodes x 3 seeds; primitive PPO 1.2M steps x 1 seed (it learned almost nothing at this budget and is reported as under-trained).
* Evaluation seed `s` uses trained model `s mod m`; see `paper/tables/table_trainseeds.csv` for the spread across training seeds.
* Development history, including bugs found in our own agents and baselines, is in `DEVLOG.md`.

## Limitations
Synthetic, deterministic, fully observable environments; hand-specified hypothesis spaces; all agents (learned included) receive hand-built skill vocabularies;
the hypothesis agent's templates match the generators; learned agents have few training seeds and small budgets.

## License
MIT. See `LICENSE`.

## Reproducing the final results
```bash
sh scripts/train_more_seeds.sh          # extends learned agents to 10 independent training runs (resumable, several CPU hours)
sh scripts/train_final_queue.sh         # regime D and Macro-GRU+belief (resumable)
sh scripts/train_belief10.sh; sh scripts/train_round2.sh   # round 2: key comparisons to 15 runs, regimes B/C/E to 10
python experiments/run_noise.py         # jammed-lock noise test (AGENT_SET=nonrl or rl)
AGENT_SET=nonrl sh scripts/run_stages.sh nonrl; AGENT_SET=rl sh scripts/run_stages.sh rl    # evaluation
python experiments/run_vault.py         # run from the repository root
python scripts/final_stats.py           # training-run-aware statistics -> paper/tables/
python visualization/final_figures.py   # figures -> paper/figures/ (PDF + PNG)
python paper/build_final.py             # writes paper/main.tex (LNCS) and main_ieee.tex (IEEEtran; also run FIG_MODE=ieee python visualization/final_figures.py) from paper/*.in and the result files; then pdflatex/bibtex
```
Only models with a `.done` marker are evaluated, so checkpoints of unfinished runs are never used.

## External check: CausalGame (survival part only)
`external/causalgame/` adapts non-language agents to the real CausalGame simulator (MIT-licensed code by the CausalGame authors; https://causalgame.github.io).
The CausalGame code itself is **not included**: clone it, set `CAUSALGAME_DIR` to the checkout and start its backend (`uvicorn api.app:app --port 8000`).
```bash
export CAUSALGAME_DIR=/path/to/CausalGame
(cd $CAUSALGAME_DIR && uvicorn api.app:app --port 8000 &)
PYTHONHASHSEED=0 python external/causalgame/run_causalgame.py 40   # 14 scenarios x 6 agents x 40 sessions -> data/results/causalgame.csv (resumable)
python external/causalgame/analyze.py                               # paper/tables/causalgame_*.{csv,tex}
```
Only the survival score is evaluated; the benchmark's LLM-judged report rubric cannot be applied to these agents. Agents see only batch survival counts.

## Papers
* `paper/main_ieee6.pdf` - 6-page IEEE conference version (source `paper/ieee6.tex.in`, built by `paper/build_final.py`).
* `paper/main.pdf` - extended Springer LNCS version.
Every number in both papers is generated from `data/results/*.csv` and `paper/tables/*.csv`; references are in `paper/references.bib` (2021-2026 entries were checked against the papers' pages or proceedings listings).
```bash
sh scripts/make_paper.sh     # statistics -> figures -> both papers
```

## Pushing to GitHub
```bash
git init && git add . && git commit -m "Matched-information study of hidden-rule discovery: code, data, models, paper"
git branch -M main && git remote add origin https://github.com/mihiit/RuleDiscovery.git && git push -u origin main
```
The repository is about 45 MB including trained models (`data/models`) and all result tables, so no Git LFS is needed.
