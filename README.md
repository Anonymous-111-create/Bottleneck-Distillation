# Bottleneck Distillation

[English](README.md) | [中文](README_ZH.md)

This repository implements **Clear the Bottleneck: Learning When Language Agents Are Ready to Act**. It demonstrates how readiness certificates turn prerequisite states into action-level distillation signals and combine them with task outcome rewards to train a policy.

## Installation and Usage

Python 3.10 or later is required.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -e .
bd-train --updates 40 --seed 0
python -m unittest discover -s tests -v
```

The included task requires the agent to perform `inspect → verify → commit`. The training command reports the sampled trajectory success rate and readiness KL loss after each update. At inference time, call `LinearPolicy.probabilities(state, actions)` without certificates.

## Method

1. `TaskAdapter.certificates` uses executable checkers to build a certificate table containing each condition, its entity binding, and an `INACTIVE`, `SAT`, or `UNSAT` status.
2. For every `UNSAT` condition, the teacher changes only that certificate to `SAT` and computes canonical action distributions for the factual and counterfactual certificate tables. Actions in this implementation are single canonical labels, so their scores are normalized directly.
3. `method.py` computes `log p_factual - log p_completed`, RTA centered under the student distribution, Jensen-Shannon divergence with natural logarithms, the top-K weighted readiness advantage, and the KL-regularized target distribution.
4. `train.py` samples groups of trajectories for the same task and updates the student with group-normalized outcome rewards. Each outcome update is paired with one `KL(target || student)` readiness distillation update. The teacher is synchronized with the student at the beginning of every update and the target distribution is stop-gradient. The defaults `K=4`, `τ=0.7`, `β=0.05`, and `λ=0.3` follow Appendix F of the paper.

The example uses a NumPy linear policy. The teacher applies an interpretable certificate-dependent action bias to synchronized parameters, producing factual and completed-certificate responses. A language model can replace `LinearPolicy` while retaining the `probabilities(state, actions, certificates)` interface.

## Connecting Tasks and Evaluation

Implement `reset`, `actions`, `certificates`, and `step` from `TaskAdapter` in [`environment.py`](src/bottleneck_distillation/environment.py). Use `feature_fn` to map task observations to fixed-width features and pass the canonical action names to `LinearPolicy`. `step` returns the normalized terminal reward.

An external evaluator can consume the student's certificate-free action distribution directly. Real-completion and matched-control branches, together with the PCG audit, can be connected separately using snapshots of the task state.

The included task and teacher exercise the complete algorithmic path. Reproducing the cross-benchmark results reported in the paper requires the corresponding environments, datasets, language models, and real-completion audits.
