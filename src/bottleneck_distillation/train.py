"""Outcome group-relative update plus one readiness distillation update per batch."""

import argparse

import numpy as np

from .environment import TaskAdapter, ToyTask
from .method import readiness_target
from .policy import LinearPolicy


def train(
    policy: LinearPolicy,
    task: TaskAdapter,
    updates: int = 40,
    groups_per_update: int = 8,
    group_size: int = 8,
    seed: int = 0,
    learning_rate: float = 0.15,
    ready_weight: float = 0.3,
) -> list[dict[str, float]]:
    if min(updates, groups_per_update, group_size) < 1:
        raise ValueError("All training counts must be positive")
    rng = np.random.default_rng(seed)
    history = []
    for _ in range(updates):
        teacher = policy.snapshot()  # stop-gradient copy, synchronized before the rollout batch
        groups = []
        for _ in range(groups_per_update):
            episodes = []
            for _ in range(group_size):
                state = task.reset()
                decisions = []
                while True:
                    actions = task.actions(state)
                    probabilities = policy.probabilities(state, actions)
                    action_index = int(rng.choice(len(actions), p=probabilities))
                    decisions.append((state, actions, action_index, probabilities))
                    transition = task.step(state, actions[action_index])
                    state = transition.state
                    if transition.done:
                        episodes.append((decisions, transition.reward))
                        break
            groups.append(episodes)

        outcome_gradient = np.zeros_like(policy.weights)
        ready_gradient = np.zeros_like(policy.weights)
        ready_losses = []
        rewards = []
        for episodes in groups:
            group_rewards = np.array([reward for _, reward in episodes])
            advantages = (group_rewards - group_rewards.mean()) / (group_rewards.std() + 1e-8)
            for (decisions, reward), advantage in zip(episodes, advantages):
                rewards.append(reward)
                for state, actions, chosen, probabilities in decisions:
                    derivative = probabilities.copy()
                    derivative[chosen] -= 1.0
                    outcome_gradient += advantage * policy.logit_gradient(state, actions, derivative)
                    result = readiness_target(teacher, state, actions, task.certificates(state))
                    if result.js_by_condition:
                        ready_gradient += policy.logit_gradient(
                            state, actions, probabilities - result.target
                        )
                        ready_losses.append(result.kl)

        outcome_gradient /= groups_per_update * group_size
        if ready_losses:
            ready_gradient /= len(ready_losses)
        policy.weights -= learning_rate * (outcome_gradient + ready_weight * ready_gradient)
        history.append(
            {
                "success_rate": float(np.mean(rewards)),
                "ready_kl": float(np.mean(ready_losses)) if ready_losses else 0.0,
            }
        )
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Bottleneck Distillation on the toy task")
    parser.add_argument("--updates", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    history = train(LinearPolicy(seed=args.seed), ToyTask(), updates=args.updates, seed=args.seed)
    for step, metrics in enumerate(history, 1):
        print(f"update={step:03d} success={metrics['success_rate']:.3f} ready_kl={metrics['ready_kl']:.4f}")


if __name__ == "__main__":
    main()
