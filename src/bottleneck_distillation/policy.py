"""A small canonical-action student with a synchronized certificate-aware teacher view."""

import numpy as np
from typing import Any, Callable

from .environment import Certificate, Status, ToyState


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    values = np.exp(shifted)
    return values / values.sum()


def toy_features(state: ToyState) -> np.ndarray:
    return np.array([1.0, float(state.inspected), float(state.verified)])


class LinearPolicy:
    def __init__(
        self,
        actions: tuple[str, ...] = ("inspect", "verify", "commit"),
        seed: int = 0,
        teacher_strength: float = 2.0,
        feature_dim: int = 3,
        feature_fn: Callable[[Any], np.ndarray] = toy_features,
    ) -> None:
        self.actions = actions
        self.teacher_strength = teacher_strength
        self.feature_fn = feature_fn
        self.weights = np.random.default_rng(seed).normal(0.0, 0.1, (feature_dim, len(actions)))

    def snapshot(self) -> "LinearPolicy":
        other = LinearPolicy(
            self.actions,
            teacher_strength=self.teacher_strength,
            feature_dim=self.weights.shape[0],
            feature_fn=self.feature_fn,
        )
        other.weights = self.weights.copy()
        return other

    def features(self, state: Any) -> np.ndarray:
        features = np.asarray(self.feature_fn(state), dtype=float)
        if features.shape != (self.weights.shape[0],):
            raise ValueError("Observation encoder returned the wrong feature shape")
        return features

    def probabilities(
        self,
        state: ToyState,
        available: tuple[str, ...],
        certificates: tuple[Certificate, ...] = (),
    ) -> np.ndarray:
        logits = self.features(state) @ self.weights
        for cert in certificates:
            if cert.status == Status.UNSAT:
                logits[self.actions.index(cert.completion_action)] += self.teacher_strength
                if "commit" in self.actions:
                    logits[self.actions.index("commit")] -= self.teacher_strength
        indices = [self.actions.index(action) for action in available]
        return softmax(logits[indices])

    def logit_gradient(
        self, state: ToyState, available: tuple[str, ...], action_gradient: np.ndarray
    ) -> np.ndarray:
        gradient = np.zeros_like(self.weights)
        for action, derivative in zip(available, action_gradient):
            gradient[:, self.actions.index(action)] = self.features(state) * derivative
        return gradient
