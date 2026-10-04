"""Equations 9–15: certificate contrasts and the KL-regularized target."""

from dataclasses import dataclass

import numpy as np

from .environment import Certificate, Status, ToyState
from .policy import LinearPolicy, softmax


@dataclass(frozen=True)
class ReadinessResult:
    student: np.ndarray
    target: np.ndarray
    kl: float
    js_by_condition: dict[str, float]


def complete_certificate(table: tuple[Certificate, ...], index: int) -> tuple[Certificate, ...]:
    cert = table[index]
    if cert.status != Status.UNSAT:
        raise ValueError("Counterfactual completion requires an UNSAT certificate")
    changed = list(table)
    changed[index] = Certificate(cert.condition, cert.binding, cert.completion_action, Status.SAT)
    return tuple(changed)


def jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    midpoint = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / midpoint)) + 0.5 * np.sum(q * np.log(q / midpoint)))


def readiness_target(
    teacher: LinearPolicy,
    state: ToyState,
    actions: tuple[str, ...],
    table: tuple[Certificate, ...],
    top_k: int = 4,
    tau: float = 0.7,
    beta: float = 0.05,
) -> ReadinessResult:
    if top_k < 1 or tau <= 0 or beta <= 0:
        raise ValueError("top_k, tau, and beta must be positive")
    student = teacher.probabilities(state, actions)
    open_indices = [i for i, cert in enumerate(table) if cert.status == Status.UNSAT]
    if not open_indices:
        return ReadinessResult(student, student.copy(), 0.0, {})

    factual = teacher.probabilities(state, actions, table)
    contrasts = []
    for index in open_indices:
        completed = teacher.probabilities(state, actions, complete_certificate(table, index))
        rta = np.log(factual) - np.log(completed)
        centered = rta - np.dot(student, rta)
        magnitude = jensen_shannon(factual, completed)
        contrasts.append((index, magnitude, centered))

    selected = sorted(contrasts, key=lambda item: item[1], reverse=True)[:top_k]
    total = sum(item[1] for item in selected)
    if total == 0:
        return ReadinessResult(student, student.copy(), 0.0, {})
    gate = 1.0 - np.exp(-total / beta)
    advantage = gate * sum((magnitude / total) * centered for _, magnitude, centered in selected)
    target = softmax(np.log(student) + advantage / tau)
    kl = float(np.sum(target * (np.log(target) - np.log(student))))
    return ReadinessResult(
        student,
        target,
        kl,
        {table[index].condition: magnitude for index, magnitude, _ in selected},
    )
