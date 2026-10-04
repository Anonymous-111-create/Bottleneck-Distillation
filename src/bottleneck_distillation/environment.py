"""Task interface and a small procedural task used by the runnable example."""

from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class Status(str, Enum):
    INACTIVE = "INACTIVE"
    SAT = "SAT"
    UNSAT = "UNSAT"


@dataclass(frozen=True)
class Certificate:
    condition: str
    binding: str
    completion_action: str
    status: Status


@dataclass(frozen=True)
class ToyState:
    inspected: bool = False
    verified: bool = False
    steps: int = 0


@dataclass(frozen=True)
class StepResult:
    state: ToyState
    reward: float
    done: bool


class TaskAdapter(Protocol):
    """Replace this adapter to connect task observations and executable checkers."""

    def reset(self) -> ToyState: ...

    def actions(self, state: ToyState) -> tuple[str, ...]: ...

    def certificates(self, state: ToyState) -> tuple[Certificate, ...]: ...

    def step(self, state: ToyState, action: str) -> StepResult: ...


class ToyTask:
    """Inspect a record, verify it, then commit before the three-step budget ends."""

    def reset(self) -> ToyState:
        return ToyState()

    def actions(self, state: ToyState) -> tuple[str, ...]:
        if not state.inspected:
            return ("inspect", "commit")
        if not state.verified:
            return ("verify", "commit")
        return ("commit",)

    def certificates(self, state: ToyState) -> tuple[Certificate, ...]:
        return (
            Certificate("inspect", "record-1", "inspect", Status.SAT if state.inspected else Status.UNSAT),
            Certificate(
                "verify",
                "record-1",
                "verify",
                Status.INACTIVE if not state.inspected else Status.SAT if state.verified else Status.UNSAT,
            ),
        )

    def step(self, state: ToyState, action: str) -> StepResult:
        if action not in self.actions(state):
            raise ValueError(f"Action {action!r} is unavailable")
        next_state = ToyState(
            inspected=state.inspected or action == "inspect",
            verified=state.verified or action == "verify",
            steps=state.steps + 1,
        )
        done = action == "commit" or next_state.steps >= 3
        reward = float(action == "commit" and state.verified)
        return StepResult(next_state, reward, done)
