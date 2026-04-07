"""AEDA — reactive state.

Observable world model shared between reactor, policies, hooks, and bridges.
No async. No I/O. No framework deps.

Taxonomy (N1):
  Entities:  AgentSnapshot, TaskRecord, ReactiveState
  Verbs:     register (turn, handoff, activity, tokens)
  Relation:  Agent → [Snapshot], Task → [Record], Session → [Mode]
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any

from aeda.models import AgentMode, SessionMode, TaskState


##### SUB-STATE #####


@dataclass(slots=True)
class AgentSnapshot:
    """Live snapshot of an agent's operational state."""

    name: str = ""
    ref: Any = field(default=None, repr=False)
    mode: AgentMode = AgentMode.IDLE

    # Timestamps (monotonic)
    entered_at: float = 0.0
    responded_at: float = 0.0
    thought_at: float = 0.0
    tool_called_at: float = 0.0

    # Counters
    turns: int = 0
    tool_calls: int = 0
    errors: int = 0

    @property
    def active_for(self) -> float:
        """Seconds since agent became active."""
        return time.monotonic() - self.entered_at if self.entered_at else 0.0

    @property
    def silent_for(self) -> float:
        """Seconds since agent last responded."""
        return time.monotonic() - self.responded_at if self.responded_at else 0.0


@dataclass(frozen=True, slots=True)
class TaskRecord:
    """Immutable metadata for an in-flight background task."""

    task_id: str
    name: str
    submitted_at: float = field(default_factory=time.monotonic)
    state: TaskState = TaskState.PENDING


##### REACTIVE STATE #####


@dataclass(slots=True)
class ReactiveState:
    """Everything the reactive engine needs to make decisions.

    Includes budget tracking (tokens), turn counters, agent snapshots,
    task registry, and arbitrary domain data.
    """

    # Agent tracking
    agent: AgentSnapshot = field(default_factory=AgentSnapshot)
    previous_agent: AgentSnapshot = field(default_factory=AgentSnapshot)
    agents: dict[str, Any] = field(default_factory=dict)

    # Session
    session: SessionMode = SessionMode.STARTING
    started_at: float = field(default_factory=time.monotonic)

    # Turn tracking
    turn_count: int = 0
    handoff_count: int = 0
    last_turn_at: float = 0.0
    last_activity_at: float = field(default_factory=time.monotonic)

    # Budget tracking
    tokens_used: int = 0
    token_budget: int = 0

    # Task tracking
    tasks: dict[str, TaskRecord] = field(default_factory=dict)

    # Arbitrary domain data
    data: dict[str, Any] = field(default_factory=dict)

    # ── Computed: timing ──

    @property
    def session_duration(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def since_last_turn(self) -> float:
        return time.monotonic() - self.last_turn_at if self.last_turn_at else 0.0

    @property
    def inactive_for(self) -> float:
        return time.monotonic() - self.last_activity_at

    # ── Computed: budget ──

    @property
    def budget_ratio(self) -> float:
        return self.tokens_used / self.token_budget if self.token_budget else 0.0

    @property
    def budget_remaining(self) -> int:
        return max(0, self.token_budget - self.tokens_used)

    @property
    def has_budget(self) -> bool:
        return self.token_budget == 0 or self.tokens_used < self.token_budget

    # ── Computed: tasks ──

    @property
    def tasks_pending(self) -> int:
        return sum(1 for t in self.tasks.values() if t.state == TaskState.PENDING)

    @property
    def tasks_running(self) -> int:
        return sum(1 for t in self.tasks.values() if t.state == TaskState.RUNNING)

    @property
    def has_pending(self) -> bool:
        return any(
            t.state in {TaskState.PENDING, TaskState.RUNNING}
            for t in self.tasks.values()
        )

    # ── Mutations ──

    def register_turn(self) -> None:
        """Record a completed turn."""
        self.turn_count += 1
        self.agent.turns += 1
        self.last_turn_at = time.monotonic()
        self.last_activity_at = time.monotonic()

    def register_handoff(self, name: str) -> None:
        """Freeze current agent, create new snapshot for incoming agent."""
        self.previous_agent = replace(self.agent)
        self.agent = AgentSnapshot(
            name=name,
            ref=self.agents.get(name),
            entered_at=time.monotonic(),
        )
        self.handoff_count += 1

    def register_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity_at = time.monotonic()

    def register_tokens(self, count: int) -> None:
        """Add tokens consumed to running total."""
        self.tokens_used += count
