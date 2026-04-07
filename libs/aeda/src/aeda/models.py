"""AEDA — pure data primitives.

Enums, events, decisions, hook results. No async. No I/O. No framework deps.

Taxonomy (N1):
  Entities:  Signal, Command, HookAction, Event, Decision, HookResult
  Suffixes:  Mode (sustained operational state), State (lifecycle phase)
  Relation:  Event → [Signal, payload, ts]
             Decision → [Command, payload]
             HookResult → [HookAction, payload, reason]
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import Any


##### SIGNALS (passive — what happened) #####


class Signal(StrEnum):
    """Observable lifecycle events. Emitted by framework hooks or internal components."""

    # Agent lifecycle
    AGENT_STARTED = auto()
    AGENT_THINKING = auto()
    AGENT_RESPONDING = auto()
    AGENT_IDLE = auto()
    AGENT_ENDED = auto()
    AGENT_ERROR = auto()

    # Tool lifecycle
    TOOL_CALLED = auto()
    TOOL_COMPLETED = auto()
    TOOL_FAILED = auto()
    TOOL_RETRIED = auto()

    # Model (LLM) lifecycle
    MODEL_REQUESTED = auto()
    MODEL_RESPONDED = auto()
    MODEL_STREAMING = auto()
    MODEL_ERROR = auto()

    # Task (background work)
    TASK_SUBMITTED = auto()
    TASK_COMPLETED = auto()
    TASK_FAILED = auto()
    TASK_TIMEOUT = auto()

    # Turn lifecycle
    TURN_STARTED = auto()
    TURN_ENDED = auto()

    # Handoff
    HANDOFF_OUT = auto()
    HANDOFF_IN = auto()

    # Session
    SESSION_STARTED = auto()
    SESSION_ENDING = auto()

    # Budget
    BUDGET_WARNING = auto()
    BUDGET_EXHAUSTED = auto()

    # State mutation notification
    STATE_CHANGED = auto()

    # Timing
    TICK = auto()


##### COMMANDS (active — what to do) #####


class Command(StrEnum):
    """Executable actions issued by policies via Decision."""

    RESPOND = auto()
    SAY = auto()
    INTERRUPT = auto()
    ENRICH = auto()
    SET_PROMPT = auto()
    SWAP = auto()
    HOLD = auto()
    RESUME = auto()


##### HOOK ACTIONS (flow control) #####


class HookAction(StrEnum):
    """What the reactor should do after a hook fires."""

    CONTINUE = auto()
    MUTATE = auto()
    BLOCK = auto()
    SKIP = auto()


##### MODES #####


class AgentMode(StrEnum):
    """Sustained operational state of an agent."""

    IDLE = auto()
    THINKING = auto()
    RESPONDING = auto()
    WAITING = auto()


class SessionMode(StrEnum):
    """Session lifecycle phase."""

    STARTING = auto()
    ACTIVE = auto()
    PAUSED = auto()
    ENDING = auto()


class TaskState(StrEnum):
    """Task lifecycle state."""

    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    TIMEOUT = auto()
    CANCELLED = auto()


##### DATA #####


@dataclass(frozen=True, slots=True)
class Event:
    """Immutable signal instance flowing into the reactor."""

    signal: Signal
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.monotonic)


@dataclass(frozen=True, slots=True)
class Decision:
    """Immutable command instance flowing out of policies."""

    command: Command
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class HookResult:
    """Returned by hook handlers to control reactor flow."""

    action: HookAction = HookAction.CONTINUE
    payload: dict[str, Any] | None = None
    reason: str | None = None
