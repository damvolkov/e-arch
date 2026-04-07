"""Reactive session architecture — pure data models.

Enums, events, decisions, and shared state.
No async. No framework deps. No I/O.

Taxonomy (N1):
  Entities:  State, Event, Decision
  Suffixes:  Kind (enum), Action (enum), State (container)
  Verbs:     emit, evaluate, act, apply, format
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import Any


##### ENUMS #####


class EventKind(StrEnum):
    """Observable events in the reactive system."""

    USER_SPEAKING = auto()
    USER_SILENT = auto()
    USER_AWAY = auto()
    AGENT_SPEAKING = auto()
    AGENT_IDLE = auto()
    AGENT_THINKING = auto()
    TASK_COMPLETED = auto()
    TASK_FAILED = auto()
    TICK = auto()


class Action(StrEnum):
    """Executable actions on session/agent."""

    INTERRUPT = auto()
    REPLY = auto()
    SAY = auto()
    UPDATE_INSTRUCTIONS = auto()
    SWAP_AGENT = auto()


##### DATA #####


@dataclass(frozen=True, slots=True)
class Event:
    """Immutable event flowing through the reactor."""

    kind: EventKind
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.monotonic)


@dataclass(frozen=True, slots=True)
class Decision:
    """Action specification produced by a policy."""

    action: Action
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ReactiveState:
    """Shared mutable state — the contract between agents and policies."""

    agent_state: str = "idle"
    user_state: str = "listening"
    turn_count: int = 0
    last_user_activity: float = field(default_factory=time.monotonic)
    data: dict[str, Any] = field(default_factory=dict)
