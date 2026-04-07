"""Reactive session architecture — pure data models.

No async. No framework deps. No I/O.

Taxonomy:
  Actors:     Agent, User, Session, Task
  Modes:      AgentMode, UserMode, SessionMode  (ongoing, has duration)
  Signals:    Signal enum  (passive — what HAPPENED)
  Commands:   Command enum (active — what TO DO)
  Data:       Event (signal instance), Decision (command instance)
  Sub-state:  AgentState, UserState (composable snapshots)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from enum import StrEnum, auto, nonmember
from typing import Any

##### MODES #####


class AgentMode(StrEnum):
    """What the agent is doing right now."""

    _color = nonmember("\033[34m")  # blue

    IDLE = auto()
    LISTENING = auto()
    THINKING = auto()
    SPEAKING = auto()


class UserMode(StrEnum):
    """What the user is doing right now."""

    _color = nonmember("\033[36m")  # cyan

    SPEAKING = auto()
    SILENT = auto()
    IDLE = auto()
    AWAY = auto()


class SessionMode(StrEnum):
    """Session lifecycle."""

    _color = nonmember("\033[35m")  # magenta

    STARTING = auto()
    ACTIVE = auto()
    PAUSED = auto()
    ENDING = auto()


##### SIGNALS #####


class Signal(StrEnum):
    """Passive — what HAPPENED, emitted by framework hooks."""

    # User (source: VAD, input detection, connection)
    USER_SPOKE = auto()
    USER_STOPPED = auto()
    USER_IDLE = auto()
    USER_LEFT = auto()
    USER_BACK = auto()
    USER_BARGED_IN = auto()

    # Agent (source: LLM pipeline, TTS pipeline)
    AGENT_THINKING = auto()
    AGENT_SPOKE = auto()
    AGENT_DONE = auto()
    AGENT_CUT_OFF = auto()
    AGENT_TOOL_CALL = auto()
    AGENT_TOOL_RESULT = auto()

    # Handoff (source: agent transfer mechanism)
    HANDOFF_OUT = auto()
    HANDOFF_IN = auto()

    # Task (source: background orchestrator)
    TASK_SENT = auto()
    TASK_DONE = auto()
    TASK_FAILED = auto()
    TASK_TIMEOUT = auto()

    # Session (source: system lifecycle)
    SESSION_STARTED = auto()
    SESSION_ENDING = auto()
    TICK = auto()

    @property
    def color(self) -> str:
        prefix = self.name.split("_", 1)[0]
        return SIGNAL_COLORS.get(prefix, "\033[0m")


SIGNAL_COLORS: dict[str, str] = {
    "USER": UserMode._color,
    "AGENT": AgentMode._color,
    "HANDOFF": "\033[95m",  # bright magenta
    "TASK": "\033[33m",  # yellow
    "SESSION": SessionMode._color,
    "TICK": "\033[90m",  # gray
}

STATE_COLOR: str = "\033[32m"  # green — state mutations


##### COMMANDS #####


class Command(StrEnum):
    """Active — what TO DO, issued by the reactive engine."""

    _color = nonmember("\033[91m")  # bright red

    # Output
    SAY = auto()
    REPLY = auto()
    INTERRUPT = auto()
    INTERRUPT_SAY = auto()
    INTERRUPT_REPLY = auto()

    # Context
    ENRICH = auto()
    SET_PROMPT = auto()

    # Agent
    SWAP = auto()

    # Task
    FLUSH = auto()
    CANCEL_TASK = auto()

    # Control
    HOLD = auto()


##### DATA #####


@dataclass(frozen=True, slots=True)
class Event:
    """Immutable signal instance flowing into the reactor."""

    signal: Signal
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.monotonic)


@dataclass(frozen=True, slots=True)
class Decision:
    """Immutable command instance flowing out of the reactor."""

    command: Command
    payload: dict[str, Any] = field(default_factory=dict)


##### SUB-STATE #####


@dataclass(slots=True)
class AgentState:
    """Live snapshot of an agent."""

    name: str = ""
    ref: Any = field(default=None, repr=False)
    mode: AgentMode = AgentMode.IDLE

    entered_at: float = 0.0
    spoke_at: float = 0.0
    finished_at: float = 0.0
    thought_at: float = 0.0
    tool_called_at: float = 0.0

    turns: int = 0
    tool_calls: int = 0
    interruptions: int = 0

    @property
    def active_for(self) -> float:
        return time.monotonic() - self.entered_at if self.entered_at else 0.0

    @property
    def silent_for(self) -> float:
        return time.monotonic() - self.finished_at if self.finished_at else 0.0


@dataclass(slots=True)
class UserState:
    """Live snapshot of user activity."""

    mode: UserMode = UserMode.SILENT

    spoke_at: float = 0.0
    stopped_at: float = 0.0
    transcript_at: float = 0.0
    active_at: float = field(default_factory=time.monotonic)

    interrupts: int = 0
    messages: int = 0

    @property
    def silent_for(self) -> float:
        return time.monotonic() - self.stopped_at if self.stopped_at else 0.0

    @property
    def inactive_for(self) -> float:
        return time.monotonic() - self.active_at


##### REACTIVE STATE #####


@dataclass(slots=True)
class ReactiveState:
    """Everything the reactive engine needs to make decisions."""

    current: AgentState = field(default_factory=AgentState)
    previous: AgentState = field(default_factory=AgentState)
    agents: dict[str, Any] = field(default_factory=dict)

    user: UserState = field(default_factory=UserState)

    session: SessionMode = SessionMode.STARTING
    started_at: float = field(default_factory=time.monotonic)
    last_turn_at: float = 0.0

    turn_count: int = 0
    handoff_count: int = 0
    tasks_running: int = 0
    tasks_pending: int = 0

    data: dict[str, Any] = field(default_factory=dict)

    # TODO(temp): PoC wake-word reaction — remove after testing
    fake_reactive_sentence: str = "Me encanta comer caramelos, amigo"

    @property
    def session_duration(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def since_last_turn(self) -> float:
        return time.monotonic() - self.last_turn_at if self.last_turn_at else 0.0

    @property
    def has_pending(self) -> bool:
        return self.tasks_pending > 0

    def register_handoff(self, name: str) -> None:
        self.previous = replace(self.current)
        self.current = AgentState(
            name=name,
            ref=self.agents.get(name),
            entered_at=time.monotonic(),
        )
        self.handoff_count += 1

    def register_turn(self) -> None:
        self.turn_count += 1
        self.current.turns += 1
        self.last_turn_at = time.monotonic()
