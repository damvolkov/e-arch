"""AEDA — protocols.

Framework-agnostic interfaces and type aliases.

Taxonomy (N1):
  SessionHandle:  framework bridge (respond, say, interrupt, enrich, set_prompt, swap)
  Policy:         rule evaluation (state + event → decisions)
  HookHandler:    lifecycle interceptor (event → HookResult with flow control)
  Subscriber:     async event observer (fire-and-forget)
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Protocol, runtime_checkable

from aeda.models import Decision, Event, HookResult
from aeda.state import ReactiveState


##### TYPES #####

type Subscriber = Callable[[Event], Awaitable[None]]
type HookHandler = Callable[[Event], Awaitable[HookResult]]


##### HANDLE #####


@runtime_checkable
class SessionHandle(Protocol):
    """Framework-agnostic session operations.

    Each agent framework implements one concrete class:
      - LiveKit: wraps AgentSession
      - Pydantic-AI: wraps RunContext + Agent
      - Console: prints actions (testing)
    """

    async def respond(self, *, instructions: str) -> None: ...
    async def say(self, *, text: str) -> None: ...
    async def interrupt(self) -> None: ...
    async def enrich(self, *, data: dict[str, Any]) -> None: ...
    async def set_prompt(self, *, prompt: str) -> None: ...
    async def swap(self, *, agent_id: str) -> None: ...


##### POLICY #####


class Policy(Protocol):
    """Rule: state + event → decisions.

    Sync (fast evaluation, no I/O). May hold private mutable state.
    Evaluated in order; all matching policies execute (no short-circuit).
    """

    def evaluate(self, state: ReactiveState, event: Event) -> tuple[Decision, ...]: ...
