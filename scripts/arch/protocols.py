"""Reactive session architecture — protocols.

Framework-agnostic interfaces. Each agent framework provides implementations.

Taxonomy (N1):
  SessionHandle:  framework bridge (interrupt, say, generate_reply, ...)
  Policy:         rule evaluation (state + event → decisions)
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from e_agents.arch.models import Decision, Event, ReactiveState


@runtime_checkable
class SessionHandle(Protocol):
    """Framework-agnostic session operations.

    Each agent framework implements one concrete class.
    LiveKit: wraps AgentSession. Console: prints actions. Pipecat: wraps Pipeline.
    """

    async def interrupt(self) -> None: ...
    async def say(self, *, text: str) -> None: ...
    async def generate_reply(self, *, instructions: str) -> None: ...
    async def update_instructions(self, *, instructions: str) -> None: ...
    async def swap_agent(self, *, agent_id: str) -> None: ...


class Policy(Protocol):
    """Rule: state + event → decisions (empty tuple = no action).

    Policies are sync (fast evaluation, no I/O).
    They may hold private state (e.g., _notified flag).
    Evaluated in order; all matching policies execute (no short-circuit).
    """

    def evaluate(self, state: ReactiveState, event: Event) -> tuple[Decision, ...]: ...
