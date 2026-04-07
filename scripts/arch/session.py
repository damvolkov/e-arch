"""Reactive session architecture — async state machine.

Two async processes:
  - Reactor: event-driven, consumes from queue, dispatches policies.
  - Ticker: time-driven, emits TICK events at fixed interval.

The session is the persistent entity — lives for the entire conversation.
Agents are ephemeral (swapped on handoffs). Orchestration belongs here.

Taxonomy (N1):
  Canonical verbs: run, stop, emit, evaluate, act, apply, format
  Private prefix:  _rs_ for all private methods
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable, Sequence

from e_agents.arch.models import Action, Decision, Event, EventKind, ReactiveState
from e_agents.arch.protocols import Policy, SessionHandle


class ReactiveSession:
    """Async state machine: observe events → evaluate policies → act on session.

    Processing flow:
      emit(event) → queue → reactor loop →
        _rs_apply(event)    update ReactiveState
        for policy:         evaluate(state, event) → decisions
        for decision:       _rs_act(decision) → session_handle.method()
    """

    def __init__(
        self,
        session: SessionHandle,
        state: ReactiveState,
        policies: Sequence[Policy],
        *,
        tick_interval: float = 1.0,
        log: Callable[[str, str], None] | None = None,
    ) -> None:
        self._session = session
        self._state = state
        self._policies = tuple(policies)
        self._tick_interval = tick_interval
        self._queue: asyncio.Queue[Event] = asyncio.Queue()
        self._running = False
        self._stop_event: asyncio.Event | None = None
        self._log = log or (lambda _tag, _msg: None)

    @property
    def state(self) -> ReactiveState:
        return self._state

    def emit(self, event: Event) -> None:
        """Push event into reactor queue."""
        self._queue.put_nowait(event)

    async def run(self) -> None:
        """Start reactor + ticker. Blocks until stop()."""
        self._running = True
        self._stop_event = asyncio.Event()
        self._log(
            "system",
            f"started (tick={self._tick_interval}s, policies={len(self._policies)})",
        )
        self._log("state", self._rs_format_state())

        reactor = asyncio.create_task(self._rs_reactor_loop())
        ticker = asyncio.create_task(self._rs_ticker_loop())

        await self._stop_event.wait()

        reactor.cancel()
        ticker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.gather(reactor, ticker)

        self._log("system", "stopped")

    def stop(self) -> None:
        """Signal shutdown."""
        self._running = False
        if self._stop_event is not None:
            self._stop_event.set()

    # ── Private: loops ──

    async def _rs_reactor_loop(self) -> None:
        while self._running:
            event = await self._queue.get()
            if not self._running:
                break
            await self._rs_dispatch(event)

    async def _rs_ticker_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self._tick_interval)
            if self._running:
                self.emit(Event(kind=EventKind.TICK))

    # ── Private: dispatch ──

    async def _rs_dispatch(self, event: Event) -> None:
        self._rs_apply(event)
        is_tick = event.kind == EventKind.TICK

        if not is_tick:
            payload_str = f" {event.payload}" if event.payload else ""
            self._log("react", f"{event.kind.value}{payload_str}")
            self._log("state", self._rs_format_state())

        for policy in self._policies:
            decisions = policy.evaluate(self._state, event)
            if not decisions:
                continue
            name = type(policy).__name__
            if is_tick:
                self._log("tick", f"{name} triggered")
            self._log("policy", f"{name} → {len(decisions)} decision(s)")
            for decision in decisions:
                await self._rs_act(decision)

    def _rs_apply(self, event: Event) -> None:
        """Update state from event (observation step)."""
        match event.kind:
            case EventKind.USER_SPEAKING:
                self._state.user_state = "speaking"
                self._state.last_user_activity = event.timestamp
            case EventKind.USER_SILENT:
                if self._state.user_state == "speaking":
                    self._state.turn_count += 1
                self._state.user_state = "listening"
                self._state.last_user_activity = event.timestamp
            case EventKind.USER_AWAY:
                self._state.user_state = "away"
            case EventKind.AGENT_SPEAKING:
                self._state.agent_state = "speaking"
                self._state.last_user_activity = event.timestamp
            case EventKind.AGENT_IDLE:
                self._state.agent_state = "idle"
            case EventKind.AGENT_THINKING:
                self._state.agent_state = "thinking"
            case EventKind.TASK_COMPLETED | EventKind.TASK_FAILED:
                self._state.data.update(event.payload)
                self._state.last_user_activity = event.timestamp

    async def _rs_act(self, decision: Decision) -> None:
        """Execute a decision via session handle."""
        p = decision.payload
        match decision.action:
            case Action.INTERRUPT:
                await self._session.interrupt()
                await asyncio.sleep(0.15)
            case Action.REPLY:
                await self._session.generate_reply(
                    instructions=p.get("instructions", ""),
                )
            case Action.SAY:
                await self._session.say(text=p.get("text", ""))
            case Action.UPDATE_INSTRUCTIONS:
                await self._session.update_instructions(
                    instructions=p.get("instructions", ""),
                )
            case Action.SWAP_AGENT:
                await self._session.swap_agent(
                    agent_id=p.get("agent_id", ""),
                )

    def _rs_format_state(self) -> str:
        s = self._state
        parts = [f"user={s.user_state}", f"agent={s.agent_state}", f"turns={s.turn_count}"]
        if s.data:
            parts.append(f"data={s.data}")
        return " ".join(parts)
