"""AEDA — reactive engine.

The Reactor: async state machine that processes events through a gated pipeline.

Pipeline (per event):
  1. bus.dispatch(event)     → hooks fire (can BLOCK the event)
  2. _re_apply(event)        → deterministic state mutation
  3. policies evaluate       → state + event → decisions
  4. _re_execute(decisions)  → SessionHandle method calls
  5. bus.notify(event)       → subscribers see updated state

Taxonomy (N1):
  Entities:  Reactor
  Verbs:     emit, run, stop (public)
  Private:   _re_reactor_loop, _re_ticker_loop, _re_dispatch, _re_apply, _re_execute
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable, Sequence

from aeda.bus import EventBus
from aeda.models import (
    AgentMode,
    Command,
    Decision,
    Event,
    HookAction,
    SessionMode,
    Signal,
)
from aeda.protocols import Policy, SessionHandle
from aeda.state import ReactiveState
from aeda.task import TaskExecutor, TaskRegistry


class Reactor:
    """Async state machine: observe events → gate via hooks → apply state → evaluate policies → act.

    The Reactor owns an internal asyncio.Queue for event ingestion and an
    EventBus for hook flow-control and subscriber notification.

    Two concurrent loops:
      - Reactor loop: consumes events, runs the full pipeline.
      - Ticker loop: emits TICK events at configurable interval.
    """

    __slots__ = (
        "_queue",
        "_bus",
        "_state",
        "_handle",
        "_policies",
        "_task_registry",
        "_task_executor",
        "_tick_interval",
        "_running",
        "_paused",
        "_stop_event",
        "_log",
    )

    def __init__(
        self,
        handle: SessionHandle,
        state: ReactiveState,
        policies: Sequence[Policy],
        *,
        bus: EventBus | None = None,
        tick_interval: float = 1.0,
        log: Callable[[str, str], None] | None = None,
    ) -> None:
        self._queue: asyncio.Queue[Event] = asyncio.Queue()
        self._bus = bus or EventBus()
        self._state = state
        self._handle = handle
        self._policies = tuple(policies)
        self._tick_interval = tick_interval
        self._running = False
        self._paused = False
        self._stop_event: asyncio.Event | None = None
        self._log = log or (lambda _tag, _msg: None)

        # Task management — emit injected to break circular dep
        self._task_registry = TaskRegistry()
        self._task_executor = TaskExecutor(self.emit, self._task_registry)

    # ── Public API ──

    @property
    def state(self) -> ReactiveState:
        return self._state

    @property
    def bus(self) -> EventBus:
        return self._bus

    @property
    def tasks(self) -> TaskExecutor:
        return self._task_executor

    @property
    def running(self) -> bool:
        return self._running

    def emit(self, event: Event) -> None:
        """Push event into reactor queue. Non-blocking, sync-safe."""
        self._queue.put_nowait(event)

    async def run(self) -> None:
        """Start reactor + ticker loops. Blocks until stop()."""
        self._running = True
        self._stop_event = asyncio.Event()
        self._state.session = SessionMode.ACTIVE
        self._log(
            "engine",
            f"started (tick={self._tick_interval}s, policies={len(self._policies)})",
        )

        reactor = asyncio.create_task(self._re_reactor_loop())
        ticker = asyncio.create_task(self._re_ticker_loop())

        await self._stop_event.wait()

        reactor.cancel()
        ticker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.gather(reactor, ticker)

        self._state.session = SessionMode.ENDING
        self._log("engine", "stopped")

    def stop(self) -> None:
        """Signal shutdown."""
        self._running = False
        if self._stop_event is not None:
            self._stop_event.set()

    # ── Private: loops (_re_ = Reactor Engine) ──

    async def _re_reactor_loop(self) -> None:
        """Main loop — consume events from queue, run through pipeline."""
        while self._running:
            event = await self._queue.get()
            if not self._running:
                break
            await self._re_dispatch(event)

    async def _re_ticker_loop(self) -> None:
        """Periodic TICK emitter for time-based policies."""
        while self._running:
            await asyncio.sleep(self._tick_interval)
            if self._running and not self._paused:
                self.emit(Event(signal=Signal.TICK))

    # ── Private: dispatch pipeline ──

    async def _re_dispatch(self, event: Event) -> None:
        """Full event pipeline: hooks → apply → policies → execute → notify.

        Hooks fire FIRST and can BLOCK the event from further processing.
        Subscribers fire LAST and always see the final state.
        """
        is_tick = event.signal == Signal.TICK

        # Phase 1: hooks (flow control gate)
        hook_result = await self._bus.dispatch(event)
        if hook_result.action == HookAction.BLOCK:
            self._log("hook", f"BLOCKED {event.signal.value}: {hook_result.reason}")
            return

        # Phase 2: state mutation
        self._re_apply(event)

        if not is_tick:
            payload_str = f" {event.payload}" if event.payload else ""
            self._log("event", f"{event.signal.value}{payload_str}")

        # Phase 3: policy evaluation (skipped when paused)
        if not self._paused:
            for policy in self._policies:
                decisions = policy.evaluate(self._state, event)
                if not decisions:
                    continue
                name = type(policy).__name__
                if is_tick:
                    self._log("tick", f"{name} triggered")
                self._log("policy", f"{name} -> {len(decisions)} decision(s)")
                for decision in decisions:
                    await self._re_execute(decision)

        # Phase 4: subscriber notification (always, even when paused)
        await self._bus.notify(event)

    def _re_apply(self, event: Event) -> None:
        """Deterministic state mutation from event signal."""
        p = event.payload
        match event.signal:
            # Agent lifecycle
            case Signal.AGENT_STARTED:
                if (name := p.get("agent_id", p.get("name", ""))):
                    self._state.agent.name = name
                self._state.agent.mode = AgentMode.IDLE
                self._state.agent.entered_at = event.timestamp
                self._state.register_activity()
            case Signal.AGENT_THINKING:
                self._state.agent.mode = AgentMode.THINKING
                self._state.agent.thought_at = event.timestamp
            case Signal.AGENT_RESPONDING:
                self._state.agent.mode = AgentMode.RESPONDING
                self._state.agent.responded_at = event.timestamp
                self._state.register_activity()
            case Signal.AGENT_IDLE:
                self._state.agent.mode = AgentMode.IDLE
            case Signal.AGENT_ENDED:
                self._state.agent.mode = AgentMode.IDLE
            case Signal.AGENT_ERROR:
                self._state.agent.mode = AgentMode.IDLE
                self._state.agent.errors += 1
            # Tool lifecycle
            case Signal.TOOL_CALLED:
                self._state.agent.mode = AgentMode.WAITING
                self._state.agent.tool_calls += 1
                self._state.agent.tool_called_at = event.timestamp
            case Signal.TOOL_COMPLETED | Signal.TOOL_FAILED:
                self._state.agent.mode = AgentMode.THINKING
            # Model lifecycle
            case Signal.MODEL_REQUESTED:
                self._state.agent.mode = AgentMode.THINKING
                self._state.agent.thought_at = event.timestamp
            case Signal.MODEL_RESPONDED:
                self._state.register_activity()
                if (tokens := p.get("tokens_used", 0)):
                    self._state.register_tokens(tokens)
            case Signal.MODEL_ERROR:
                self._state.agent.errors += 1
            # Turn lifecycle
            case Signal.TURN_STARTED:
                self._state.register_activity()
            case Signal.TURN_ENDED:
                self._state.register_turn()
            # Task lifecycle
            case Signal.TASK_SUBMITTED:
                self._state.register_activity()
            case Signal.TASK_COMPLETED:
                self._state.register_activity()
                if (task_id := p.get("task_id")):
                    self._task_registry.resolve(task_id)
                if isinstance(p.get("result"), dict):
                    self._state.data.update(p["result"])
            case Signal.TASK_FAILED:
                self._state.register_activity()
                if (task_id := p.get("task_id")):
                    self._task_registry.resolve(task_id)
            case Signal.TASK_TIMEOUT:
                if (task_id := p.get("task_id")):
                    self._task_registry.resolve(task_id)
            # Handoff
            case Signal.HANDOFF_OUT:
                if (agent_id := p.get("agent_id")):
                    self._state.register_handoff(agent_id)
            case Signal.HANDOFF_IN:
                self._state.register_activity()
            # Session
            case Signal.SESSION_STARTED:
                self._state.session = SessionMode.ACTIVE
            case Signal.SESSION_ENDING:
                self._state.session = SessionMode.ENDING

    async def _re_execute(self, decision: Decision) -> None:
        """Execute a decision via session handle or internal control."""
        p = decision.payload
        match decision.command:
            case Command.INTERRUPT:
                await self._handle.interrupt()
                await asyncio.sleep(0.15)
            case Command.RESPOND:
                await self._handle.respond(instructions=p.get("instructions", ""))
            case Command.SAY:
                await self._handle.say(text=p.get("text", ""))
            case Command.SET_PROMPT:
                await self._handle.set_prompt(prompt=p.get("prompt", ""))
            case Command.ENRICH:
                await self._handle.enrich(data=p.get("data", {}))
            case Command.SWAP:
                await self._handle.swap(agent_id=p.get("agent_id", ""))
            case Command.HOLD:
                self._paused = True
                self._state.session = SessionMode.PAUSED
                self._log("engine", "paused")
            case Command.RESUME:
                self._paused = False
                self._state.session = SessionMode.ACTIVE
                self._log("engine", "resumed")
