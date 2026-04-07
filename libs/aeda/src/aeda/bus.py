"""AEDA — event bus.

Typed async pub/sub with two layers:
  1. Hooks: priority-ordered, flow-controlling (CONTINUE/MUTATE/BLOCK/SKIP).
  2. Subscribers: observe-only, fire-and-forget.

Reactor calls dispatch() for hooks (gating), then notify() for subscribers.
Standalone mode: emit() + run() for self-contained event processing.

Taxonomy (N1):
  Entities:  EventBus
  Verbs:     hook, subscribe, subscribe_all, dispatch, notify, emit, run, stop
  Private:   _eb_run_hooks, _eb_notify
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from aeda.models import Event, HookAction, HookResult, Signal


type Subscriber = Callable[[Event], Awaitable[None]]
type HookHandler = Callable[[Event], Awaitable[HookResult]]


class EventBus:
    """Typed async event bus with hook flow-control and subscriber notification.

    Two registration methods:
      - hook(): flow-controlling handlers (priority-ordered, can BLOCK/MUTATE/SKIP)
      - subscribe() / subscribe_all(): observe-only handlers

    Two dispatch methods:
      - dispatch(event) → HookResult: run hooks only (for Reactor gating)
      - notify(event): notify subscribers only (after Reactor processing)
    """

    __slots__ = ("_hooks", "_subscribers", "_wildcard", "_queue", "_running")

    def __init__(self) -> None:
        self._hooks: dict[Signal, list[tuple[int, HookHandler]]] = {}
        self._subscribers: dict[Signal, list[Subscriber]] = {}
        self._wildcard: list[Subscriber] = []
        self._queue: asyncio.Queue[Event] = asyncio.Queue()
        self._running = False

    # ── Hook registration (flow-controlling) ──

    def hook(self, signal: Signal, handler: HookHandler, *, priority: int = 100) -> None:
        """Register hook handler. Lower priority = runs first. Can BLOCK/MUTATE/SKIP."""
        self._hooks.setdefault(signal, []).append((priority, handler))
        self._hooks[signal].sort(key=lambda x: x[0])

    # ── Subscriber registration (observe-only) ──

    def subscribe(self, signal: Signal, callback: Subscriber) -> None:
        """Register observer for a specific signal. O(1) append."""
        self._subscribers.setdefault(signal, []).append(callback)

    def subscribe_all(self, callback: Subscriber) -> None:
        """Register wildcard observer for ALL events. O(1) append."""
        self._wildcard.append(callback)

    # ── Dispatch: hooks with flow control ──

    async def dispatch(self, event: Event) -> HookResult:
        """Run hooks for event in priority order. Returns flow-control result.

        Hook chain semantics:
          CONTINUE → next hook
          MUTATE   → update event payload, next hook
          BLOCK    → abort immediately, return BLOCK result
          SKIP     → skip remaining hooks, return CONTINUE
        """
        return await self._eb_run_hooks(event)

    # ── Notify: subscribers (observe-only) ──

    async def notify(self, event: Event) -> None:
        """Notify subscribers. Fire-and-forget, no flow control."""
        await self._eb_notify(event)

    # ── Standalone mode ──

    def emit(self, event: Event) -> None:
        """Push event into internal queue. Non-blocking, sync-safe."""
        self._queue.put_nowait(event)

    async def run(self) -> None:
        """Standalone loop: consume from queue, dispatch hooks + notify subscribers."""
        self._running = True
        while self._running:
            event = await self._queue.get()
            if not self._running:
                break
            result = await self._eb_run_hooks(event)
            if result.action != HookAction.BLOCK:
                await self._eb_notify(event)
            self._queue.task_done()

    def stop(self) -> None:
        """Signal shutdown of standalone run() loop."""
        self._running = False

    # ── Private: _eb_ (EventBus) ──

    async def _eb_run_hooks(self, event: Event) -> HookResult:
        """Execute hooks in priority order with flow control. O(n) per event."""
        handlers = self._hooks.get(event.signal, [])
        if not handlers:
            return HookResult(HookAction.CONTINUE)

        for _, handler in handlers:
            result = await handler(event)
            match result.action:
                case HookAction.BLOCK:
                    return result
                case HookAction.MUTATE:
                    pass
                case HookAction.SKIP:
                    return HookResult(HookAction.CONTINUE)
                case HookAction.CONTINUE:
                    pass

        return HookResult(HookAction.CONTINUE)

    async def _eb_notify(self, event: Event) -> None:
        """Notify all matching subscribers + wildcard. Concurrent via gather."""
        handlers = self._subscribers.get(event.signal, []) + self._wildcard
        if handlers:
            await asyncio.gather(*(h(event) for h in handlers))
