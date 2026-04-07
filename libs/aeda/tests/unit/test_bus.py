"""Tests for aeda.bus — EventBus with hooks + subscribers."""

from __future__ import annotations

import asyncio

import pytest

from aeda.bus import EventBus
from aeda.models import Event, HookAction, HookResult, Signal
from tests.conftest import allow_hook, block_hook, skip_hook


##### HOOKS — FLOW CONTROL #####


async def test_hook_continue() -> None:
    bus = EventBus()
    bus.hook(Signal.AGENT_STARTED, allow_hook)
    result = await bus.dispatch(Event(signal=Signal.AGENT_STARTED))
    assert result.action == HookAction.CONTINUE


async def test_hook_block() -> None:
    bus = EventBus()
    bus.hook(Signal.TOOL_CALLED, block_hook)
    result = await bus.dispatch(Event(signal=Signal.TOOL_CALLED))
    assert result.action == HookAction.BLOCK
    assert result.reason == "blocked by test"


async def test_hook_skip_stops_chain() -> None:
    bus = EventBus()
    reached: list[str] = []

    async def first_hook(event: Event) -> HookResult:
        reached.append("first")
        return HookResult(HookAction.SKIP)

    async def second_hook(event: Event) -> HookResult:
        reached.append("second")
        return HookResult(HookAction.CONTINUE)

    bus.hook(Signal.TICK, first_hook, priority=10)
    bus.hook(Signal.TICK, second_hook, priority=20)
    result = await bus.dispatch(Event(signal=Signal.TICK))
    assert result.action == HookAction.CONTINUE
    assert reached == ["first"]


async def test_hook_priority_order() -> None:
    bus = EventBus()
    order: list[int] = []

    async def hook_a(event: Event) -> HookResult:
        order.append(1)
        return HookResult(HookAction.CONTINUE)

    async def hook_b(event: Event) -> HookResult:
        order.append(2)
        return HookResult(HookAction.CONTINUE)

    bus.hook(Signal.TICK, hook_b, priority=200)
    bus.hook(Signal.TICK, hook_a, priority=10)
    await bus.dispatch(Event(signal=Signal.TICK))
    assert order == [1, 2]


async def test_hook_block_stops_chain() -> None:
    bus = EventBus()
    reached: list[str] = []

    async def blocker(event: Event) -> HookResult:
        reached.append("blocker")
        return HookResult(HookAction.BLOCK, reason="security")

    async def after_blocker(event: Event) -> HookResult:
        reached.append("after")
        return HookResult(HookAction.CONTINUE)

    bus.hook(Signal.TOOL_CALLED, blocker, priority=10)
    bus.hook(Signal.TOOL_CALLED, after_blocker, priority=20)
    result = await bus.dispatch(Event(signal=Signal.TOOL_CALLED))
    assert result.action == HookAction.BLOCK
    assert reached == ["blocker"]


async def test_no_hooks_returns_continue() -> None:
    bus = EventBus()
    result = await bus.dispatch(Event(signal=Signal.TICK))
    assert result.action == HookAction.CONTINUE


##### SUBSCRIBERS #####


async def test_subscribe_and_notify() -> None:
    bus = EventBus()
    received: list[Event] = []

    async def handler(event: Event) -> None:
        received.append(event)

    bus.subscribe(Signal.AGENT_STARTED, handler)
    event = Event(signal=Signal.AGENT_STARTED)
    await bus.notify(event)
    assert len(received) == 1
    assert received[0] is event


async def test_wildcard_receives_all() -> None:
    bus = EventBus()
    received: list[Signal] = []

    async def handler(event: Event) -> None:
        received.append(event.signal)

    bus.subscribe_all(handler)
    await bus.notify(Event(signal=Signal.TICK))
    await bus.notify(Event(signal=Signal.AGENT_IDLE))
    assert received == [Signal.TICK, Signal.AGENT_IDLE]


async def test_no_subscribers_no_error() -> None:
    bus = EventBus()
    await bus.notify(Event(signal=Signal.TICK))


async def test_multiple_subscribers_same_signal() -> None:
    bus = EventBus()
    results: list[int] = []

    async def handler_a(event: Event) -> None:
        results.append(1)

    async def handler_b(event: Event) -> None:
        results.append(2)

    bus.subscribe(Signal.TOOL_COMPLETED, handler_a)
    bus.subscribe(Signal.TOOL_COMPLETED, handler_b)
    await bus.notify(Event(signal=Signal.TOOL_COMPLETED))
    assert sorted(results) == [1, 2]


##### STANDALONE MODE #####


async def test_standalone_run_with_hook_block() -> None:
    bus = EventBus()
    notified: list[Signal] = []

    async def observer(event: Event) -> None:
        notified.append(event.signal)

    bus.subscribe_all(observer)
    bus.hook(Signal.TOOL_CALLED, block_hook)

    bus.emit(Event(signal=Signal.TOOL_CALLED))
    bus.emit(Event(signal=Signal.AGENT_STARTED))

    task = asyncio.create_task(bus.run())
    await asyncio.sleep(0.05)
    bus.stop()
    task.cancel()

    # TOOL_CALLED was blocked → subscriber never saw it
    assert Signal.TOOL_CALLED not in notified
    assert Signal.AGENT_STARTED in notified
