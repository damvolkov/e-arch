"""Tests for aeda.engine — Reactor (core state machine)."""

from __future__ import annotations

import asyncio

import pytest

from aeda.engine import Reactor
from aeda.models import (
    AgentMode,
    Command,
    Event,
    HookAction,
    HookResult,
    SessionMode,
    Signal,
)
from aeda.policies import BudgetPolicy, IdlePolicy, TaskCompletedPolicy
from aeda.state import ReactiveState
from tests.conftest import ActionLog, MockHandle, run_reactor_briefly


##### LIFECYCLE #####


async def test_reactor_starts_and_stops(reactor: Reactor, state: ReactiveState) -> None:
    task = asyncio.create_task(reactor.run())
    await asyncio.sleep(0.05)
    assert reactor.running
    assert state.session == SessionMode.ACTIVE
    reactor.stop()
    await task
    assert not reactor.running
    assert state.session == SessionMode.ENDING


##### STATE MUTATIONS #####


async def test_apply_agent_started(reactor: Reactor, state: ReactiveState) -> None:
    await run_reactor_briefly(reactor, [
        Event(signal=Signal.AGENT_STARTED, payload={"agent_id": "helper"}),
    ])
    assert state.agent.name == "helper"
    assert state.agent.mode == AgentMode.IDLE


async def test_apply_agent_thinking(reactor: Reactor, state: ReactiveState) -> None:
    await run_reactor_briefly(reactor, [Event(signal=Signal.AGENT_THINKING)])
    assert state.agent.mode == AgentMode.THINKING


async def test_apply_agent_responding(reactor: Reactor, state: ReactiveState) -> None:
    await run_reactor_briefly(reactor, [Event(signal=Signal.AGENT_RESPONDING)])
    assert state.agent.mode == AgentMode.RESPONDING


async def test_apply_agent_error(reactor: Reactor, state: ReactiveState) -> None:
    await run_reactor_briefly(reactor, [
        Event(signal=Signal.AGENT_ERROR),
        Event(signal=Signal.AGENT_ERROR),
    ])
    assert state.agent.errors == 2
    assert state.agent.mode == AgentMode.IDLE


async def test_apply_tool_called(reactor: Reactor, state: ReactiveState) -> None:
    await run_reactor_briefly(reactor, [
        Event(signal=Signal.TOOL_CALLED, payload={"tool": "search"}),
    ])
    assert state.agent.mode == AgentMode.WAITING
    assert state.agent.tool_calls == 1


async def test_apply_tool_completed(reactor: Reactor, state: ReactiveState) -> None:
    await run_reactor_briefly(reactor, [
        Event(signal=Signal.TOOL_CALLED),
        Event(signal=Signal.TOOL_COMPLETED),
    ])
    assert state.agent.mode == AgentMode.THINKING


async def test_apply_model_responded_tracks_tokens(reactor: Reactor, state: ReactiveState) -> None:
    await run_reactor_briefly(reactor, [
        Event(signal=Signal.MODEL_RESPONDED, payload={"tokens_used": 500}),
        Event(signal=Signal.MODEL_RESPONDED, payload={"tokens_used": 300}),
    ])
    assert state.tokens_used == 800


async def test_apply_model_error(reactor: Reactor, state: ReactiveState) -> None:
    await run_reactor_briefly(reactor, [Event(signal=Signal.MODEL_ERROR)])
    assert state.agent.errors == 1


async def test_apply_turn_ended(reactor: Reactor, state: ReactiveState) -> None:
    await run_reactor_briefly(reactor, [Event(signal=Signal.TURN_ENDED)])
    assert state.turn_count == 1
    assert state.agent.turns == 1


async def test_apply_handoff(reactor: Reactor, state: ReactiveState) -> None:
    state.agent.name = "agent_a"
    state.agent.turns = 3
    await run_reactor_briefly(reactor, [
        Event(signal=Signal.HANDOFF_OUT, payload={"agent_id": "agent_b"}),
    ])
    assert state.agent.name == "agent_b"
    assert state.previous_agent.name == "agent_a"
    assert state.handoff_count == 1


async def test_apply_task_completed_merges_data(reactor: Reactor, state: ReactiveState) -> None:
    await run_reactor_briefly(reactor, [
        Event(signal=Signal.TASK_COMPLETED, payload={"task_id": "t1", "result": {"weather": "sunny"}}),
    ])
    assert state.data["weather"] == "sunny"


async def test_apply_session_ending(reactor: Reactor, state: ReactiveState) -> None:
    await run_reactor_briefly(reactor, [Event(signal=Signal.SESSION_ENDING)])
    assert state.session == SessionMode.ENDING


##### HOOK INTEGRATION — BLOCK GATE #####


async def test_hook_blocks_event_processing() -> None:
    log = ActionLog()
    handle = MockHandle(log)
    state = ReactiveState()
    policy = TaskCompletedPolicy(cooldown=0.0)
    reactor = Reactor(handle, state, (policy,), tick_interval=60.0)

    # Register a hook that blocks TASK_COMPLETED
    async def block_tasks(event: Event) -> HookResult:
        return HookResult(HookAction.BLOCK, reason="test block")

    reactor.bus.hook(Signal.TASK_COMPLETED, block_tasks)

    await run_reactor_briefly(reactor, [
        Event(signal=Signal.TASK_COMPLETED, payload={"message": "should not fire"}),
    ], settle_time=0.1)

    # Policy should NOT have fired because hook blocked the event
    assert "respond" not in log.methods
    assert "interrupt" not in log.methods


##### POLICY INTEGRATION #####


async def test_task_completed_policy_fires() -> None:
    log = ActionLog()
    handle = MockHandle(log)
    state = ReactiveState()
    policy = TaskCompletedPolicy(cooldown=0.0)
    reactor = Reactor(handle, state, (policy,), tick_interval=60.0)

    await run_reactor_briefly(reactor, [
        Event(signal=Signal.TASK_COMPLETED, payload={"message": "Weather is sunny"}),
    ], settle_time=0.3)

    assert "interrupt" in log.methods
    assert "respond" in log.methods


async def test_idle_policy_fires_on_timeout() -> None:
    log = ActionLog()
    handle = MockHandle(log)
    state = ReactiveState()
    state.last_activity_at -= 100.0
    policy = IdlePolicy(timeout=1.0, text="Still there?")
    reactor = Reactor(handle, state, (policy,), tick_interval=0.05)

    task = asyncio.create_task(reactor.run())
    await asyncio.sleep(0.2)
    reactor.stop()
    await task

    assert "say" in log.methods
    say_call = log.last("say")
    assert say_call is not None
    assert say_call["text"] == "Still there?"


async def test_budget_policy_warns_then_holds() -> None:
    log = ActionLog()
    handle = MockHandle(log)
    state = ReactiveState()
    state.token_budget = 1000
    state.tokens_used = 950
    policy = BudgetPolicy(warn_ratio=0.9)
    reactor = Reactor(handle, state, (policy,), tick_interval=0.05)

    task = asyncio.create_task(reactor.run())
    await asyncio.sleep(0.15)

    # Should have warned (950/1000 = 95% > 90%)
    assert "set_prompt" in log.methods

    # Push over budget
    reactor.emit(Event(signal=Signal.MODEL_RESPONDED, payload={"tokens_used": 100}))
    await asyncio.sleep(0.15)
    reactor.stop()
    await task

    # Should have exhausted
    assert "say" in log.methods
    assert state.session == SessionMode.ENDING  # reactor.stop() sets ENDING


##### BUS SUBSCRIBERS #####


async def test_subscribers_receive_events(reactor: Reactor) -> None:
    received: list[Signal] = []

    async def observer(event: Event) -> None:
        received.append(event.signal)

    reactor.bus.subscribe(Signal.AGENT_STARTED, observer)
    await run_reactor_briefly(reactor, [
        Event(signal=Signal.AGENT_STARTED, payload={"agent_id": "test"}),
    ])
    assert Signal.AGENT_STARTED in received


async def test_wildcard_subscriber(reactor: Reactor) -> None:
    received: list[Signal] = []

    async def observer(event: Event) -> None:
        received.append(event.signal)

    reactor.bus.subscribe_all(observer)
    await run_reactor_briefly(reactor, [
        Event(signal=Signal.AGENT_THINKING),
        Event(signal=Signal.AGENT_RESPONDING),
    ])
    assert Signal.AGENT_THINKING in received
    assert Signal.AGENT_RESPONDING in received


async def test_blocked_event_not_seen_by_subscribers(reactor: Reactor) -> None:
    received: list[Signal] = []

    async def observer(event: Event) -> None:
        received.append(event.signal)

    reactor.bus.subscribe_all(observer)
    reactor.bus.hook(Signal.TOOL_CALLED, lambda e: asyncio.coroutine(
        lambda: HookResult(HookAction.BLOCK, reason="test"),
    )())

    # Use a proper async hook
    async def block_tool(event: Event) -> HookResult:
        return HookResult(HookAction.BLOCK, reason="test")

    reactor.bus._hooks.clear()
    reactor.bus.hook(Signal.TOOL_CALLED, block_tool)

    await run_reactor_briefly(reactor, [
        Event(signal=Signal.TOOL_CALLED),
        Event(signal=Signal.AGENT_IDLE),
    ])
    assert Signal.TOOL_CALLED not in received
    assert Signal.AGENT_IDLE in received


##### TASK EXECUTOR INTEGRATION #####


async def test_task_executor_submit() -> None:
    log = ActionLog()
    handle = MockHandle(log)
    state = ReactiveState()
    policy = TaskCompletedPolicy(cooldown=0.0)
    reactor = Reactor(handle, state, (policy,), tick_interval=60.0)

    task = asyncio.create_task(reactor.run())
    await asyncio.sleep(0.05)

    async def bg_work() -> dict[str, str]:
        await asyncio.sleep(0.05)
        return {"answer": "42"}

    task_id = await reactor.tasks.submit("compute", bg_work)
    assert task_id.startswith("compute:")

    await asyncio.sleep(0.5)
    reactor.stop()
    await task

    assert "interrupt" in log.methods
    assert "respond" in log.methods
