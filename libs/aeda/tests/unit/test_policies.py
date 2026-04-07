"""Tests for aeda.policies — IdlePolicy, TaskCompletedPolicy, EscalationPolicy, BudgetPolicy."""

from __future__ import annotations

import time

import pytest

from aeda.models import Command, Event, Signal
from aeda.policies import BudgetPolicy, EscalationPolicy, IdlePolicy, TaskCompletedPolicy
from aeda.state import ReactiveState


##### IDLE POLICY #####


async def test_idle_triggers_on_timeout() -> None:
    policy = IdlePolicy(timeout=1.0, text="Hello?")
    state = ReactiveState()
    state.last_activity_at = time.monotonic() - 5.0
    decisions = policy.evaluate(state, Event(signal=Signal.TICK))
    assert len(decisions) == 1
    assert decisions[0].command == Command.SAY
    assert decisions[0].payload["text"] == "Hello?"


async def test_idle_no_trigger_if_recent() -> None:
    policy = IdlePolicy(timeout=10.0)
    state = ReactiveState()
    assert policy.evaluate(state, Event(signal=Signal.TICK)) == ()


async def test_idle_notifies_once() -> None:
    policy = IdlePolicy(timeout=0.0)
    state = ReactiveState()
    state.last_activity_at = time.monotonic() - 100.0
    d1 = policy.evaluate(state, Event(signal=Signal.TICK))
    d2 = policy.evaluate(state, Event(signal=Signal.TICK))
    assert len(d1) == 1
    assert d2 == ()


async def test_idle_resets_on_activity() -> None:
    policy = IdlePolicy(timeout=0.0)
    state = ReactiveState()
    state.last_activity_at = time.monotonic() - 100.0
    policy.evaluate(state, Event(signal=Signal.TICK))
    policy.evaluate(state, Event(signal=Signal.AGENT_RESPONDING))
    d = policy.evaluate(state, Event(signal=Signal.TICK))
    assert len(d) == 1


async def test_idle_resets_on_turn_ended() -> None:
    policy = IdlePolicy(timeout=0.0)
    state = ReactiveState()
    state.last_activity_at = time.monotonic() - 100.0
    policy.evaluate(state, Event(signal=Signal.TICK))
    policy.evaluate(state, Event(signal=Signal.TURN_ENDED))
    d = policy.evaluate(state, Event(signal=Signal.TICK))
    assert len(d) == 1


async def test_idle_ignores_non_tick() -> None:
    policy = IdlePolicy(timeout=0.0)
    state = ReactiveState()
    state.last_activity_at = time.monotonic() - 100.0
    assert policy.evaluate(state, Event(signal=Signal.AGENT_IDLE)) == ()


##### TASK COMPLETED POLICY #####


async def test_task_completed_interrupts_and_responds() -> None:
    policy = TaskCompletedPolicy(cooldown=0.0)
    state = ReactiveState()
    decisions = policy.evaluate(state, Event(
        signal=Signal.TASK_COMPLETED, payload={"message": "Done!"},
    ))
    assert len(decisions) == 2
    assert decisions[0].command == Command.INTERRUPT
    assert decisions[1].command == Command.RESPOND
    assert "Done!" in decisions[1].payload["instructions"]


async def test_task_completed_cooldown() -> None:
    policy = TaskCompletedPolicy(cooldown=10.0)
    state = ReactiveState()
    event = Event(signal=Signal.TASK_COMPLETED, payload={"message": "x"})
    d1 = policy.evaluate(state, event)
    d2 = policy.evaluate(state, event)
    assert len(d1) == 2
    assert d2 == ()


async def test_task_failed_responds() -> None:
    policy = TaskCompletedPolicy(cooldown=0.0)
    state = ReactiveState()
    decisions = policy.evaluate(state, Event(
        signal=Signal.TASK_FAILED, payload={"error": "timeout"},
    ))
    assert len(decisions) == 1
    assert decisions[0].command == Command.RESPOND
    assert "timeout" in decisions[0].payload["instructions"]


async def test_task_completed_ignores_other() -> None:
    assert TaskCompletedPolicy().evaluate(ReactiveState(), Event(signal=Signal.TICK)) == ()


##### ESCALATION POLICY #####


async def test_escalation_triggers() -> None:
    policy = EscalationPolicy(threshold=3, target_agent="supervisor")
    state = ReactiveState()
    state.turn_count = 5
    decisions = policy.evaluate(state, Event(signal=Signal.TURN_ENDED))
    assert len(decisions) == 2
    assert decisions[0].command == Command.SET_PROMPT
    assert decisions[1].command == Command.SWAP
    assert decisions[1].payload["agent_id"] == "supervisor"


async def test_escalation_below_threshold() -> None:
    policy = EscalationPolicy(threshold=10)
    state = ReactiveState()
    state.turn_count = 3
    assert policy.evaluate(state, Event(signal=Signal.TURN_ENDED)) == ()


async def test_escalation_once() -> None:
    policy = EscalationPolicy(threshold=1)
    state = ReactiveState()
    state.turn_count = 5
    d1 = policy.evaluate(state, Event(signal=Signal.TURN_ENDED))
    d2 = policy.evaluate(state, Event(signal=Signal.TURN_ENDED))
    assert len(d1) == 2
    assert d2 == ()


async def test_escalation_wrong_signal() -> None:
    policy = EscalationPolicy(threshold=1)
    state = ReactiveState()
    state.turn_count = 100
    assert policy.evaluate(state, Event(signal=Signal.TICK)) == ()


##### BUDGET POLICY #####


async def test_budget_warns_at_threshold() -> None:
    policy = BudgetPolicy(warn_ratio=0.9)
    state = ReactiveState()
    state.token_budget = 1000
    state.tokens_used = 950
    decisions = policy.evaluate(state, Event(signal=Signal.MODEL_RESPONDED))
    assert len(decisions) == 1
    assert decisions[0].command == Command.SET_PROMPT
    assert "95%" in decisions[0].payload["prompt"]


async def test_budget_holds_at_exhaustion() -> None:
    policy = BudgetPolicy(warn_ratio=0.9)
    state = ReactiveState()
    state.token_budget = 1000
    state.tokens_used = 1000
    decisions = policy.evaluate(state, Event(signal=Signal.MODEL_RESPONDED))
    assert len(decisions) == 2
    assert decisions[0].command == Command.SAY
    assert decisions[1].command == Command.HOLD


async def test_budget_no_trigger_without_budget() -> None:
    policy = BudgetPolicy()
    state = ReactiveState()
    assert policy.evaluate(state, Event(signal=Signal.MODEL_RESPONDED)) == ()


async def test_budget_ignores_wrong_signal() -> None:
    policy = BudgetPolicy()
    state = ReactiveState()
    state.token_budget = 100
    state.tokens_used = 100
    assert policy.evaluate(state, Event(signal=Signal.AGENT_IDLE)) == ()


async def test_budget_exhaustion_fires_once() -> None:
    policy = BudgetPolicy()
    state = ReactiveState()
    state.token_budget = 100
    state.tokens_used = 200
    d1 = policy.evaluate(state, Event(signal=Signal.MODEL_RESPONDED))
    d2 = policy.evaluate(state, Event(signal=Signal.MODEL_RESPONDED))
    assert len(d1) == 2
    assert d2 == ()
