"""Tests for aeda.state — AgentSnapshot, TaskRecord, ReactiveState."""

from __future__ import annotations

import time

import pytest

from aeda.models import AgentMode, SessionMode, TaskState
from aeda.state import AgentSnapshot, ReactiveState, TaskRecord


##### AGENT SNAPSHOT #####


async def test_agent_snapshot_defaults() -> None:
    snap = AgentSnapshot()
    assert snap.name == ""
    assert snap.mode == AgentMode.IDLE
    assert snap.turns == 0
    assert snap.tool_calls == 0
    assert snap.errors == 0


async def test_agent_snapshot_active_for() -> None:
    snap = AgentSnapshot(entered_at=time.monotonic() - 5.0)
    assert snap.active_for >= 4.9


async def test_agent_snapshot_silent_for_zero_when_no_response() -> None:
    assert AgentSnapshot().silent_for == 0.0


async def test_agent_snapshot_silent_for() -> None:
    snap = AgentSnapshot(responded_at=time.monotonic() - 3.0)
    assert snap.silent_for >= 2.9


##### TASK RECORD #####


async def test_task_record_is_frozen() -> None:
    record = TaskRecord(task_id="t1", name="search")
    with pytest.raises(AttributeError):
        record.name = "other"  # type: ignore[misc]


async def test_task_record_default_state() -> None:
    assert TaskRecord(task_id="t1", name="search").state == TaskState.PENDING


##### REACTIVE STATE — TIMING #####


async def test_state_session_duration(state: ReactiveState) -> None:
    assert state.session_duration >= 0.0


async def test_state_inactive_for(state: ReactiveState) -> None:
    assert state.inactive_for >= 0.0


async def test_state_since_last_turn_zero(state: ReactiveState) -> None:
    assert state.since_last_turn == 0.0


##### REACTIVE STATE — BUDGET #####


async def test_state_budget_ratio_zero_when_no_budget(state: ReactiveState) -> None:
    assert state.budget_ratio == 0.0
    assert state.has_budget


async def test_state_budget_ratio(state: ReactiveState) -> None:
    state.token_budget = 1000
    state.tokens_used = 750
    assert state.budget_ratio == 0.75
    assert state.budget_remaining == 250
    assert state.has_budget


async def test_state_budget_exhausted(state: ReactiveState) -> None:
    state.token_budget = 100
    state.tokens_used = 150
    assert state.budget_ratio == 1.5
    assert state.budget_remaining == 0
    assert not state.has_budget


async def test_register_tokens(state: ReactiveState) -> None:
    state.register_tokens(500)
    state.register_tokens(300)
    assert state.tokens_used == 800


##### REACTIVE STATE — TASKS #####


async def test_state_tasks_empty(state: ReactiveState) -> None:
    assert state.tasks_pending == 0
    assert state.tasks_running == 0
    assert not state.has_pending


async def test_state_tasks_mixed(state: ReactiveState) -> None:
    state.tasks["t1"] = TaskRecord(task_id="t1", name="a", state=TaskState.PENDING)
    state.tasks["t2"] = TaskRecord(task_id="t2", name="b", state=TaskState.RUNNING)
    state.tasks["t3"] = TaskRecord(task_id="t3", name="c", state=TaskState.COMPLETED)
    assert state.tasks_pending == 1
    assert state.tasks_running == 1
    assert state.has_pending


##### REACTIVE STATE — MUTATIONS #####


async def test_register_turn(state: ReactiveState) -> None:
    state.register_turn()
    assert state.turn_count == 1
    assert state.agent.turns == 1
    assert state.last_turn_at > 0


async def test_register_handoff(state: ReactiveState) -> None:
    state.agent.name = "agent_a"
    state.agent.turns = 5
    state.register_handoff("agent_b")
    assert state.previous_agent.name == "agent_a"
    assert state.previous_agent.turns == 5
    assert state.agent.name == "agent_b"
    assert state.agent.turns == 0
    assert state.handoff_count == 1


async def test_register_handoff_uses_agents_registry(state: ReactiveState) -> None:
    sentinel = object()
    state.agents["bot"] = sentinel
    state.register_handoff("bot")
    assert state.agent.ref is sentinel


async def test_register_activity(state: ReactiveState) -> None:
    old = state.last_activity_at
    state.register_activity()
    assert state.last_activity_at >= old
