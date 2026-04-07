"""Tests for aeda.models — enums, Event, Decision, HookResult."""

from __future__ import annotations

import time

import pytest

from aeda.models import (
    AgentMode,
    Command,
    Decision,
    Event,
    HookAction,
    HookResult,
    SessionMode,
    Signal,
    TaskState,
)


##### SIGNAL #####


async def test_signal_values_are_strings() -> None:
    assert isinstance(Signal.AGENT_STARTED, str)
    assert Signal.AGENT_STARTED == "agent_started"


async def test_signal_count() -> None:
    assert len(Signal) == 28


##### COMMAND #####


async def test_command_count() -> None:
    assert len(Command) == 8


##### HOOK ACTION #####


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        (HookAction.CONTINUE, "continue"),
        (HookAction.MUTATE, "mutate"),
        (HookAction.BLOCK, "block"),
        (HookAction.SKIP, "skip"),
    ],
    ids=["continue", "mutate", "block", "skip"],
)
async def test_hook_action_values(action: HookAction, expected: str) -> None:
    assert action == expected


##### MODE ENUMS #####


@pytest.mark.parametrize(
    ("enum_cls", "members"),
    [
        (AgentMode, {"IDLE", "THINKING", "RESPONDING", "WAITING"}),
        (SessionMode, {"STARTING", "ACTIVE", "PAUSED", "ENDING"}),
        (TaskState, {"PENDING", "RUNNING", "COMPLETED", "FAILED", "TIMEOUT", "CANCELLED"}),
    ],
    ids=["agent-mode", "session-mode", "task-state"],
)
async def test_mode_members(enum_cls: type, members: set[str]) -> None:
    assert {m.name for m in enum_cls} == members


##### EVENT #####


async def test_event_is_frozen() -> None:
    event = Event(signal=Signal.TICK)
    with pytest.raises(AttributeError):
        event.signal = Signal.AGENT_IDLE  # type: ignore[misc]


async def test_event_default_timestamp() -> None:
    before = time.monotonic()
    event = Event(signal=Signal.TICK)
    after = time.monotonic()
    assert before <= event.timestamp <= after


async def test_event_payload_default_empty() -> None:
    assert Event(signal=Signal.AGENT_STARTED).payload == {}


async def test_event_with_payload() -> None:
    event = Event(signal=Signal.TOOL_CALLED, payload={"tool": "search"})
    assert event.payload["tool"] == "search"


##### DECISION #####


async def test_decision_is_frozen() -> None:
    decision = Decision(command=Command.RESPOND)
    with pytest.raises(AttributeError):
        decision.command = Command.SAY  # type: ignore[misc]


##### HOOK RESULT #####


async def test_hook_result_defaults() -> None:
    result = HookResult()
    assert result.action == HookAction.CONTINUE
    assert result.payload is None
    assert result.reason is None


async def test_hook_result_block_with_reason() -> None:
    result = HookResult(HookAction.BLOCK, reason="dangerous command")
    assert result.action == HookAction.BLOCK
    assert result.reason == "dangerous command"


async def test_hook_result_is_frozen() -> None:
    result = HookResult()
    with pytest.raises(AttributeError):
        result.action = HookAction.BLOCK  # type: ignore[misc]
