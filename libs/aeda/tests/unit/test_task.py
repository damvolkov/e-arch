"""Tests for aeda.task — TaskRegistry, TaskExecutor."""

from __future__ import annotations

import asyncio

import pytest

from aeda.models import Event, Signal, TaskState
from aeda.state import TaskRecord
from aeda.task import TaskExecutor, TaskRegistry


##### TASK REGISTRY #####


async def test_registry_register_and_get() -> None:
    reg = TaskRegistry()
    record = TaskRecord(task_id="t1", name="search")
    reg.register(record)
    assert reg.get("t1") is record
    assert reg.count == 1


async def test_registry_resolve_removes() -> None:
    reg = TaskRegistry()
    reg.register(TaskRecord(task_id="t1", name="search"))
    resolved = reg.resolve("t1")
    assert resolved is not None
    assert resolved.task_id == "t1"
    assert reg.count == 0
    assert reg.resolve("t1") is None


async def test_registry_pending_filters() -> None:
    reg = TaskRegistry()
    reg.register(TaskRecord(task_id="t1", name="a", state=TaskState.PENDING))
    reg.register(TaskRecord(task_id="t2", name="b", state=TaskState.RUNNING))
    reg.register(TaskRecord(task_id="t3", name="c", state=TaskState.COMPLETED))
    pending = reg.pending
    assert set(pending.keys()) == {"t1", "t2"}


##### TASK EXECUTOR #####


async def test_executor_submit_emits_events() -> None:
    emitted: list[Event] = []

    def capture_emit(event: Event) -> None:
        emitted.append(event)

    reg = TaskRegistry()
    executor = TaskExecutor(capture_emit, reg)

    async def quick_fn() -> dict[str, str]:
        return {"result": "ok"}

    task_id = await executor.submit("quick", quick_fn)
    assert task_id.startswith("quick:")
    assert reg.count == 1

    # Wait for background task to complete
    await asyncio.sleep(0.1)

    signals = [e.signal for e in emitted]
    assert Signal.TASK_SUBMITTED in signals
    assert Signal.TASK_COMPLETED in signals


async def test_executor_failing_task_emits_failed() -> None:
    emitted: list[Event] = []

    def capture_emit(event: Event) -> None:
        emitted.append(event)

    reg = TaskRegistry()
    executor = TaskExecutor(capture_emit, reg)

    async def bad_fn() -> None:
        msg = "boom"
        raise ValueError(msg)

    await executor.submit("bad", bad_fn)
    await asyncio.sleep(0.1)

    signals = [e.signal for e in emitted]
    assert Signal.TASK_SUBMITTED in signals
    assert Signal.TASK_FAILED in signals

    failed_event = next(e for e in emitted if e.signal == Signal.TASK_FAILED)
    assert "boom" in failed_event.payload["error"]
