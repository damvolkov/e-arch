"""AEDA — background task management.

Fire-and-forget task execution with lifecycle events emitted back to the reactor.

Taxonomy (N1):
  Entities:  TaskRegistry, TaskExecutor
  Verbs:     submit, resolve (public) | _te_run (private)
  Relation:  TaskExecutor → [emit, TaskRegistry]
             TaskRegistry → [TaskRecord]
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from aeda.models import Event, Signal, TaskState
from aeda.state import TaskRecord


type TaskFn = Callable[..., Awaitable[Any]]


##### REGISTRY #####


class TaskRegistry:
    """Tracks in-flight background tasks. O(1) register/resolve/get."""

    __slots__ = ("_tasks",)

    def __init__(self) -> None:
        self._tasks: dict[str, TaskRecord] = {}

    def register(self, record: TaskRecord) -> None:
        self._tasks[record.task_id] = record

    def resolve(self, task_id: str) -> TaskRecord | None:
        """Pop and return completed task. O(1)."""
        return self._tasks.pop(task_id, None)

    def get(self, task_id: str) -> TaskRecord | None:
        return self._tasks.get(task_id)

    @property
    def pending(self) -> dict[str, TaskRecord]:
        return {
            k: v for k, v in self._tasks.items()
            if v.state in {TaskState.PENDING, TaskState.RUNNING}
        }

    @property
    def count(self) -> int:
        return len(self._tasks)


##### EXECUTOR #####


class TaskExecutor:
    """Fire-and-forget: wraps async fn → background task → emits lifecycle events.

    Events flow back into the reactor via the injected `emit` callable,
    triggering state mutations and policy evaluation.
    """

    __slots__ = ("_emit", "_registry")

    def __init__(
        self,
        emit: Callable[[Event], None],
        registry: TaskRegistry,
    ) -> None:
        self._emit = emit
        self._registry = registry

    @property
    def registry(self) -> TaskRegistry:
        return self._registry

    async def submit(self, name: str, fn: TaskFn, **kwargs: Any) -> str:
        """Submit task for background execution. Returns task_id immediately."""
        task_id = f"{name}:{uuid.uuid4().hex[:8]}"
        record = TaskRecord(task_id=task_id, name=name)
        self._registry.register(record)
        self._emit(Event(
            signal=Signal.TASK_SUBMITTED,
            payload={"task_id": task_id, "name": name},
        ))
        asyncio.create_task(self._te_run(fn, task_id, **kwargs))
        return task_id

    async def _te_run(self, fn: TaskFn, task_id: str, **kwargs: Any) -> None:
        """Execute fn and emit result event back to reactor."""
        try:
            result = await fn(**kwargs)
            self._emit(Event(
                signal=Signal.TASK_COMPLETED,
                payload={"task_id": task_id, "result": result},
            ))
        except Exception as exc:
            self._emit(Event(
                signal=Signal.TASK_FAILED,
                payload={"task_id": task_id, "error": str(exc)},
            ))
