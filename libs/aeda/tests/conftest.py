"""Root conftest — shared fixtures for all test suites."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from aeda.engine import Reactor
from aeda.models import Event, HookAction, HookResult, Signal
from aeda.state import ReactiveState


##### MOCK HANDLE #####


@dataclass
class ActionLog:
    """Records all actions executed by the reactor via SessionHandle."""

    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def last(self, method: str) -> dict[str, Any] | None:
        for name, payload in reversed(self.calls):
            if name == method:
                return payload
        return None

    @property
    def methods(self) -> list[str]:
        return [name for name, _ in self.calls]


class MockHandle:
    """Mock SessionHandle that records all calls."""

    def __init__(self, log: ActionLog) -> None:
        self._log = log

    async def respond(self, *, instructions: str) -> None:
        self._log.calls.append(("respond", {"instructions": instructions}))

    async def say(self, *, text: str) -> None:
        self._log.calls.append(("say", {"text": text}))

    async def interrupt(self) -> None:
        self._log.calls.append(("interrupt", {}))

    async def enrich(self, *, data: dict[str, Any]) -> None:
        self._log.calls.append(("enrich", {"data": data}))

    async def set_prompt(self, *, prompt: str) -> None:
        self._log.calls.append(("set_prompt", {"prompt": prompt}))

    async def swap(self, *, agent_id: str) -> None:
        self._log.calls.append(("swap", {"agent_id": agent_id}))


##### MOCK HOOKS #####


async def allow_hook(event: Event) -> HookResult:
    """Hook that always continues."""
    return HookResult(HookAction.CONTINUE)


async def block_hook(event: Event) -> HookResult:
    """Hook that always blocks."""
    return HookResult(HookAction.BLOCK, reason="blocked by test")


async def skip_hook(event: Event) -> HookResult:
    """Hook that skips remaining hooks."""
    return HookResult(HookAction.SKIP)


##### FIXTURES #####


@pytest.fixture()
def action_log() -> ActionLog:
    return ActionLog()


@pytest.fixture()
def mock_handle(action_log: ActionLog) -> MockHandle:
    return MockHandle(action_log)


@pytest.fixture()
def state() -> ReactiveState:
    return ReactiveState()


@pytest.fixture()
def reactor(mock_handle: MockHandle, state: ReactiveState) -> Reactor:
    """Reactor with no policies and long tick for testing."""
    return Reactor(mock_handle, state, policies=(), tick_interval=60.0)


##### HELPERS #####


async def run_reactor_briefly(
    reactor: Reactor,
    events: list[Event],
    *,
    settle_time: float = 0.05,
) -> None:
    """Start reactor, emit events, wait for processing, then stop."""
    task = asyncio.create_task(reactor.run())
    await asyncio.sleep(0.01)
    for event in events:
        reactor.emit(event)
    await asyncio.sleep(settle_time)
    reactor.stop()
    await task
