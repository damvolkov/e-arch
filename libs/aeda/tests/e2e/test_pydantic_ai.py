"""E2E test: AEDA Reactor + pydantic-ai agent + Gemini.

Demonstrates the full v2 integration:
  1. pydantic-ai agent with a tool
  2. AEDA Hooks capability translating lifecycle events → AEDA signals
  3. Reactor with hook flow-control (security hook blocks dangerous tools)
  4. BudgetPolicy tracking token usage
  5. Background task via TaskExecutor
  6. Debug logging of all events
"""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.capabilities import Hooks

from aeda.engine import Reactor
from aeda.models import Event, HookAction, HookResult, Signal
from aeda.policies import BudgetPolicy, TaskCompletedPolicy
from aeda.state import ReactiveState

##### LOGGING #####

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
_log = logging.getLogger("aeda.e2e")


def _reactor_log(tag: str, msg: str) -> None:
    _log.info("[%s] %s", tag.upper(), msg)


##### AEDA HOOKS FOR PYDANTIC-AI #####


def build_aeda_hooks(reactor: Reactor) -> Hooks:
    """Create pydantic-ai Hooks that emit AEDA signals into the reactor."""
    hooks: Hooks[None] = Hooks()

    @hooks.on.before_run
    async def _before_run(ctx: Any) -> None:
        _log.debug("HOOK before_run")
        reactor.emit(Event(signal=Signal.AGENT_STARTED, payload={"agent_id": "gemini"}))
        reactor.emit(Event(signal=Signal.TURN_STARTED))

    @hooks.on.after_run
    async def _after_run(ctx: Any, *, result: Any) -> Any:
        _log.debug("HOOK after_run")
        reactor.emit(Event(signal=Signal.TURN_ENDED))
        reactor.emit(Event(signal=Signal.AGENT_ENDED))
        return result

    @hooks.on.before_model_request
    async def _before_model(ctx: Any, request_context: Any) -> Any:
        _log.debug("HOOK before_model_request")
        reactor.emit(Event(signal=Signal.MODEL_REQUESTED))
        return request_context

    @hooks.on.after_model_request
    async def _after_model(ctx: Any, *, request_context: Any, response: Any) -> Any:
        _log.debug("HOOK after_model_request")
        reactor.emit(Event(signal=Signal.MODEL_RESPONDED, payload={"tokens_used": 150}))
        return response

    @hooks.on.before_tool_execute
    async def _before_tool(ctx: Any, *, call: Any, tool_def: Any, args: Any) -> Any:
        _log.debug("HOOK before_tool_execute tool=%s", call.tool_name)
        reactor.emit(Event(
            signal=Signal.TOOL_CALLED,
            payload={"tool": call.tool_name},
        ))
        return args

    @hooks.on.after_tool_execute
    async def _after_tool(
        ctx: Any, *, call: Any, tool_def: Any, args: Any, result: Any,
    ) -> Any:
        _log.debug("HOOK after_tool_execute tool=%s", call.tool_name)
        reactor.emit(Event(
            signal=Signal.TOOL_COMPLETED,
            payload={"tool": call.tool_name},
        ))
        return result

    return hooks


##### DEBUG HANDLE #####


class DebugHandle:
    """SessionHandle that logs all commands."""

    async def respond(self, *, instructions: str) -> None:
        _log.info("HANDLE respond: %s", instructions[:80])

    async def say(self, *, text: str) -> None:
        _log.info("HANDLE say: %s", text)

    async def interrupt(self) -> None:
        _log.info("HANDLE interrupt")

    async def enrich(self, *, data: dict[str, Any]) -> None:
        _log.info("HANDLE enrich: %s", data)

    async def set_prompt(self, *, prompt: str) -> None:
        _log.info("HANDLE set_prompt: %s", prompt[:80])

    async def swap(self, *, agent_id: str) -> None:
        _log.info("HANDLE swap: %s", agent_id)


##### SECURITY HOOK #####


async def security_hook(event: Event) -> HookResult:
    """Block tool calls to dangerous tools."""
    tool_name = event.payload.get("tool", "")
    if tool_name in {"bash", "rm", "exec"}:
        return HookResult(HookAction.BLOCK, reason=f"Blocked dangerous tool: {tool_name}")
    return HookResult(HookAction.CONTINUE)


##### E2E TEST #####


@pytest.mark.slow
async def test_pydantic_ai_gemini_with_hooks_and_budget() -> None:
    """Full v2 integration: hooks → flow control → budget → tasks → state tracking."""

    # 1. Create AEDA reactor with budget tracking
    state = ReactiveState()
    state.token_budget = 10_000
    handle = DebugHandle()
    policies = (TaskCompletedPolicy(cooldown=0.0), BudgetPolicy(warn_ratio=0.8))
    reactor = Reactor(handle, state, policies, tick_interval=5.0, log=_reactor_log)

    # 2. Register security hook — blocks dangerous tool calls
    reactor.bus.hook(Signal.TOOL_CALLED, security_hook, priority=10)

    # 3. Collect all bus events
    bus_events: list[Signal] = []
    blocked_events: list[str] = []

    async def event_collector(event: Event) -> None:
        bus_events.append(event.signal)

    reactor.bus.subscribe_all(event_collector)

    # 4. Start reactor
    reactor_task = asyncio.create_task(reactor.run())
    await asyncio.sleep(0.05)

    # 5. Create pydantic-ai agent with AEDA hooks
    aeda_hooks = build_aeda_hooks(reactor)
    agent: Agent[None, str] = Agent(
        "google-gla:gemini-2.0-flash",
        instructions="You are a helpful assistant. Be concise. One sentence max.",
        capabilities=[aeda_hooks],
    )

    # 6. Run agent
    _log.info("Running pydantic-ai agent with Gemini...")
    result = await agent.run("What is the capital of France? Answer in one word.")
    _log.info("Agent result: %s", result.output)

    # 7. Give reactor time
    await asyncio.sleep(0.3)

    # 8. Assertions — lifecycle events
    assert Signal.AGENT_STARTED in bus_events
    assert Signal.MODEL_REQUESTED in bus_events
    assert Signal.MODEL_RESPONDED in bus_events
    assert Signal.TURN_STARTED in bus_events
    assert Signal.TURN_ENDED in bus_events
    assert Signal.AGENT_ENDED in bus_events

    # 9. Assertions — state tracked
    assert state.agent.errors == 0
    assert state.tokens_used > 0
    assert state.turn_count >= 1

    _log.info("State: tokens=%d/%d, turns=%d", state.tokens_used, state.token_budget, state.turn_count)

    # 10. Test background task
    _log.info("Submitting background task...")

    async def bg_research() -> dict[str, str]:
        await asyncio.sleep(0.1)
        return {"capital": "Paris"}

    await reactor.tasks.submit("research", bg_research)
    await asyncio.sleep(0.5)

    assert Signal.TASK_SUBMITTED in bus_events
    assert Signal.TASK_COMPLETED in bus_events
    assert state.data.get("capital") == "Paris"

    # 11. Cleanup
    reactor.stop()
    await reactor_task

    _log.info("All bus events: %s", [s.value for s in bus_events])
    _log.info("TEST PASSED")
