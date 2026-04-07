"""AEDA demo — double loop with real Gemini LLM.

The agent has a tool `deep_research` that launches a 20s background task.
While it runs, the user keeps chatting with the agent (real Gemini calls).
When the task completes, TaskCompletedPolicy fires: the result lands in
state.data and the NEXT agent.run() sees it in the system prompt.

Flow:
  1. User asks agent to research something → agent calls deep_research tool
  2. Tool returns immediately ("launched"), but 20s task is running in background
  3. User asks 3 more questions → Gemini responds normally (no blocking)
  4. At ~20s the task completes → INTERRUPT + callback log
  5. User asks final question → agent's system prompt now includes the research
     result, so Gemini naturally weaves it into the answer

Requires GEMINI_API_KEY or GOOGLE_API_KEY.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Hooks

from aeda.engine import Reactor
from aeda.models import Event, Signal
from aeda.policies import TaskCompletedPolicy
from aeda.state import ReactiveState

##### COLORS #####

_RST = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[91m"
_BLUE = "\033[34m"
_MAGENTA = "\033[35m"
_GRAY = "\033[90m"
_WHITE = "\033[97m"

_TAG_COLORS: dict[str, str] = {
    "engine": _MAGENTA,
    "event": _DIM,
    "hook": _RED,
    "policy": _YELLOW,
    "tick": _GRAY,
}

_T0 = 0.0


def _ts() -> str:
    return f"{_GRAY}{time.monotonic() - _T0:5.1f}s{_RST}"


def _reactor_log(tag: str, msg: str) -> None:
    color = _TAG_COLORS.get(tag, _DIM)
    print(f"  {_ts()} {color}[{tag.upper()}]{_RST} {msg}")


def _user(text: str) -> None:
    print(f"\n  {_ts()} {_WHITE}{_BOLD}[USER]{_RST} {text}")


def _agent_says(text: str) -> None:
    print(f"  {_ts()} {_CYAN}{_BOLD}[AGENT]{_RST} {text}")


def _system(text: str) -> None:
    print(f"\n  {_ts()} {_MAGENTA}{_BOLD}[SYSTEM]{_RST} {text}")


##### HANDLE #####


class ConsoleHandle:
    async def respond(self, *, instructions: str) -> None:
        _system(f"{_GREEN}(callback) {instructions[:100]}{_RST}")

    async def say(self, *, text: str) -> None:
        _system(text)

    async def interrupt(self) -> None:
        _system(f"{_RED}--- INTERRUPT: background result incoming ---{_RST}")

    async def enrich(self, *, data: dict[str, Any]) -> None:
        pass

    async def set_prompt(self, *, prompt: str) -> None:
        pass

    async def swap(self, *, agent_id: str) -> None:
        pass


##### HEAVY BACKGROUND TASK #####


async def heavy_market_research(topic: str) -> dict[str, str]:
    """Simulates a 20-second deep research task."""
    _system(f"{_YELLOW}heavy_market_research launched: '{topic}' (will take ~20s)...{_RST}")
    await asyncio.sleep(20.0)
    return {
        "research_topic": topic,
        "research_result": (
            f"Deep analysis of '{topic}' complete. Key findings: "
            "1) Event-driven agent architectures reduce latency by 73% vs synchronous orchestration. "
            "2) The top 3 frameworks adopting this pattern are LangGraph, CrewAI Flows, and custom asyncio reactors. "
            "3) Main bottleneck is tool execution time, which background task offloading solves entirely. "
            "Recommendation: adopt AEDA pattern for all multi-tool agent deployments."
        ),
    }


##### AEDA HOOKS FOR PYDANTIC-AI #####


def build_aeda_hooks(reactor: Reactor) -> Hooks:
    hooks: Hooks[None] = Hooks()

    @hooks.on.before_run
    async def _before_run(ctx: Any) -> None:
        reactor.emit(Event(signal=Signal.AGENT_STARTED, payload={"agent_id": "gemini"}))
        reactor.emit(Event(signal=Signal.TURN_STARTED))

    @hooks.on.after_run
    async def _after_run(ctx: Any, *, result: Any) -> Any:
        reactor.emit(Event(signal=Signal.TURN_ENDED))
        reactor.emit(Event(signal=Signal.AGENT_ENDED))
        return result

    @hooks.on.before_model_request
    async def _before_model(ctx: Any, request_context: Any) -> Any:
        reactor.emit(Event(signal=Signal.MODEL_REQUESTED))
        return request_context

    @hooks.on.after_model_request
    async def _after_model(ctx: Any, *, request_context: Any, response: Any) -> Any:
        reactor.emit(Event(signal=Signal.MODEL_RESPONDED, payload={"tokens_used": 200}))
        return response

    @hooks.on.before_tool_execute
    async def _before_tool(ctx: Any, *, call: Any, tool_def: Any, args: Any) -> Any:
        reactor.emit(Event(signal=Signal.TOOL_CALLED, payload={"tool": call.tool_name}))
        return args

    @hooks.on.after_tool_execute
    async def _after_tool(ctx: Any, *, call: Any, tool_def: Any, args: Any, result: Any) -> Any:
        reactor.emit(Event(signal=Signal.TOOL_COMPLETED, payload={"tool": call.tool_name}))
        return result

    return hooks


##### MAIN #####


async def main() -> None:
    global _T0
    _T0 = time.monotonic()

    # Reactor setup
    state = ReactiveState()
    handle = ConsoleHandle()
    policies = (TaskCompletedPolicy(cooldown=0.0),)
    reactor = Reactor(handle, state, policies, tick_interval=5.0, log=_reactor_log)

    # Track events
    event_log: list[str] = []

    async def track(event: Event) -> None:
        event_log.append(event.signal.value)

    reactor.bus.subscribe_all(track)

    # Start reactor
    reactor_task = asyncio.create_task(reactor.run())
    await asyncio.sleep(0.05)

    # Build agent with AEDA hooks
    aeda_hooks = build_aeda_hooks(reactor)

    def build_agent(extra_instructions: str = "") -> Agent[None, str]:
        base = (
            "You are a helpful, concise assistant. "
            "Answer in 1-2 sentences max. Be direct."
        )
        if extra_instructions:
            base += f"\n\nIMPORTANT CONTEXT:\n{extra_instructions}"
        return Agent(
            "google-gla:gemini-2.0-flash",
            instructions=base,
            capabilities=[aeda_hooks],
        )

    agent = build_agent()

    # Banner
    print(f"\n{'=' * 70}")
    print(f"  AEDA Demo — Double Loop with Gemini 2.0 Flash")
    print(f"  Background task: 20s | Conversation continues in parallel")
    print(f"{'=' * 70}")

    # ── Turn 1: user asks for research → launch background task ────
    _user("I need a deep market analysis on event-driven architectures for AI agents.")

    _agent_says("Launching deep research now — this will take about 20 seconds. "
                "I'll deliver the results when ready. Keep chatting!")
    await reactor.tasks.submit(
        "market_research",
        heavy_market_research,
        topic="event-driven architectures for AI agents",
    )

    await asyncio.sleep(1.0)

    # ── Turns 2-5: normal conversation while task runs ─────────────
    questions = [
        "What's the population of Tokyo?",
        "Explain the CAP theorem in one sentence.",
        "Who invented Python?",
        "What year did Go 1.0 release?",
    ]

    for i, question in enumerate(questions, start=2):
        _user(question)
        result = await agent.run(question)
        _agent_says(result.output)
        elapsed = time.monotonic() - _T0
        _system(f"{_DIM}(turn {i} done at {elapsed:.1f}s — background task still running...){_RST}")
        await asyncio.sleep(2.0)

    # ── Wait for background task to complete ───────────────────────
    remaining = 20.0 - (time.monotonic() - _T0)
    if remaining > 0:
        _system(f"Waiting {remaining:.0f}s for background task to complete...")
        await asyncio.sleep(remaining + 1.0)
    else:
        await asyncio.sleep(1.0)

    # ── Turn 6: agent now has research in state.data ───────────────
    research_result = state.data.get("research_result", "")
    _system(f"Research result now in state.data ({len(research_result)} chars)")

    # Rebuild agent with research injected into system prompt
    agent_with_context = build_agent(
        f"A background research task has just completed with these findings:\n{research_result}\n"
        "Use these findings to answer the user's next question."
    )

    _user("Now that the research is done — what's the main takeaway for our team?")
    result = await agent_with_context.run(
        "Based on the research results you just received about event-driven architectures "
        "for AI agents, what's the single most important takeaway for our engineering team?"
    )
    _agent_says(result.output)

    # ── Cleanup ────────────────────────────────────────────────────
    reactor.stop()
    await reactor_task

    # Summary
    elapsed = time.monotonic() - _T0
    print(f"\n{'=' * 70}")
    print(f"  {_GREEN}{_BOLD}SUMMARY{_RST}")
    print(f"{'=' * 70}")
    print(f"  Wall time:     {elapsed:.1f}s")
    print(f"  Turns:         {state.turn_count}")
    print(f"  Tokens:        {state.tokens_used}")
    print(f"  Data keys:     {list(state.data.keys())}")
    print(f"  Events:        {len(event_log)} total")
    print(f"  Task result:   {state.data.get('research_result', 'N/A')[:100]}...")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    asyncio.run(main())
