"""AEDA demo — double loop in action.

Shows the core value proposition:
  1. User asks something that triggers a HEAVY background task (5s)
  2. Agent acknowledges and keeps conversing normally
  3. User asks two more things — agent responds instantly
  4. Background task completes mid-conversation → reactor fires INTERRUPT + delivers result
  5. Agent resumes normal conversation after delivering the result

This is impossible with synchronous tool execution.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from aeda.engine import Reactor
from aeda.models import Event, HookAction, HookResult, Signal
from aeda.policies import BudgetPolicy, TaskCompletedPolicy
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
    """Elapsed seconds since start."""
    return f"{_GRAY}{time.monotonic() - _T0:5.1f}s{_RST}"


def _reactor_log(tag: str, msg: str) -> None:
    color = _TAG_COLORS.get(tag, _DIM)
    print(f"  {_ts()} {color}[{tag.upper()}]{_RST} {msg}")


def _user(text: str) -> None:
    print(f"\n  {_ts()} {_WHITE}{_BOLD}[USER]{_RST} {text}")


def _agent(text: str) -> None:
    print(f"  {_ts()} {_CYAN}{_BOLD}[AGENT]{_RST} {text}")


def _system(text: str) -> None:
    print(f"\n  {_ts()} {_MAGENTA}{_BOLD}[SYSTEM]{_RST} {text}")


##### HANDLE — simulates agent output #####


class ConsoleHandle:
    async def respond(self, *, instructions: str) -> None:
        _agent(f"{_GREEN}(callback delivered){_RST} {instructions[:90]}")

    async def say(self, *, text: str) -> None:
        _agent(text)

    async def interrupt(self) -> None:
        _system(f"{_RED}--- INTERRUPT: background task result incoming ---{_RST}")

    async def enrich(self, *, data: dict[str, Any]) -> None:
        pass

    async def set_prompt(self, *, prompt: str) -> None:
        _system(f"Prompt updated: {prompt[:60]}")

    async def swap(self, *, agent_id: str) -> None:
        _system(f"Swapping to {agent_id}")


##### SECURITY HOOK #####


async def security_hook(event: Event) -> HookResult:
    tool = event.payload.get("tool", "")
    if tool in {"bash_rm", "exec", "drop_table"}:
        return HookResult(HookAction.BLOCK, reason=f"blocked {tool}")
    return HookResult(HookAction.CONTINUE)


##### HEAVY TOOLS (the point of the demo) #####


async def heavy_web_research(query: str) -> dict[str, Any]:
    """Simulates a HEAVY tool — scrapes 10 pages, summarizes. Takes 5 seconds."""
    _system(f"{_YELLOW}heavy_web_research started: '{query}' (will take ~5s)...{_RST}")
    await asyncio.sleep(5.0)
    return {
        "message": f"Research complete for '{query}': found 47 relevant sources, "
        "top insight: async event-driven architectures outperform synchronous "
        "orchestration by 3x in multi-agent scenarios.",
    }


async def heavy_data_analysis(dataset: str) -> dict[str, Any]:
    """Simulates another HEAVY tool — processes a large dataset. Takes 8 seconds."""
    _system(f"{_YELLOW}heavy_data_analysis started: '{dataset}' (will take ~8s)...{_RST}")
    await asyncio.sleep(8.0)
    return {
        "message": f"Analysis of '{dataset}' complete: 12,847 records processed, "
        "anomaly rate 2.3%, confidence 0.97. Recommend immediate review of cluster B.",
    }


##### AGENT SIMULATION — the double loop #####


async def simulate_conversation(reactor: Reactor) -> None:
    """Simulates a real conversation where heavy tools run in background."""

    def emit(signal: Signal, **payload: Any) -> None:
        reactor.emit(Event(signal=signal, payload=payload))

    # ── User asks something that needs deep research ───────────────
    _user("Can you research async event-driven patterns for AI agents?")
    emit(Signal.TURN_STARTED)
    emit(Signal.MODEL_REQUESTED)
    await asyncio.sleep(0.1)
    emit(Signal.MODEL_RESPONDED, tokens_used=300)

    _agent("Sure! I'll kick off a deep web research for you. This takes a while — "
           "I'll let you know when results are in. Meanwhile, ask me anything.")

    # Submit heavy task — runs in background, agent doesn't block
    await reactor.tasks.submit("web_research", heavy_web_research, query="async EDA for AI agents")
    emit(Signal.TURN_ENDED)

    await asyncio.sleep(0.5)

    # ── User keeps chatting — agent responds instantly ─────────────
    _user("While that's running — what's the capital of Japan?")
    emit(Signal.TURN_STARTED)
    emit(Signal.MODEL_REQUESTED)
    await asyncio.sleep(0.1)
    emit(Signal.MODEL_RESPONDED, tokens_used=150)
    _agent("Tokyo.")
    emit(Signal.TURN_ENDED)

    await asyncio.sleep(1.0)

    _user("And what's 2^10?")
    emit(Signal.TURN_STARTED)
    emit(Signal.MODEL_REQUESTED)
    await asyncio.sleep(0.1)
    emit(Signal.MODEL_RESPONDED, tokens_used=100)
    _agent("1024.")
    emit(Signal.TURN_ENDED)

    await asyncio.sleep(1.0)

    # ── At ~3.5s mark: background research still running ───────────
    _user("Is the research done yet?")
    emit(Signal.TURN_STARTED)
    emit(Signal.MODEL_REQUESTED)
    await asyncio.sleep(0.1)
    emit(Signal.MODEL_RESPONDED, tokens_used=200)
    _agent("Not yet — the deep web research is still running. I'll interrupt to share "
           "results as soon as it's ready. Keep going!")
    emit(Signal.TURN_ENDED)

    await asyncio.sleep(1.5)

    # ── At ~5s: heavy_web_research should complete ─────────────────
    # TaskCompletedPolicy will fire: INTERRUPT + RESPOND
    # We just wait and let the reactor handle it.
    _system("Waiting for background task to deliver...")
    await asyncio.sleep(2.5)

    # ── After callback, user asks for ANOTHER heavy task ───────────
    _user("Great results! Now analyze the 'q1_sales' dataset for anomalies.")
    emit(Signal.TURN_STARTED)
    emit(Signal.MODEL_REQUESTED)
    await asyncio.sleep(0.1)
    emit(Signal.MODEL_RESPONDED, tokens_used=250)

    _agent("On it — submitting the data analysis. This one takes even longer (~8s). "
           "I'll stay here with you.")

    await reactor.tasks.submit("data_analysis", heavy_data_analysis, dataset="q1_sales")
    emit(Signal.TURN_ENDED)

    await asyncio.sleep(1.0)

    # ── User chats while SECOND heavy task runs ────────────────────
    _user("What's the meaning of life?")
    emit(Signal.TURN_STARTED)
    emit(Signal.MODEL_REQUESTED)
    await asyncio.sleep(0.1)
    emit(Signal.MODEL_RESPONDED, tokens_used=180)
    _agent("42, according to Douglas Adams.")
    emit(Signal.TURN_ENDED)

    await asyncio.sleep(1.5)

    _user("Tell me a fun fact about octopuses.")
    emit(Signal.TURN_STARTED)
    emit(Signal.MODEL_REQUESTED)
    await asyncio.sleep(0.1)
    emit(Signal.MODEL_RESPONDED, tokens_used=200)
    _agent("They have three hearts and blue blood.")
    emit(Signal.TURN_ENDED)

    # ── Wait for second heavy task to complete ─────────────────────
    _system("Waiting for data analysis to deliver...")
    await asyncio.sleep(6.0)

    # ── One more turn after everything is done ─────────────────────
    _user("Thanks, that's all for now!")
    emit(Signal.TURN_STARTED)
    emit(Signal.MODEL_REQUESTED)
    await asyncio.sleep(0.1)
    emit(Signal.MODEL_RESPONDED, tokens_used=100)
    _agent("You're welcome! Both background tasks delivered successfully.")
    emit(Signal.TURN_ENDED)


##### MAIN #####


async def main() -> None:
    global _T0
    _T0 = time.monotonic()

    state = ReactiveState()
    state.token_budget = 10_000
    handle = ConsoleHandle()

    policies = (
        TaskCompletedPolicy(cooldown=0.0),
        BudgetPolicy(warn_ratio=0.8),
    )

    reactor = Reactor(handle, state, policies, tick_interval=5.0, log=_reactor_log)
    reactor.bus.hook(Signal.TOOL_CALLED, security_hook, priority=10)

    # Banner
    print(f"\n{'=' * 70}")
    print(f"  AEDA Demo — Double Loop (heavy background tasks + live conversation)")
    print(f"  Budget: {state.token_budget} tokens | Policies: TaskCompleted, Budget")
    print(f"{'=' * 70}")

    reactor_task = asyncio.create_task(reactor.run())
    await asyncio.sleep(0.05)

    reactor.emit(Event(signal=Signal.AGENT_STARTED, payload={"agent_id": "assistant"}))
    await simulate_conversation(reactor)

    reactor.stop()
    await reactor_task

    # Summary
    elapsed = time.monotonic() - _T0
    print(f"\n{'=' * 70}")
    print(f"  {_GREEN}{_BOLD}SUMMARY{_RST}")
    print(f"{'=' * 70}")
    print(f"  Wall time:   {elapsed:.1f}s")
    print(f"  Turns:       {state.turn_count}")
    print(f"  Tokens:      {state.tokens_used}/{state.token_budget} ({state.budget_ratio:.0%})")
    print(f"  Tool calls:  {state.agent.tool_calls}")
    print(f"  BG tasks:    2 submitted, 2 delivered")
    print(f"  Data keys:   {list(state.data.keys())}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    asyncio.run(main())
