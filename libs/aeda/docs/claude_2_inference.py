"""
Agent Architecture — Inspired by Claude Code internals.

Layers:
    1. HookEvent taxonomy (27+ lifecycle events across 4 execution types)
    2. GlobalState singleton (mutable, fast bootstrap, no DI container)
    3. Tool abstraction with concurrency safety declarations
    4. HookRegistry + AgentLoop reactor

Run: python agent_arch.py
"""

from __future__ import annotations

import asyncio
import inspect
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import Any, Self


# ═══════════════════════════════════════════════════════════════════════
# 1. HOOK EVENT TAXONOMY
#    Clean enum-based lifecycle events grouped by execution domain.
#    Each hook point maps to a specific moment in the agent loop.
# ═══════════════════════════════════════════════════════════════════════

class HookEvent(StrEnum):
    """Lifecycle events the hook system can intercept."""

    # ── Tool lifecycle ──────────────────────────────────────────────
    ON_TOOL_REGISTER = auto()       # Tool added to registry
    ON_TOOL_PRE_VALIDATE = auto()   # Before permission check
    ON_TOOL_AUTHORIZED = auto()     # Permission granted
    ON_TOOL_DENIED = auto()         # Permission denied → abort
    ON_TOOL_PRE_EXECUTE = auto()    # Right before execution
    ON_TOOL_POST_EXECUTE = auto()   # After execution, has result
    ON_TOOL_ERROR = auto()          # Execution failed
    ON_TOOL_RETRY = auto()          # Retrying after failure

    # ── LLM lifecycle ──────────────────────────────────────────────
    ON_LLM_PRE_REQUEST = auto()     # Before API call (can mutate prompt)
    ON_LLM_POST_RESPONSE = auto()   # After API response (can inspect/transform)
    ON_LLM_STREAM_CHUNK = auto()    # Each streaming chunk
    ON_LLM_TOKEN_LIMIT = auto()     # Context window approaching limit
    ON_LLM_ERROR = auto()           # API call failed
    ON_LLM_COMPACTION = auto()      # Context compacted (summarized)

    # ── Agent loop lifecycle ───────────────────────────────────────
    ON_TURN_START = auto()          # New reasoning turn begins
    ON_TURN_END = auto()            # Turn completed (tool + LLM cycle)
    ON_PLAN_CREATED = auto()        # Agent created execution plan
    ON_PLAN_STEP_COMPLETE = auto()  # One plan step finished
    ON_BUDGET_WARNING = auto()      # Token/cost budget approaching
    ON_BUDGET_EXHAUSTED = auto()    # Budget hit → force stop
    ON_AGENT_IDLE = auto()          # No pending tasks, waiting for input

    # ── Session lifecycle ──────────────────────────────────────────
    ON_SESSION_START = auto()       # Bootstrap complete
    ON_SESSION_RESUME = auto()      # Resumed from checkpoint
    ON_SESSION_END = auto()         # Graceful shutdown
    ON_CONTEXT_INJECTED = auto()    # External context (CLAUDE.md equiv) loaded
    ON_MEMORY_LOADED = auto()       # Long-term memory retrieved
    ON_SUBAGENT_SPAWNED = auto()    # Child agent created
    ON_SUBAGENT_RETURNED = auto()   # Child agent finished


# ═══════════════════════════════════════════════════════════════════════
# 2. HOOK SYSTEM
#    Hooks can: inspect, mutate, block, or short-circuit any lifecycle point.
#    Each hook returns a HookResult that controls flow.
# ═══════════════════════════════════════════════════════════════════════

class HookAction(StrEnum):
    """What the agent loop should do after a hook fires."""
    CONTINUE = auto()    # Proceed normally
    MUTATE = auto()      # Proceed with mutated payload
    BLOCK = auto()       # Abort this operation
    SKIP = auto()        # Skip remaining hooks, proceed with current payload


@dataclass(frozen=True, slots=True)
class HookResult:
    """Returned by every hook handler to control agent loop flow."""
    action: HookAction = HookAction.CONTINUE
    payload: Any = None
    reason: str | None = None


type HookHandler = Callable[[HookEvent, dict[str, Any]], Awaitable[HookResult]]


class HookRegistry:
    """
    Central hook dispatcher. Handlers registered per event, executed in priority order.
    Any handler can block/mutate/skip — first BLOCK wins.

    O(n) per event dispatch where n = handlers for that event.
    """
    __slots__ = ("_hooks",)

    def __init__(self) -> None:
        self._hooks: dict[HookEvent, list[tuple[int, HookHandler]]] = {}

    def on(self, event: HookEvent, handler: HookHandler, *, priority: int = 100) -> Self:
        """Register handler. Lower priority = runs first. Chainable."""
        self._hooks.setdefault(event, []).append((priority, handler))
        self._hooks[event].sort(key=lambda x: x[0])
        return self

    async def dispatch(self, event: HookEvent, context: dict[str, Any]) -> HookResult:
        """
        Fire all handlers for event in priority order.
        First BLOCK → abort. MUTATE → update context. SKIP → stop chain.
        """
        if not (handlers := self._hooks.get(event)):
            return HookResult(HookAction.CONTINUE)

        for _, handler in handlers:
            result = await handler(event, context)
            match result.action:
                case HookAction.BLOCK:
                    return result
                case HookAction.MUTATE:
                    context.update(result.payload or {})
                case HookAction.SKIP:
                    return HookResult(HookAction.CONTINUE, payload=context)
                case HookAction.CONTINUE:
                    pass

        return HookResult(HookAction.CONTINUE, payload=context)


# ═══════════════════════════════════════════════════════════════════════
# 3. GLOBAL STATE SINGLETON
#    Mutable session state. Fast bootstrap, no DI container overhead.
#    All modules read/write from this. Controversial but effective.
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class AgentState:
    """
    Global mutable singleton — the 'session object' everything reads from.
    Mirrors Claude Code's bootstrap/state.ts pattern.
    """
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    turn_count: int = 0
    total_tokens_used: int = 0
    token_budget: int = 200_000
    conversation: list[dict[str, Any]] = field(default_factory=list)
    tool_results: dict[str, Any] = field(default_factory=dict)
    pending_tasks: dict[str, asyncio.Task[Any]] = field(default_factory=dict)
    memory: dict[str, Any] = field(default_factory=dict)
    is_running: bool = False
    created_at: float = field(default_factory=time.time)

    @property
    def budget_remaining(self) -> int:
        return self.token_budget - self.total_tokens_used

    @property
    def budget_ratio(self) -> float:
        return self.total_tokens_used / self.token_budget if self.token_budget else 0.0


# Module-level singleton — import from anywhere
_state: AgentState | None = None


def get_state() -> AgentState:
    """Access global state. Lazy init on first call."""
    global _state  # noqa: PLW0603
    if _state is None:
        _state = AgentState()
    return _state


def reset_state() -> AgentState:
    """Reset for new session. Returns fresh state."""
    global _state  # noqa: PLW0603
    _state = AgentState()
    return _state


# ═══════════════════════════════════════════════════════════════════════
# 4. TOOL ABSTRACTION WITH CONCURRENCY SAFETY
#    Each tool declares if it's safe to run concurrently.
#    The executor uses a semaphore for unsafe tools.
# ═══════════════════════════════════════════════════════════════════════

class PermissionLevel(StrEnum):
    """Tool permission tiers — from most to least permissive."""
    UNRESTRICTED = auto()    # No checks (internal/testing)
    AUTO_APPROVE = auto()    # Logged but no prompt
    CLASSIFIER = auto()      # LLM-based allow/deny
    WRITE_AUTO = auto()      # File edits auto-approved, mutations prompt
    INTERACTIVE = auto()     # User approves each action
    READ_ONLY = auto()       # All mutations blocked
    DELEGATED = auto()       # Escalate to parent agent (subagent mode)


@dataclass(frozen=True, slots=True)
class ToolSchema:
    """Tool metadata — the uniform interface every tool exposes."""
    name: str
    description: str
    concurrency_safe: bool = False       # Can run in parallel?
    permission: PermissionLevel = PermissionLevel.INTERACTIVE
    max_retries: int = 2
    timeout_seconds: float = 30.0

    def is_concurrent(self) -> bool:
        """Claude Code pattern: each tool declares its own concurrency safety."""
        return self.concurrency_safe


type ToolFn = Callable[..., Awaitable[Any]]


@dataclass(slots=True)
class Tool:
    """Wraps a callable with schema + metadata. Uniform interface for the agent loop."""
    schema: ToolSchema
    execute: ToolFn


class ToolRegistry:
    """
    Central tool catalog. Tools register here, agent loop queries here.
    O(1) lookup by name.
    """
    __slots__ = ("_tools", "_hooks")

    def __init__(self, hooks: HookRegistry) -> None:
        self._tools: dict[str, Tool] = {}
        self._hooks = hooks

    async def register(self, tool: Tool) -> None:
        self._tools[tool.schema.name] = tool
        await self._hooks.dispatch(
            HookEvent.ON_TOOL_REGISTER,
            {"tool_name": tool.schema.name, "schema": tool.schema},
        )

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    @property
    def names(self) -> list[str]:
        return list(self._tools)


# ═══════════════════════════════════════════════════════════════════════
# 5. TOOL EXECUTOR — CONCURRENCY-AWARE
#    Unsafe tools go through a semaphore (serialized).
#    Safe tools run via asyncio.gather (parallel).
# ═══════════════════════════════════════════════════════════════════════

class ToolExecutor:
    """
    Executes tools respecting concurrency declarations.
    Unsafe tools → serialized via semaphore.
    Safe tools → parallel via gather.
    Hooks fire at every lifecycle point.
    """
    __slots__ = ("_hooks", "_registry", "_unsafe_sem")

    def __init__(self, hooks: HookRegistry, registry: ToolRegistry) -> None:
        self._hooks = hooks
        self._registry = registry
        self._unsafe_sem = asyncio.Semaphore(1)  # Mutex for unsafe tools

    async def execute(self, tool_name: str, **kwargs: Any) -> Any:
        """Full lifecycle: validate → authorize → execute → handle result/error."""
        tool = self._registry.get(tool_name)
        if tool is None:
            raise ValueError(f"Tool '{tool_name}' not registered")

        task_id = f"{tool_name}:{uuid.uuid4().hex[:8]}"
        ctx = {"task_id": task_id, "tool_name": tool_name, "args": kwargs, "schema": tool.schema}

        # ── Pre-validation hook (can block) ─────────────────────────
        if (r := await self._hooks.dispatch(HookEvent.ON_TOOL_PRE_VALIDATE, ctx)).action == HookAction.BLOCK:
            await self._hooks.dispatch(HookEvent.ON_TOOL_DENIED, {**ctx, "reason": r.reason})
            return None

        # ── Authorization hook ──────────────────────────────────────
        auth_ctx = {**ctx, "permission": tool.schema.permission}
        if (r := await self._hooks.dispatch(HookEvent.ON_TOOL_AUTHORIZED, auth_ctx)).action == HookAction.BLOCK:
            await self._hooks.dispatch(HookEvent.ON_TOOL_DENIED, {**ctx, "reason": r.reason})
            return None

        # ── Execute with concurrency safety ─────────────────────────
        await self._hooks.dispatch(HookEvent.ON_TOOL_PRE_EXECUTE, ctx)

        for attempt in range(tool.schema.max_retries + 1):
            try:
                result = await self._execute_with_safety(tool, kwargs)
                ctx["result"] = result
                await self._hooks.dispatch(HookEvent.ON_TOOL_POST_EXECUTE, ctx)
                return result

            except Exception as exc:
                ctx["error"] = str(exc)
                ctx["attempt"] = attempt
                await self._hooks.dispatch(HookEvent.ON_TOOL_ERROR, ctx)

                if attempt < tool.schema.max_retries:
                    await self._hooks.dispatch(HookEvent.ON_TOOL_RETRY, ctx)
                else:
                    raise

    async def _execute_with_safety(self, tool: Tool, kwargs: dict[str, Any]) -> Any:
        """Concurrency gate: safe tools run free, unsafe go through semaphore."""
        if tool.schema.is_concurrent():
            return await asyncio.wait_for(tool.execute(**kwargs), timeout=tool.schema.timeout_seconds)

        async with self._unsafe_sem:
            return await asyncio.wait_for(tool.execute(**kwargs), timeout=tool.schema.timeout_seconds)

    async def execute_batch(self, calls: list[tuple[str, dict[str, Any]]]) -> list[Any]:
        """
        Intelligent batching: safe tools run in parallel, unsafe serialized.
        Groups by concurrency safety, then gathers safe ones.
        """
        safe, unsafe = [], []
        for tool_name, kwargs in calls:
            tool = self._registry.get(tool_name)
            if tool and tool.schema.is_concurrent():
                safe.append((tool_name, kwargs))
            else:
                unsafe.append((tool_name, kwargs))

        # Safe tools → parallel
        safe_coros = [self.execute(name, **kw) for name, kw in safe]
        # Unsafe tools → sequential (respects semaphore)
        unsafe_coros = [self.execute(name, **kw) for name, kw in unsafe]

        safe_results = await asyncio.gather(*safe_coros, return_exceptions=True) if safe_coros else []
        unsafe_results = []
        for coro in unsafe_coros:
            unsafe_results.append(await coro)

        return [*safe_results, *unsafe_results]


# ═══════════════════════════════════════════════════════════════════════
# 6. AGENT LOOP — THE REACTOR
#    Multi-turn loop: reason → select tools → execute → inject results → repeat.
#    Hooks fire at every boundary. Budget tracking built-in.
# ═══════════════════════════════════════════════════════════════════════

class AgentLoop:
    """
    The brain. Orchestrates turns until task complete or budget exhausted.
    Each turn: hooks → LLM call → tool selection → tool execution → result injection.
    """
    __slots__ = ("_hooks", "_executor", "_state", "_llm_fn")

    def __init__(
        self,
        hooks: HookRegistry,
        executor: ToolExecutor,
        state: AgentState,
        llm_fn: Callable[[list[dict[str, Any]]], Awaitable[dict[str, Any]]],
    ) -> None:
        self._hooks = hooks
        self._executor = executor
        self._state = state
        self._llm_fn = llm_fn

    async def run(self, user_input: str) -> str:
        """Main agent loop. Runs until LLM says stop or budget exhausted."""
        self._state.is_running = True
        self._state.conversation.append({"role": "user", "content": user_input})

        await self._hooks.dispatch(HookEvent.ON_SESSION_START, {"session_id": self._state.session_id})

        final_response = ""

        while self._state.is_running:
            self._state.turn_count += 1
            turn_ctx = {"turn": self._state.turn_count, "budget_ratio": self._state.budget_ratio}

            # ── Budget check ────────────────────────────────────────
            if self._state.budget_ratio > 0.9:
                await self._hooks.dispatch(HookEvent.ON_BUDGET_WARNING, turn_ctx)
            if self._state.budget_ratio >= 1.0:
                await self._hooks.dispatch(HookEvent.ON_BUDGET_EXHAUSTED, turn_ctx)
                break

            # ── Turn start ──────────────────────────────────────────
            await self._hooks.dispatch(HookEvent.ON_TURN_START, turn_ctx)

            # ── LLM call (hooks can mutate the conversation) ────────
            llm_ctx = {"messages": self._state.conversation}
            hook_result = await self._hooks.dispatch(HookEvent.ON_LLM_PRE_REQUEST, llm_ctx)
            if hook_result.action == HookAction.BLOCK:
                break

            llm_response = await self._llm_fn(self._state.conversation)

            self._state.total_tokens_used += llm_response.get("tokens_used", 0)
            await self._hooks.dispatch(
                HookEvent.ON_LLM_POST_RESPONSE,
                {"response": llm_response, "tokens": self._state.total_tokens_used},
            )

            # ── Process LLM decision ────────────────────────────────
            match llm_response.get("action"):
                case "tool_calls":
                    # LLM wants to use tools → execute → inject results
                    tool_calls = llm_response["tool_calls"]
                    results = await self._executor.execute_batch(
                        [(tc["name"], tc.get("args", {})) for tc in tool_calls],
                    )
                    # Inject results back into conversation for next turn
                    for tc, result in zip(tool_calls, results, strict=True):
                        self._state.conversation.append({
                            "role": "tool_result",
                            "tool": tc["name"],
                            "content": result,
                        })
                        self._state.tool_results[tc["name"]] = result

                case "final_answer":
                    final_response = llm_response["content"]
                    self._state.conversation.append({"role": "assistant", "content": final_response})
                    self._state.is_running = False

                case "ask_user":
                    # Agent needs clarification → pause loop
                    final_response = llm_response["content"]
                    self._state.is_running = False

                case _:
                    self._state.is_running = False

            await self._hooks.dispatch(HookEvent.ON_TURN_END, {**turn_ctx, "action": llm_response.get("action")})

        await self._hooks.dispatch(HookEvent.ON_SESSION_END, {"turns": self._state.turn_count})
        return final_response


# ═══════════════════════════════════════════════════════════════════════
# 7. EXAMPLE: WIRING IT ALL TOGETHER
# ═══════════════════════════════════════════════════════════════════════

# ── Fake tools ──────────────────────────────────────────────────────

async def grep_search(pattern: str, path: str = ".") -> dict[str, Any]:
    """Read-only, concurrency safe — like Claude Code's Grep tool."""
    await asyncio.sleep(0.1)
    return {"matches": [f"{path}/main.py:42: {pattern}", f"{path}/utils.py:17: {pattern}"]}


async def file_write(path: str, content: str) -> dict[str, Any]:
    """Mutation — NOT concurrency safe. Must serialize."""
    await asyncio.sleep(0.3)
    return {"written": path, "bytes": len(content)}


async def bash_exec(command: str) -> dict[str, Any]:
    """Shell execution — NOT concurrency safe, needs permission."""
    await asyncio.sleep(0.5)
    return {"stdout": f"executed: {command}", "exit_code": 0}


async def web_search(query: str) -> dict[str, Any]:
    """External API — concurrency safe, read-only."""
    await asyncio.sleep(0.8)
    return {"results": [f"Result for: {query}"]}


# ── Example hooks ───────────────────────────────────────────────────

async def log_hook(event: HookEvent, ctx: dict[str, Any]) -> HookResult:
    """Universal logger hook — observes everything."""
    emoji_map = {
        "ON_TOOL_": "🔧", "ON_LLM_": "🧠", "ON_TURN_": "🔄",
        "ON_SESSION_": "📡", "ON_BUDGET_": "💰",
    }
    emoji = next((v for k, v in emoji_map.items() if event.value.startswith(k.lower())), "📌")
    print(f"  {emoji} {event.value.upper()}: {', '.join(f'{k}={v}' for k, v in ctx.items() if k != 'messages')}")
    return HookResult(HookAction.CONTINUE)


async def security_hook(event: HookEvent, ctx: dict[str, Any]) -> HookResult:
    """Blocks dangerous bash commands. Fires on ON_TOOL_PRE_VALIDATE."""
    if ctx.get("tool_name") == "bash" and "rm -rf" in str(ctx.get("args", {})):
        return HookResult(HookAction.BLOCK, reason="Dangerous command blocked by security hook")
    return HookResult(HookAction.CONTINUE)


async def token_tracker_hook(event: HookEvent, ctx: dict[str, Any]) -> HookResult:
    """Tracks token usage, fires compaction warning at 80%."""
    state = get_state()
    if state.budget_ratio > 0.8:
        print(f"  ⚠️  TOKEN_WARNING: {state.budget_ratio:.0%} budget used")
    return HookResult(HookAction.CONTINUE)


# ── Fake LLM function (simulates multi-turn reasoning) ──────────────

async def fake_llm(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Simulates LLM decisions across turns:
      Turn 1 → calls grep + web_search (parallel, both safe)
      Turn 2 → calls file_write (serialized, unsafe)
      Turn 3 → final answer
    """
    await asyncio.sleep(0.2)
    turn = sum(1 for m in messages if m["role"] in ("assistant", "tool_result")) // 2 + 1

    match turn:
        case 1:
            return {
                "action": "tool_calls",
                "tokens_used": 1500,
                "tool_calls": [
                    {"name": "grep", "args": {"pattern": "async def", "path": "src/"}},
                    {"name": "web_search", "args": {"query": "asyncio best practices 2025"}},
                ],
            }
        case 2:
            return {
                "action": "tool_calls",
                "tokens_used": 800,
                "tool_calls": [
                    {"name": "file_write", "args": {"path": "src/fix.py", "content": "# patched"}},
                ],
            }
        case _:
            return {
                "action": "final_answer",
                "tokens_used": 500,
                "content": "Done. Found 2 async functions, applied fix, verified with web search.",
            }


# ── Bootstrap & run ─────────────────────────────────────────────────

async def main() -> None:
    # 1. Fresh state
    state = reset_state()
    state.token_budget = 50_000

    # 2. Hook registry — wire lifecycle handlers
    hooks = HookRegistry()
    hooks.on(HookEvent.ON_TOOL_PRE_VALIDATE, security_hook, priority=10)   # Security first
    hooks.on(HookEvent.ON_LLM_POST_RESPONSE, token_tracker_hook, priority=50)

    # Attach logger to all events for visibility
    for event in HookEvent:
        hooks.on(event, log_hook, priority=999)  # Lowest priority = runs last

    # 3. Tool registry — declare concurrency safety per tool
    registry = ToolRegistry(hooks)
    await registry.register(Tool(
        schema=ToolSchema("grep", "Search code", concurrency_safe=True, permission=PermissionLevel.READ_ONLY),
        execute=grep_search,
    ))
    await registry.register(Tool(
        schema=ToolSchema("web_search", "Search web", concurrency_safe=True, permission=PermissionLevel.READ_ONLY),
        execute=web_search,
    ))
    await registry.register(Tool(
        schema=ToolSchema("file_write", "Write files", concurrency_safe=False, permission=PermissionLevel.WRITE_AUTO),
        execute=file_write,
    ))
    await registry.register(Tool(
        schema=ToolSchema("bash", "Run commands", concurrency_safe=False, permission=PermissionLevel.INTERACTIVE),
        execute=bash_exec,
    ))

    # 4. Agent loop — the reactor
    agent = AgentLoop(hooks, ToolExecutor(hooks, registry), state, fake_llm)

    print(f"\n{'═' * 60}")
    print(f"  🚀 SESSION {state.session_id} — budget: {state.token_budget:,} tokens")
    print(f"{'═' * 60}\n")

    result = await agent.run("Find all async functions and apply the performance fix")

    print(f"\n{'═' * 60}")
    print(f"  ✅ FINAL: {result}")
    print(f"  📊 Turns: {state.turn_count} | Tokens: {state.total_tokens_used:,}/{state.token_budget:,}")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    asyncio.run(main())