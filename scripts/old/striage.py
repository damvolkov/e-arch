# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULE 1: e_agents/rtc/state.py                                           ║
# ║  Framework-agnostic reactive state. Zero LiveKit dependency.                ║
# ║  Reusable across any agent framework that supports async + userdata.        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

from __future__ import annotations

import asyncio
import logging
import sys
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from enum import IntEnum, StrEnum, auto
from typing import Any
from uuid import uuid4

from livekit import agents
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    RunContext,
    cli,
    function_tool,
)
from livekit.plugins import google
from pydantic import BaseModel, ConfigDict, Field, computed_field

logger = logging.getLogger(__name__)


# ─── Priority & Interruption ────────────────────────────────────────────────

class Priority(IntEnum):
    """Task priority — lower value = higher urgency."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


class TaskStatus(StrEnum):
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()


class InterruptionStrategy(StrEnum):
    IMMEDIATE = "immediate"
    TURN_BOUNDARY = "turn_boundary"
    NATURAL_PAUSE = "natural_pause"
    ENQUEUE = "enqueue"


class InterruptionPolicy(BaseModel):
    """Configurable priority → interruption behavior mapping. O(1) resolve."""
    model_config = ConfigDict(frozen=True)

    rules: dict[Priority, InterruptionStrategy] = Field(default_factory=lambda: {
        Priority.CRITICAL: InterruptionStrategy.IMMEDIATE,
        Priority.HIGH: InterruptionStrategy.TURN_BOUNDARY,
        Priority.NORMAL: InterruptionStrategy.TURN_BOUNDARY,
        Priority.LOW: InterruptionStrategy.NATURAL_PAUSE,
        Priority.BACKGROUND: InterruptionStrategy.ENQUEUE,
    })
    idle_timeout_seconds: float = 3.0

    def resolve(self, priority: Priority) -> InterruptionStrategy:
        return self.rules.get(priority, InterruptionStrategy.ENQUEUE)


# ─── State Update (structured payload from background agents/tools) ──────────

class StateUpdate(BaseModel):
    """Immutable structured result written by background tasks into shared state."""
    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    task_id: str
    source: str
    priority: Priority = Priority.NORMAL
    status: TaskStatus = TaskStatus.COMPLETED
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.monotonic)

    @computed_field
    @property
    def age_seconds(self) -> float:
        return time.monotonic() - self.created_at


# ─── Background Task ────────────────────────────────────────────────────────

class BackgroundTask(BaseModel):
    """A unit of async work dispatched by the orchestrator."""
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    name: str
    priority: Priority = Priority.NORMAL
    handler: Callable[..., Awaitable[dict[str, Any]]] = Field(exclude=True)
    kwargs: dict[str, Any] = Field(default_factory=dict)
    source: str = "system"


# ─── Reactive State (observable, event-driven, framework-agnostic) ───────────

class ReactiveState:
    """Observable shared state with pub/sub for mutations.

    Background agents write updates; front agent subscribes and reacts.
    asyncio.Queue provides backpressure-safe delivery.
    """
    __slots__ = ("_updates", "_queue", "_context", "_lock")

    def __init__(self) -> None:
        self._updates: dict[str, StateUpdate] = {}
        self._queue: asyncio.Queue[StateUpdate] = asyncio.Queue()
        self._context: dict[str, Any] = {}
        self._lock = asyncio.Lock()

    async def push(self, update: StateUpdate) -> None:
        """Called by background tasks on completion. O(1)."""
        async with self._lock:
            self._updates[update.id] = update
        await self._queue.put(update)

    async def wait(self, timeout: float | None = None) -> StateUpdate | None:
        """Blocking wait for next update. Returns None on timeout."""
        with suppress(asyncio.TimeoutError):
            return await asyncio.wait_for(self._queue.get(), timeout=timeout)
        return None

    def drain(self) -> list[StateUpdate]:
        """Non-blocking: grab all pending, sorted by priority. O(k log k)."""
        pending: list[StateUpdate] = []
        while not self._queue.empty():
            with suppress(asyncio.QueueEmpty):
                pending.append(self._queue.get_nowait())
        return sorted(pending, key=lambda u: (u.priority, u.created_at))

    async def set_ctx(self, key: str, value: Any) -> None:
        async with self._lock:
            self._context[key] = value

    def get_ctx(self, key: str, default: Any = None) -> Any:
        return self._context.get(key, default)

    @property
    def pending_count(self) -> int:
        return self._queue.qsize()


# ─── Background Orchestrator (inner loop, framework-agnostic) ────────────────

class BackgroundOrchestrator:
    """Dispatches tasks concurrently, writes results to reactive state."""
    __slots__ = ("_state", "_semaphore", "_handles")

    def __init__(self, state: ReactiveState, max_concurrency: int = 5) -> None:
        self._state = state
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._handles: dict[str, asyncio.Task[None]] = {}

    async def submit(self, task: BackgroundTask) -> str:
        """Submit task for background execution. Returns task_id."""
        await self._state.push(StateUpdate(
            task_id=task.id,
            source=task.source,
            priority=Priority.BACKGROUND,
            status=TaskStatus.RUNNING,
            payload={"task_name": task.name},
        ))
        self._handles[task.id] = asyncio.create_task(self._run(task))
        return task.id

    async def _run(self, task: BackgroundTask) -> None:
        async with self._semaphore:
            try:
                result = await task.handler(**task.kwargs)
                await self._state.push(StateUpdate(
                    task_id=task.id,
                    source=task.source,
                    priority=task.priority,
                    status=TaskStatus.COMPLETED,
                    payload=result,
                ))
            except Exception as exc:
                await self._state.push(StateUpdate(
                    task_id=task.id,
                    source=task.source,
                    priority=Priority.HIGH,
                    status=TaskStatus.FAILED,
                    payload={"error": str(exc), "error_type": type(exc).__name__},
                ))
            finally:
                self._handles.pop(task.id, None)

    async def cancel(self, task_id: str) -> bool:
        if handle := self._handles.pop(task_id, None):
            handle.cancel()
            return True
        return False

    @property
    def active_count(self) -> int:
        return len(self._handles)

    async def shutdown(self) -> None:
        """Cancel all active tasks gracefully."""
        for handle in self._handles.values():
            handle.cancel()
        if self._handles:
            await asyncio.gather(*self._handles.values(), return_exceptions=True)
        self._handles.clear()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULE 2: e_agents/rtc/base.py                                            ║
# ║  LiveKit-aware reactive base agent.                                         ║
# ║  Bridges framework-agnostic state with LiveKit's Agent lifecycle.           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝



# ─── Session State (LiveKit userdata wrapper) ────────────────────────────────

class SessionState[A]:
    """Typed container for LiveKit's AgentSession.userdata.

    Generic over A (agent type) so the agent registry is type-safe.
    Composes ReactiveState + BackgroundOrchestrator + agent-transfer state.
    """
    __slots__ = (
        "reactive", "orchestrator", "agents",
        "prev_agent", "policy", "_idle_since",
    )

    def __init__(
        self,
        agents: dict[str, A],
        *,
        policy: InterruptionPolicy | None = None,
        max_concurrency: int = 5,
    ) -> None:
        self.reactive = ReactiveState()
        self.orchestrator = BackgroundOrchestrator(
            self.reactive, max_concurrency=max_concurrency,
        )
        self.agents: dict[str, A] = agents
        self.prev_agent: A | None = None
        self.policy = policy or InterruptionPolicy()
        self._idle_since: float = time.monotonic()

    def mark_idle(self) -> None:
        self._idle_since = time.monotonic()

    @property
    def idle_duration(self) -> float:
        return time.monotonic() - self._idle_since

    async def shutdown(self) -> None:
        await self.orchestrator.shutdown()


# ─── Chat Context Helpers ────────────────────────────────────────────────────

def truncate_chat_ctx(
    items: list,
    keep_last_n: int = 6,
    *,
    keep_system: bool = False,
    keep_fn_calls: bool = False,
) -> list:
    """Keep last N relevant messages from a chat context. O(n)."""
    def _keep(item) -> bool:
        if not keep_system and item.type == "message" and item.role == "system":
            return False
        if not keep_fn_calls and item.type in ("function_call", "function_call_output"):
            return False
        return True

    kept: list = []
    for item in reversed(items):
        if _keep(item):
            kept.append(item)
        if len(kept) >= keep_last_n:
            break
    kept.reverse()

    # Strip leading function calls (orphaned without context)
    while kept and kept[0].type in ("function_call", "function_call_output"):
        kept.pop(0)
    return kept


# ─── Reactive Agent Base (LiveKit Agent subclass) ────────────────────────────

class ReactiveAgent(Agent):
    """Base agent implementing the outer loop of the double-loop pattern.

    Hooks into LiveKit's Agent lifecycle to monitor reactive state
    and deliver background results based on InterruptionPolicy.

    Subclasses define domain tools and prompts; this base handles:
    - Context preservation across agent transfers
    - Background state monitoring (inner loop bridge)
    - Priority-based interruption delivery
    - Background task submission convenience
    """

    def __init__(
        self,
        *,
        instructions: str,
        monitor_interval: float = 1.0,
        keep_last_n_ctx: int = 6,
        **kwargs: Any,
    ) -> None:
        super().__init__(instructions=instructions, **kwargs)
        self._monitor_interval = monitor_interval
        self._keep_last_n_ctx = keep_last_n_ctx
        self._monitor_task: asyncio.Task[None] | None = None
        self._pending_deliveries: list[StateUpdate] = []

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def state(self) -> SessionState:
        """Typed access to session userdata."""
        return self.session.userdata

    @property
    def reactive(self) -> ReactiveState:
        return self.state.reactive

    @property
    def policy(self) -> InterruptionPolicy:
        return self.state.policy

    # ── LiveKit Lifecycle Hooks ──────────────────────────────────────────

    async def on_enter(self) -> None:
        """Called when this agent becomes active. Merge context from previous agent."""
        agent_name = type(self).__name__
        logger.info("AGENT_ENTER agent=%s", agent_name)
        await self._merge_prev_context()

    async def on_exit(self) -> None:
        """Called when this agent is being replaced. Cleanup monitor."""
        self._stop_monitor()
        logger.info("AGENT_EXIT agent=%s", type(self).__name__)

    async def on_user_turn_completed(
        self,
        turn_ctx: Any,
        new_message: Any = None,
    ) -> None:
        """Inject deferred results at the natural turn boundary (official hook)."""
        pending = self.reactive.drain()
        if not pending and not self._pending_deliveries:
            return

        deliverable = self._pending_deliveries + [
            u for u in pending
            if u.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
        ]
        self._pending_deliveries.clear()

        for update in sorted(deliverable, key=lambda u: u.priority):
            if msg := self._format_update(update):
                turn_ctx.add_message(role="system", content=msg)
                logger.info(
                    "INJECT_UPDATE source=%s priority=%s",
                    update.source, update.priority.name,
                )

    # ── Agent Transfer (shared across all reactive agents) ───────────────

    async def _transfer_to(self, name: str, context: RunContext[SessionState]) -> Agent:
        """Transfer to named agent preserving reactive state + context."""
        state = context.userdata
        state.prev_agent = context.session.current_agent
        target = state.agents.get(name)
        if target is None:
            msg = f"Unknown agent: {name!r}. Available: {list(state.agents)}"
            raise ValueError(msg)
        return target

    # ── Background Task Submission ───────────────────────────────────────

    async def submit_task(self, task: BackgroundTask) -> str:
        """Submit a background task from within a tool or hook."""
        task_id = await self.state.orchestrator.submit(task)
        self._start_monitor()
        return task_id

    # ── Context Preservation ─────────────────────────────────────────────

    async def _merge_prev_context(self) -> None:
        """Merge relevant history from the previous agent's chat context."""
        prev = self.state.prev_agent
        if prev is None:
            return

        chat_ctx = self.chat_ctx.copy()
        prev_items = truncate_chat_ctx(
            prev.chat_ctx.items,
            keep_last_n=self._keep_last_n_ctx,
            keep_fn_calls=True,
        )
        existing_ids = {item.id for item in chat_ctx.items}
        chat_ctx.items.extend(i for i in prev_items if i.id not in existing_ids)
        chat_ctx.add_message(
            role="system",
            content=f"Context transferred. You are now {type(self).__name__}.",
        )
        await self.update_chat_ctx(chat_ctx)

    # ── State Monitor (inner loop bridge) ────────────────────────────────

    def _start_monitor(self) -> None:
        if self._monitor_task is None or self._monitor_task.done():
            self._monitor_task = asyncio.create_task(self._monitor_loop())

    def _stop_monitor(self) -> None:
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            self._monitor_task = None

    async def _monitor_loop(self) -> None:
        """Background loop: polls reactive state for IMMEDIATE-priority updates.

        Uses sleep+drain instead of wait_for to avoid asyncio task churn
        that interferes with LiveKit's speech pipeline synchronizer.
        """
        with suppress(asyncio.CancelledError):
            while True:
                await asyncio.sleep(self._monitor_interval)
                updates = self.reactive.drain()
                for update in updates:
                    await self._evaluate(update)

                if (
                    self._pending_deliveries
                    and self.state.idle_duration >= self.policy.idle_timeout_seconds
                ):
                    await self._flush_pending()

    async def _evaluate(self, update: StateUpdate) -> None:
        """Evaluate a single update against the interruption policy."""
        strategy = self.policy.resolve(update.priority)

        match strategy:
            case InterruptionStrategy.IMMEDIATE:
                await self._deliver(update)
            case InterruptionStrategy.TURN_BOUNDARY:
                self._pending_deliveries.append(update)
            case InterruptionStrategy.NATURAL_PAUSE | InterruptionStrategy.ENQUEUE:
                self._pending_deliveries.append(update)

    async def _deliver(self, update: StateUpdate) -> None:
        """Inject an IMMEDIATE update via generate_reply (non-blocking to pipeline)."""
        if msg := self._format_update(update):
            logger.info(
                "DELIVER_UPDATE source=%s priority=%s status=%s",
                update.source, update.priority.name, update.status,
            )
            await self.session.generate_reply(instructions=msg)

    async def _flush_pending(self) -> None:
        """Deliver all accumulated low-priority updates during idle."""
        deliverable = sorted(self._pending_deliveries, key=lambda u: u.priority)
        self._pending_deliveries.clear()
        for update in deliverable:
            if msg := self._format_update(update):
                await self.session.generate_reply(instructions=msg)
        self.state.mark_idle()

    def _format_update(self, update: StateUpdate) -> str | None:
        """Format a StateUpdate for speech delivery. Override for custom formatting."""
        match update.status:
            case TaskStatus.COMPLETED:
                return update.payload.get("message", "")
            case TaskStatus.FAILED:
                return f"Ha ocurrido un error: {update.payload.get('error', 'desconocido')}"
            case _:
                return None


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULE 3: e_agents/rtc/agents/triage.py                                   ║
# ║  Concrete triage agents. Pure domain logic, no framework plumbing.          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝



# ─── Prompts ─────────────────────────────────────────────────────────────────

_TRIAGE_PROMPT = (
    "Eres el agente de Triaje de una clínica médica. Eres el primer punto de contacto.\n"
    "Tu trabajo es entender la necesidad del paciente y derivarlo:\n"
    "- Servicios médicos, citas, síntomas → transferir a soporte\n"
    "- Seguros, facturación, pagos, reclamaciones → transferir a facturación\n"
    "Haz preguntas de clarificación si es necesario. Sé cálido, profesional y conciso.\n"
    "Hablas SIEMPRE en español."
)

_SUPPORT_PROMPT = (
    "Eres el agente de Soporte al Paciente. Te encargas de:\n"
    "- Agendamiento y reagendamiento de citas\n"
    "- Pre-evaluación de síntomas y derivación médica\n"
    "- Consultas generales sobre servicios médicos\n"
    "Si el paciente necesita ayuda con facturación, transfiere a facturación.\n"
    "Si la consulta está fuera de tu alcance, transfiere de vuelta a triaje.\n"
    "Hablas SIEMPRE en español."
)

_BILLING_PROMPT = (
    "Eres el agente de Facturación Médica. Te encargas de:\n"
    "- Verificación de seguros y estado de reclamaciones\n"
    "- Planes de pago y saldos pendientes\n"
    "- Disputas y ajustes de facturación\n"
    "Si el paciente necesita servicios médicos, transfiere a soporte.\n"
    "Si la consulta está fuera de tu alcance, transfiere de vuelta a triaje.\n"
    "Hablas SIEMPRE en español."
)


# ─── Agents ──────────────────────────────────────────────────────────────────

class TriageAgent(ReactiveAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions=_TRIAGE_PROMPT,
        )

    @function_tool()
    async def transferir_a_soporte(self, context: RunContext[SessionState]) -> Agent:
        """Transfiere al paciente a Soporte para servicios médicos."""
        await self.session.say("Te transfiero con nuestro equipo de Soporte al Paciente.")
        return await self._transfer_to("support", context)

    @function_tool()
    async def transferir_a_facturacion(self, context: RunContext[SessionState]) -> Agent:
        """Transfiere al paciente a Facturación para seguros y pagos."""
        await self.session.say("Te transfiero con nuestro departamento de Facturación.")
        return await self._transfer_to("billing", context)


class SupportAgent(ReactiveAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions=_SUPPORT_PROMPT,
        )

    @function_tool()
    async def transferir_a_triaje(self, context: RunContext[SessionState]) -> Agent:
        """Transfiere de vuelta a triaje para re-derivación."""
        await self.session.say("Te transfiero de vuelta con nuestro agente de Triaje.")
        return await self._transfer_to("triage", context)

    @function_tool()
    async def transferir_a_facturacion(self, context: RunContext[SessionState]) -> Agent:
        """Transfiere a Facturación para seguros y pagos."""
        await self.session.say("Te transfiero con nuestro departamento de Facturación.")
        return await self._transfer_to("billing", context)


class BillingAgent(ReactiveAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions=_BILLING_PROMPT,
        )

    @function_tool()
    async def transferir_a_triaje(self, context: RunContext[SessionState]) -> Agent:
        """Transfiere de vuelta a triaje para re-derivación."""
        await self.session.say("Te transfiero de vuelta con nuestro agente de Triaje.")
        return await self._transfer_to("triage", context)

    @function_tool()
    async def transferir_a_soporte(self, context: RunContext[SessionState]) -> Agent:
        """Transfiere a Soporte para servicios médicos."""
        await self.session.say("Te transfiero con nuestro equipo de Soporte al Paciente.")
        return await self._transfer_to("support", context)


# ─── Server Entrypoint ──────────────────────────────────────────────────────

server = AgentServer()



# Custom policy: voice-optimized (transfers are HIGH, searches NORMAL)
_TRIAGE_POLICY = InterruptionPolicy(rules={
    Priority.CRITICAL: InterruptionStrategy.IMMEDIATE,
    Priority.HIGH: InterruptionStrategy.IMMEDIATE,
    Priority.NORMAL: InterruptionStrategy.TURN_BOUNDARY,
    Priority.LOW: InterruptionStrategy.NATURAL_PAUSE,
    Priority.BACKGROUND: InterruptionStrategy.ENQUEUE,
})


@server.rtc_session(agent_name="triage")
async def entrypoint(ctx: agents.JobContext) -> None:
    triage = TriageAgent()
    support = SupportAgent()
    billing = BillingAgent()

    state = SessionState(
        agents={"triage": triage, "support": support, "billing": billing},
        policy=_TRIAGE_POLICY,
        max_concurrency=5,
    )

    session = AgentSession[SessionState](
        userdata=state,
        llm=google.LLM(model="gemini-2.0-flash"),
        max_tool_steps=5,
        allow_interruptions=True,
        min_endpointing_delay=0.5,
        max_endpointing_delay=3.0,
    )
    await session.start(agent=triage, room=ctx.room)


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "console", "--text"]
    cli.run_app(server)
