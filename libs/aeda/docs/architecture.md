# AEDA — Agentic Event-Driven Architecture

Framework-agnostic reactive engine for AI agents. Pure asyncio, zero dependencies.

## Core Insight

All agent frameworks converge on two integration surfaces:

1. **State / RunContext** — shared mutable object carrying session data, agent refs, counters.
2. **Hooks / Callbacks** — lifecycle interception points: before/after agent, model, tool, handoff.

AEDA plugs into both. It observes state changes through **Signals**, reacts through **Policies**, and executes through a framework-specific **SessionHandle**.

## Component Diagram

```
                    ┌──────────────────────────────────────────────┐
                    │              Framework Bridge                │
                    │  (pydantic-ai, openai-agents, adk, livekit) │
                    │                                              │
                    │  hooks → Signal events                       │
                    │  SessionHandle ← Command execution           │
                    └────────┬─────────────────────┬───────────────┘
                             │ emit()              │ handle.*()
                             ▼                     ▲
┌────────────────────────────────────────────────────────────────────┐
│                          REACTOR                                   │
│                                                                    │
│  ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐  │
│  │  Queue   │───▶│  Apply   │───▶│ Evaluate │───▶│   Execute    │  │
│  │ (events) │    │ (state)  │    │(policies)│    │  (handle)    │  │
│  └─────────┘    └──────────┘    └──────────┘    └──────────────┘  │
│       ▲                                               │            │
│       │              ┌───────────┐                    ▼            │
│       └──────────────│  Ticker   │         ┌──────────────┐       │
│       │              └───────────┘         │   EventBus   │       │
│       │                                    │ (subscribers) │       │
│       │  ┌──────────────┐                  └──────────────┘       │
│       └──│ TaskExecutor  │                        │                │
│          │ (fire&forget) │                        ▼                │
│          └──────────────┘              external observers          │
│                                     (logging, telemetry, UI)      │
└────────────────────────────────────────────────────────────────────┘

                    ┌──────────────────────────────────────────────┐
                    │              ReactiveState                    │
                    │                                              │
                    │  AgentSnapshot (current + previous)          │
                    │  SessionMode, turn_count, handoff_count      │
                    │  TaskRecord map (in-flight tasks)            │
                    │  data: dict[str, Any] (domain-specific)      │
                    └──────────────────────────────────────────────┘
```

## Processing Flow

```
1. emit(event)           Sync, non-blocking. Framework hooks call this.
2. Queue pickup           Reactor loop consumes from asyncio.Queue.
3. _re_apply(event)       Deterministic state mutation via match-case on Signal.
4. Policy evaluation      Each policy sees UPDATED state + event → tuple[Decision].
5. _re_execute(decision)  Calls SessionHandle methods (respond, interrupt, swap, ...).
6. bus.dispatch(event)    Notifies external subscribers (they see final state).
```

**Critical ordering**: State is mutated BEFORE policies evaluate. Policies see the world AFTER the event happened. External subscribers are notified AFTER policies have acted.

## Dual-Loop Architecture

AEDA implements the dual-loop pattern from the e-agents system:

```
 OUTER LOOP (Reactor)                    INNER LOOP (Agent Framework)
 ════════════════════                    ════════════════════════════

 Persistent — lives for entire session   Ephemeral — per-run/per-turn
 Observes all signals                    Tight LLM ↔ tool cycle
 Evaluates policies                      Model generates → tool → result → loop
 Manages background tasks                Framework-controlled
 Tracks state across turns               Hooks emit signals to outer loop

 Communicates via:                       Communicates via:
 ─ ReactiveState (shared)                ─ Signal events (to reactor)
 ─ Decisions (to framework)              ─ SessionHandle (from reactor)
```

**Inner loop** = the agent framework's native run cycle (model → tool → model → respond). AEDA observes it through hooks.

**Outer loop** = AEDA's Reactor. Persists across turns, runs policies, manages background tasks, detects patterns (idle, escalation, timeouts).

## Event Taxonomy

### Signals (passive — what happened)

| Category | Signals | State Effect |
|----------|---------|--------------|
| Agent | `AGENT_STARTED`, `AGENT_THINKING`, `AGENT_RESPONDING`, `AGENT_IDLE`, `AGENT_ENDED`, `AGENT_ERROR` | Mode transitions on AgentSnapshot |
| Tool | `TOOL_CALLED`, `TOOL_COMPLETED`, `TOOL_FAILED` | WAITING → THINKING, tool_calls++ |
| Model | `MODEL_REQUESTED`, `MODEL_RESPONDED` | THINKING mode, timestamps |
| Task | `TASK_SUBMITTED`, `TASK_COMPLETED`, `TASK_FAILED`, `TASK_TIMEOUT` | TaskRecord lifecycle, data merge |
| Handoff | `HANDOFF_OUT`, `HANDOFF_IN` | Freeze → new AgentSnapshot |
| Session | `SESSION_STARTED`, `SESSION_ENDING` | SessionMode transition |
| State | `STATE_CHANGED` | (custom — no built-in mutation) |
| Timing | `TICK` | (no mutation — drives time-based policies) |

### Commands (active — what to do)

| Command | SessionHandle Method | Purpose |
|---------|---------------------|---------|
| `RESPOND` | `respond(instructions=)` | Generate LLM response with ephemeral instruction |
| `SAY` | `say(text=)` | Emit predefined text (no LLM) |
| `INTERRUPT` | `interrupt()` | Stop current output/generation |
| `ENRICH` | `enrich(data=)` | Add context to conversation |
| `SET_PROMPT` | `set_prompt(prompt=)` | Update system prompt |
| `SWAP` | `swap(agent_id=)` | Hand off to another agent |
| `HOLD` | (internal) | Pause policy evaluation |
| `RESUME` | (internal) | Resume policy evaluation |

## Policies

Policies are sync functions: `evaluate(state, event) → tuple[Decision, ...]`

- **Sync** — fast, no I/O. Policies must never await.
- **Composable** — multiple policies evaluate independently, all matching execute.
- **Stateful** — may hold private state (cooldowns, flags).
- **Ordered** — evaluated in sequence, no short-circuit.

Built-in policies:

| Policy | Trigger | Behavior |
|--------|---------|----------|
| `IdlePolicy` | TICK | After N seconds of inactivity → SAY prompt |
| `TaskCompletedPolicy` | TASK_COMPLETED/FAILED | Interrupt + share results (with cooldown) |
| `EscalationPolicy` | AGENT_IDLE/MODEL_RESPONDED | After N turns → SET_PROMPT + SWAP |

## State Architecture

```python
ReactiveState
├── agent: AgentSnapshot         # Current agent (mode, timestamps, counters)
├── previous_agent: AgentSnapshot # Frozen snapshot from last handoff
├── agents: dict[str, Any]       # Agent registry (name → ref)
├── session: SessionMode         # STARTING → ACTIVE → PAUSED → ENDING
├── turn_count, handoff_count    # Aggregate counters
├── last_turn_at, last_activity_at  # Timing
├── tasks: dict[str, TaskRecord] # In-flight background tasks
└── data: dict[str, Any]         # Domain-specific data (task results, etc.)
```

**AgentSnapshot** tracks per-agent operational state:
- `mode`: IDLE → THINKING → RESPONDING → WAITING (tool) → back to THINKING
- Timestamps: `entered_at`, `responded_at`, `thought_at`, `tool_called_at`
- Counters: `turns`, `tool_calls`, `errors`
- Computed: `active_for`, `silent_for`

## Task Management

Background tasks are fire-and-forget async functions that emit lifecycle events:

```python
# Submit a background task — returns task_id immediately
task_id = await reactor.tasks.submit("weather", fetch_weather, city="Madrid")

# The TaskExecutor:
# 1. Creates TaskRecord in registry
# 2. Emits TASK_SUBMITTED event
# 3. Spawns asyncio.create_task(fn(**kwargs))
# 4. On completion: emits TASK_COMPLETED with result
# 5. On failure: emits TASK_FAILED with error

# Policies react to these events (e.g., TaskCompletedPolicy interrupts + shares)
```

## Integration Pattern

To integrate AEDA with any agent framework:

### 1. Implement SessionHandle

Map framework-specific operations to the protocol:

```python
class PydanticAIHandle:
    def __init__(self, agent, run_context): ...

    async def respond(self, *, instructions: str) -> None:
        # Use agent.run() with modified prompt
        ...

    async def interrupt(self) -> None:
        # Cancel current run
        ...
```

### 2. Create a Bridge

Translate framework hooks into AEDA signals:

```python
# In pydantic-ai: use Capabilities (before/after hooks)
class AEDACapability:
    def __init__(self, reactor: Reactor): ...

    async def before_model_request(self, ctx):
        self.reactor.emit(Event(signal=Signal.MODEL_REQUESTED))

    async def after_model_request(self, ctx, response):
        self.reactor.emit(Event(signal=Signal.MODEL_RESPONDED))

    async def before_tool_execute(self, ctx, tool_name, args):
        self.reactor.emit(Event(signal=Signal.TOOL_CALLED, payload={"tool": tool_name}))
```

### 3. Wire the Reactor

```python
state = ReactiveState()
handle = PydanticAIHandle(agent, ctx)
policies = (IdlePolicy(timeout=30), TaskCompletedPolicy())
reactor = Reactor(handle, state, policies, tick_interval=2.0)

# Subscribe observers
reactor.bus.subscribe_all(telemetry_handler)
reactor.bus.subscribe(Signal.AGENT_ERROR, alert_handler)

# Run
await reactor.run()
```

## Module Map

```
aeda/
├── models.py       Signal, Command, AgentMode, SessionMode, TaskState, Event, Decision
├── state.py        AgentSnapshot, TaskRecord, ReactiveState
├── protocols.py    SessionHandle, Policy, Subscriber
├── bus.py          EventBus, MemoryBackend
├── task.py         TaskRegistry, TaskExecutor
├── engine.py       Reactor
└── policies.py     IdlePolicy, TaskCompletedPolicy, EscalationPolicy
```

**Zero dependencies.** Pure Python 3.13 + asyncio. Framework bridges are separate packages.
