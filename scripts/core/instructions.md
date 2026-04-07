# Reactive Session Architecture

## 1. Overview

We are building a **reactive session layer** for multi-agent voice/text systems. The core idea is the **double-loop pattern**:

- **Outer loop**: A front agent always attends the user. Never blocks. Never goes silent.
- **Inner loop**: Background agents, tools, and tasks run in parallel. Their results flow into a shared reactive state.
- **Reactor**: An async monitor watches the state. When something changes, it evaluates policies and issues commands — interrupt the agent, enrich context, swap agents, flush results, etc.

The architecture is **framework-agnostic**. The core models and reactor have zero dependency on LiveKit, OpenAI, Google ADK, or any other framework. A thin integration layer bridges the reactive system with the framework's hooks.

**Current target framework**: LiveKit Agents SDK (Python).

---

## 2. Taxonomy

Strict naming rules. Every class, enum value, and method follows these conventions.

### 2.1 Actors

| Actor | What it is |
|-------|-----------|
| **Agent** | An LLM-powered participant in the session |
| **User** | The human interacting with the system |
| **Session** | The overall conversation lifecycle |
| **Task** | A background async work unit |

### 2.2 Naming patterns

| Concept | Suffix | Example | Meaning |
|---------|--------|---------|---------|
| Ongoing state | `Mode` | `AgentMode`, `UserMode`, `SessionMode` | What IS right now (has duration) |
| Passive occurrence | `Signal` | `Signal.USER_SPOKE` | What HAPPENED (observed by hooks) |
| Active instruction | `Command` | `Command.INTERRUPT_SAY` | What TO DO (issued by reactor) |
| Signal instance | `Event` | `Event(signal=..., payload=...)` | Frozen signal with data and timestamp |
| Command instance | `Decision` | `Decision(command=..., payload=...)` | Frozen command with data |
| Actor snapshot | `XState` | `AgentState`, `UserState` | Live mutable state of an actor |
| Full world model | `ReactiveState` | `ReactiveState` | Everything the reactor reads |

### 2.3 Passive vs Active

This is the most important distinction in the system:

- **Signal** (passive): Emitted by framework hooks when something is **observed**. The system didn't cause it — it just happened. Example: `USER_BARGED_IN`, `AGENT_DONE`, `TASK_DONE`.
- **Command** (active): Issued by the reactive engine when it **decides** to act. The system is causing something. Example: `INTERRUPT`, `SWAP`, `ENRICH`.

Signals flow **into** the reactor. Commands flow **out**. They never mix.

### 2.4 Method verbs

One verb per semantic field. Used consistently everywhere:

| Verb | Meaning | Used for |
|------|---------|----------|
| `emit` | Fire a signal into the reactor | `emit(Event(...))` |
| `evaluate` | Read state + event, produce decisions | Reactor core |
| `apply` | Mutate state in response to a signal | State transitions |
| `execute` | Carry out a command via framework | Integration layer |
| `format` | Transform data for user delivery | Event → text |
| `register` | Record a compound state change | `register_handoff()`, `register_turn()` |

---

## 3. Data Models (`reactive/models.py`)

Pure Python. No async. No framework deps. No I/O. Uses `dataclasses` with `slots=True`.

### 3.1 Modes (sustained operational state)

```python
class AgentMode(StrEnum):
    IDLE        # No active work
    LISTENING   # Waiting for user input
    THINKING    # LLM generating
    SPEAKING    # TTS playing audio

class UserMode(StrEnum):
    SPEAKING    # VAD detected voice
    SILENT      # Short pause after speaking
    IDLE        # Extended silence (> threshold)
    AWAY        # Disconnected / absent

class SessionMode(StrEnum):
    STARTING    # Initializing
    ACTIVE      # Normal operation
    PAUSED      # Temporarily suspended
    ENDING      # Teardown in progress
```

### 3.2 Signals (passive — what happened)

Grouped by source actor. Each signal has a clear hook point in the framework.

```
USER_SPOKE          — VAD onset
USER_STOPPED        — VAD offset
USER_IDLE           — Silence > threshold
USER_LEFT           — Disconnect / timeout
USER_BACK           — Returned after IDLE/AWAY
USER_BARGED_IN      — Spoke while agent SPEAKING

AGENT_THINKING      — LLM call started
AGENT_SPOKE         — TTS playback started
AGENT_DONE          — TTS finished / response complete
AGENT_CUT_OFF       — Speech interrupted
AGENT_TOOL_CALL     — Function tool invoked
AGENT_TOOL_RESULT   — Tool returned

HANDOFF_OUT         — Current agent initiated transfer
HANDOFF_IN          — New agent became active

TASK_SENT           — Background task launched
TASK_DONE           — Completed with results
TASK_FAILED         — Errored
TASK_TIMEOUT        — Exceeded deadline

SESSION_STARTED     — Init complete
SESSION_ENDING      — Teardown begins
TICK                — Periodic heartbeat
```

### 3.3 Commands (active — what to do)

Issued by the reactor. Executed by the framework integration layer.

```
SAY                 — Speak specific text verbatim
REPLY               — Trigger LLM to generate a response
INTERRUPT           — Stop agent's current speech
INTERRUPT_SAY       — Stop speech + say text
INTERRUPT_REPLY     — Stop speech + trigger LLM

ENRICH              — Push data into conversation thread silently
SET_PROMPT          — Change agent's system instructions

SWAP                — Force handoff to a different agent

FLUSH               — Deliver all pending task results now
CANCEL_TASK         — Cancel a running background task

HOLD                — Explicit no-op / defer
```

### 3.4 Data structures

```python
@dataclass(frozen=True, slots=True)
class Event:
    signal: Signal
    payload: dict[str, Any]
    timestamp: float

@dataclass(frozen=True, slots=True)
class Decision:
    command: Command
    payload: dict[str, Any]
```

### 3.5 Sub-states

`AgentState` and `UserState` are mutable snapshots. Same `AgentState` model is used for both `current` and `previous` — when a handoff occurs, `current` is copied to `previous` as-is.

**AgentState fields:**
- Identity: `name`, `ref` (framework agent instance)
- Mode: `mode` (AgentMode)
- Timestamps: `entered_at`, `spoke_at`, `finished_at`, `thought_at`, `tool_called_at`
- Counters: `turns`, `tool_calls`, `interruptions`
- Derived: `active_for`, `silent_for`

**UserState fields:**
- Mode: `mode` (UserMode)
- Timestamps: `spoke_at`, `stopped_at`, `active_at`
- Counters: `interrupts`, `messages`
- Derived: `silent_for`, `inactive_for`

### 3.6 ReactiveState (the complete world model)

```python
@dataclass(slots=True)
class ReactiveState:
    current: AgentState       # Active agent (live, updating)
    previous: AgentState      # Previous agent (frozen at handoff)
    agents: dict[str, Any]    # Registry: name → framework ref
    user: UserState           # User activity snapshot
    session: SessionMode      # Lifecycle phase
    started_at: float         # Session start
    last_turn_at: float       # Last completed turn
    turn_count: int
    handoff_count: int
    tasks_running: int
    tasks_pending: int        # Results waiting to be delivered
    data: dict[str, Any]      # Open k/v store for domain data
```

Compound operations:
- `register_handoff(name)` — copies current → previous, inits new current
- `register_turn()` — increments counters, updates timestamps

---

## 4. Architecture Modules

Build order matters. Each module depends only on those above it.

### 4.1 `reactive/models.py` ✅ DONE
Pure data models. Enums, dataclasses, zero deps.

### 4.2 `reactive/engine.py` 🔲 TODO
The async reactor. Core responsibilities:
- Owns an `asyncio.Queue[Event]` for incoming signals
- Runs a monitor loop: `wait for event → apply(event, state) → evaluate(event, state) → execute(decisions)`
- `apply()` is a pure function: given (event, state), mutates state deterministically (update modes, timestamps, counters)
- `evaluate()` consults policies: given (event, state), produces `list[Decision]`
- `execute()` dispatches each Decision to the framework integration layer
- Manages background tasks (submit, cancel, track)
- Handles TICK generation on a configurable interval

Key design:
- The engine does NOT know about LiveKit/OpenAI/etc.
- It calls abstract methods that the integration layer implements
- Policies are pluggable functions: `(Event, ReactiveState) -> list[Decision]`

### 4.3 `reactive/policies.py` 🔲 TODO
Default policy functions. Each is a pure function `(Event, ReactiveState) -> list[Decision]`.
Examples:
- Task delivery policy: when `TASK_DONE` arrives and agent is `IDLE`, issue `SAY`. If agent is `SPEAKING`, issue `HOLD` and increment pending.
- Barge-in policy: on `USER_BARGED_IN`, issue `INTERRUPT` + log.
- Idle policy: on `TICK`, if `user.silent_for > N` and `has_pending`, issue `FLUSH`.
- Escalation policy: if `user.interrupts > 3`, switch delivery strategy.

### 4.4 `integrations/livekit.py` 🔲 TODO
LiveKit-specific bridge. Responsibilities:
- Subclass or mixin that hooks into LiveKit's `Agent` lifecycle
- Maps LiveKit hooks → `emit(Event(...))`:
  - `on_enter()` → `emit(HANDOFF_IN)`
  - VAD events → `emit(USER_SPOKE)`, `emit(USER_STOPPED)`, etc.
  - TTS events → `emit(AGENT_SPOKE)`, `emit(AGENT_DONE)`, etc.
- Maps Commands → LiveKit operations:
  - `SAY` → `session.say(text)`
  - `REPLY` → `session.generate_reply()`
  - `INTERRUPT` → cancel TTS playback
  - `SWAP` → return agent ref from function_tool
  - `ENRICH` → `update_chat_ctx()` with system message

### 4.5 Concrete agents 🔲 TODO
Pure domain logic. Each agent defines:
- System prompt (instructions)
- Tools (`@function_tool`)
- Optional `format_event()` override for domain-specific delivery
- Optional custom policy rules

---

## 5. Signal → State mapping

When a signal arrives, the engine calls `apply(event, state)` BEFORE evaluating policies. This is deterministic state mutation:

```
USER_SPOKE          → user.mode = SPEAKING, user.spoke_at = now, user.messages++, user.active_at = now
USER_STOPPED        → user.mode = SILENT, user.stopped_at = now
USER_IDLE           → user.mode = IDLE
USER_LEFT           → user.mode = AWAY
USER_BACK           → user.mode = SILENT, user.active_at = now
USER_BARGED_IN      → user.interrupts++, user.active_at = now

AGENT_THINKING      → current.mode = THINKING, current.thought_at = now
AGENT_SPOKE         → current.mode = SPEAKING, current.spoke_at = now
AGENT_DONE          → current.mode = IDLE, current.finished_at = now
AGENT_CUT_OFF       → current.mode = IDLE, current.interruptions++, current.finished_at = now
AGENT_TOOL_CALL     → current.tool_calls++, current.tool_called_at = now
AGENT_TOOL_RESULT   → (payload stored in event, policy decides what to do)

HANDOFF_OUT         → (no state change — outgoing agent's hooks handle it)
HANDOFF_IN          → register_handoff(event.payload["name"])

TASK_SENT           → tasks_running++
TASK_DONE           → tasks_running--, tasks_pending++
TASK_FAILED         → tasks_running--
TASK_TIMEOUT        → tasks_running--

TURN_STARTED        → (captured by USER_SPOKE / endpointing)
TURN_ENDED          → register_turn()

SESSION_STARTED     → session = ACTIVE
SESSION_ENDING      → session = ENDING

TICK                → (no state change — policies use timing derived properties)
```

---

## 6. Build Plan

Incremental. Each step produces working, testable code.

### Step 1: Models ✅
`reactive/models.py` — enums, dataclasses, state. No deps.

### Step 2: Engine core
`reactive/engine.py` — the reactor loop, apply(), policy evaluation. Async, no framework deps.

### Step 3: Default policies
`reactive/policies.py` — task delivery, barge-in, idle flush, escalation. Pure functions.

### Step 4: LiveKit bridge
`integrations/livekit.py` — hooks → signals, commands → operations.

### Step 5: First agent pair
Attendant (outer) + Reasoner (inner). Test the full loop.

### Step 6: Iterate
Add policies, tune thresholds, add more agent types.

---

## 7. Constraints

- **Python ≥ 3.13**. Use `match-case`, walrus operator, modern type syntax.
- **Async everywhere**. The engine and integration layer are fully async.
- **No relative imports**. Absolute only.
- **No imports inside functions**.
- **Dataclasses for models** (`slots=True`). Pydantic only where validation is needed.
- **`StrEnum`** for all enums (human-readable serialization).
- **One file = one concern**. Models, engine, policies, integration are separate files.
- **Pure functions for policies**. `(Event, ReactiveState) -> list[Decision]`. No side effects.
- **Timestamps, not durations**. Store `spoke_at`, derive `silent_for` via properties.
- **Same model for current/previous agent**. `AgentState` used for both. Handoff copies current → previous.

---

## 8. Testing Strategy

- **Models**: Unit tests for `register_handoff()`, `register_turn()`, derived properties.
- **Apply**: Parameterized tests — given (signal, initial_state), assert final_state.
- **Policies**: Parameterized tests — given (event, state), assert decisions.
- **Engine**: Integration test with mock framework layer. Emit events, assert commands.
- **LiveKit bridge**: Integration test with `AgentSession` mock / console mode.

---

## 9. Files to create

```
reactive/
  __init__.py          — (empty or minimal re-exports)
  models.py            — Enums, dataclasses, ReactiveState        ← DONE
  engine.py            — ReactiveEngine (async reactor loop)
  policies.py          — Default policy functions

integrations/
  __init__.py
  livekit.py           — LiveKit hooks ↔ signals/commands bridge

agents/
  __init__.py
  duo.py               — Attendant + Reasoner concrete agents
```