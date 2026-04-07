# Framework Hook Mappings

How to bridge each agent framework's hooks into AEDA signals.

## Universal Hook Taxonomy

All major frameworks expose the same lifecycle points. The table below shows the convergence:

| Lifecycle Point | Pydantic-AI | OpenAI Agents SDK | Google ADK | LangGraph | LiveKit | AEDA Signal |
|---|---|---|---|---|---|---|
| Agent start | `before_run` | `on_agent_start` | `before_agent_callback` | Node entry | `on_enter` | `AGENT_STARTED` |
| Agent end | `after_run` | `on_agent_end` | `after_agent_callback` | Node exit | `on_exit` | `AGENT_ENDED` |
| LLM request | `before_model_request` | `on_llm_start` | `before_model_callback` | — | — | `MODEL_REQUESTED` |
| LLM response | `after_model_request` | `on_llm_end` | `after_model_callback` | — | — | `MODEL_RESPONDED` |
| Tool start | `before_tool_execute` | `on_tool_start` | `before_tool_callback` | — | — | `TOOL_CALLED` |
| Tool end | `after_tool_execute` | `on_tool_end` | `after_tool_callback` | — | — | `TOOL_COMPLETED` |
| Handoff | — | `on_handoff` | `transfer_to_agent` | Edge transition | `update_agent` | `HANDOFF_OUT` |
| Error | `on_*_error` | — | — | — | — | `AGENT_ERROR` |

## Pydantic-AI Bridge

Pydantic-AI uses **Capabilities** (formerly hooks). Four variants per phase:
- `before_*` — observe (FIFO order)
- `after_*` — modify result (LIFO order)
- `wrap_*` — full middleware/AOP (nested)
- `on_*_error` — error recovery

### Signal Mapping

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability

from aeda.engine import Reactor
from aeda.models import Event, Signal


class AEDACapability(AbstractCapability):
    """Pydantic-AI capability that emits AEDA signals."""

    def __init__(self, reactor: Reactor) -> None:
        self._reactor = reactor

    # Agent lifecycle
    async def before_run(self, ctx):
        self._reactor.emit(Event(signal=Signal.AGENT_STARTED, payload={"agent_id": ctx.agent.name}))

    async def after_run(self, ctx, result):
        self._reactor.emit(Event(signal=Signal.AGENT_ENDED))

    # Model lifecycle
    async def before_model_request(self, ctx, messages):
        self._reactor.emit(Event(signal=Signal.MODEL_REQUESTED))

    async def after_model_request(self, ctx, response):
        self._reactor.emit(Event(signal=Signal.MODEL_RESPONDED))

    # Tool lifecycle
    async def before_tool_execute(self, ctx, tool_name, args):
        self._reactor.emit(Event(
            signal=Signal.TOOL_CALLED,
            payload={"tool": tool_name, "args": args},
        ))

    async def after_tool_execute(self, ctx, tool_name, result):
        self._reactor.emit(Event(
            signal=Signal.TOOL_COMPLETED,
            payload={"tool": tool_name, "result": result},
        ))

    # Tool validation
    async def before_tool_validate(self, ctx, tool_name, raw_args):
        pass  # No AEDA signal — internal to framework

    async def after_tool_validate(self, ctx, tool_name, validated_args):
        pass  # No AEDA signal — internal to framework
```

### SessionHandle for Pydantic-AI

```python
class PydanticAIHandle:
    """Bridges AEDA commands to pydantic-ai operations."""

    def __init__(self, agent: Agent, deps: Any = None) -> None:
        self._agent = agent
        self._deps = deps
        self._current_run = None

    async def respond(self, *, instructions: str) -> None:
        # Run agent with ephemeral instruction prepended
        self._current_run = await self._agent.run(instructions, deps=self._deps)

    async def say(self, *, text: str) -> None:
        # Direct output (no LLM)
        print(text)

    async def interrupt(self) -> None:
        # Cancel current run if active
        if self._current_run:
            self._current_run = None

    async def enrich(self, *, data: dict) -> None:
        # Add to deps or context
        pass

    async def set_prompt(self, *, prompt: str) -> None:
        # Update agent system prompt
        self._agent._system_prompts = [prompt]

    async def swap(self, *, agent_id: str) -> None:
        # Agent handoff — framework-specific
        pass
```

## OpenAI Agents SDK Bridge

Uses `RunHooks` (global) and `AgentHooks` (per-agent) subclasses.

```python
from agents import RunHooks, RunContextWrapper

class AEDARunHooks(RunHooks):
    def __init__(self, reactor: Reactor) -> None:
        self._reactor = reactor

    async def on_agent_start(self, context: RunContextWrapper, agent) -> None:
        self._reactor.emit(Event(signal=Signal.AGENT_STARTED, payload={"agent_id": agent.name}))

    async def on_agent_end(self, context: RunContextWrapper, agent, output) -> None:
        self._reactor.emit(Event(signal=Signal.AGENT_ENDED))

    async def on_llm_start(self, context: RunContextWrapper, agent, model_settings) -> None:
        self._reactor.emit(Event(signal=Signal.MODEL_REQUESTED))

    async def on_llm_end(self, context: RunContextWrapper, agent, response) -> None:
        self._reactor.emit(Event(signal=Signal.MODEL_RESPONDED))

    async def on_tool_start(self, context: RunContextWrapper, agent, tool) -> None:
        self._reactor.emit(Event(signal=Signal.TOOL_CALLED, payload={"tool": tool.name}))

    async def on_tool_end(self, context: RunContextWrapper, agent, tool, result) -> None:
        self._reactor.emit(Event(signal=Signal.TOOL_COMPLETED, payload={"tool": tool.name}))

    async def on_handoff(self, context: RunContextWrapper, from_agent, to_agent) -> None:
        self._reactor.emit(Event(signal=Signal.HANDOFF_OUT, payload={"agent_id": to_agent.name}))
```

## Google ADK Bridge

Uses callback functions that receive `CallbackContext` with session state access.
Callbacks can **block execution** by returning a value instead of `None`.

```python
def aeda_before_agent(callback_context) -> None:
    reactor.emit(Event(signal=Signal.AGENT_STARTED))

def aeda_after_agent(callback_context) -> None:
    reactor.emit(Event(signal=Signal.AGENT_ENDED))

def aeda_before_model(callback_context) -> None:
    reactor.emit(Event(signal=Signal.MODEL_REQUESTED))

def aeda_after_model(callback_context) -> None:
    reactor.emit(Event(signal=Signal.MODEL_RESPONDED))

def aeda_before_tool(callback_context, tool_name, args) -> None:
    reactor.emit(Event(signal=Signal.TOOL_CALLED, payload={"tool": tool_name}))

def aeda_after_tool(callback_context, tool_name, result) -> None:
    reactor.emit(Event(signal=Signal.TOOL_COMPLETED, payload={"tool": tool_name}))

# Register on ADK Agent:
agent = Agent(
    before_agent_callback=aeda_before_agent,
    after_agent_callback=aeda_after_agent,
    before_model_callback=aeda_before_model,
    after_model_callback=aeda_after_model,
    before_tool_callback=aeda_before_tool,
    after_tool_callback=aeda_after_tool,
)
```

## LangGraph Bridge

Uses wrap hooks on nodes and edges. State is a `TypedDict` with `Annotated` reducers.

```python
from langgraph.graph import StateGraph

def aeda_node_hook(state, node_name):
    reactor.emit(Event(signal=Signal.AGENT_THINKING, payload={"node": node_name}))

graph = StateGraph(AgentState)
graph.add_wrap_call_node_hook(aeda_node_hook)
```

## LiveKit Agents Bridge

LiveKit uses session events + agent hooks. This is the pattern from the e-agents project.

```python
class LiveKitHandle:
    """Wraps LiveKit AgentSession."""

    def __init__(self, session) -> None:
        self._session = session

    async def respond(self, *, instructions: str) -> None:
        self._session.generate_reply(instructions=instructions)

    async def say(self, *, text: str) -> None:
        self._session.say(text, add_to_chat_ctx=True)

    async def interrupt(self) -> None:
        self._session.interrupt()

    async def enrich(self, *, data: dict) -> None:
        self._session.update_chat_ctx(...)

    async def set_prompt(self, *, prompt: str) -> None:
        self._session.agent.update_instructions(prompt)

    async def swap(self, *, agent_id: str) -> None:
        self._session.update_agent(agents_registry[agent_id])
```

### Wiring LiveKit Events → AEDA Signals

```python
@session.on("agent_started_speaking")
def on_speaking():
    reactor.emit(Event(signal=Signal.AGENT_RESPONDING))

@session.on("agent_stopped_speaking")
def on_stopped():
    reactor.emit(Event(signal=Signal.AGENT_IDLE))

@session.on("user_started_speaking")
def on_user():
    reactor.emit(Event(signal=Signal.STATE_CHANGED, payload={"user": "speaking"}))

@session.on("user_stopped_speaking")
def on_user_stop():
    reactor.emit(Event(signal=Signal.STATE_CHANGED, payload={"user": "silent"}))
    state.register_turn()
```

## CrewAI Bridge

CrewAI Flows (2026) use `EventListener` for event-driven orchestration.

```python
from crewai.flow.flow import Flow, listen

class AEDAFlow(Flow):
    @listen("task_started")
    async def on_task_start(self, event):
        reactor.emit(Event(signal=Signal.AGENT_STARTED))

    @listen("task_completed")
    async def on_task_done(self, event):
        reactor.emit(Event(signal=Signal.AGENT_ENDED))
```

## AutoGen v0.4 Bridge

AutoGen uses the actor model with async message passing. Each agent has its own state.

```python
# AutoGen agents communicate via messages
# Bridge by wrapping the agent's message handler
class AEDAAgentWrapper:
    async def on_message(self, message, ctx):
        reactor.emit(Event(signal=Signal.MODEL_REQUESTED))
        result = await self._inner.on_message(message, ctx)
        reactor.emit(Event(signal=Signal.MODEL_RESPONDED))
        return result
```

## Key Takeaway

The bridge pattern is always the same:

1. **Hook into framework lifecycle** — subscribe to before/after events
2. **Translate to AEDA Signal** — map framework event → canonical Signal enum
3. **Call reactor.emit()** — sync, non-blocking, safe from any context
4. **Implement SessionHandle** — map AEDA Commands → framework operations

The Reactor handles everything else: state mutation, policy evaluation, decision execution, subscriber notification.
