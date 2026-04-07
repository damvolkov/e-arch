# Session Hooks

Hooks registrados en `AgentSession` via `@session.on()` o callbacks directos.

---

## 1. Eventos de Sesión (`@session.on()`)

### `agent_state_changed` — Transiciones de estado del agente

Estados: `initializing` → `idle` → `listening` → `thinking` → `speaking`

```python
@session.on("agent_state_changed")
def on_agent_state(ev: AgentStateChangedEvent):
    if ev.new_state == "speaking":
        set_agent_speaking(True)
    elif ev.old_state == "speaking":
        set_agent_speaking(False)
    logger.info(f"Agent → {ev.new_state}")
```

### `user_state_changed` — Transiciones de estado del usuario

Estados: `speaking` / `listening` / `away`

```python
@session.on("user_state_changed")
def on_user_state(ev: UserStateChangedEvent):
    if ev.new_state == "away":
        asyncio.create_task(check_user_presence())
```

### `user_input_transcribed` — Texto raw del speech del usuario

```python
@session.on("user_input_transcribed")
def on_transcript(ev):
    if not ev.is_final:
        return
    logger.info(f"User said: {ev.transcript[:80]}")
```

### `conversation_item_added` — Nuevo item en la conversación

Se dispara cuando se añade un mensaje (user/agent), tool call, o tool result al historial.

```python
@session.on("conversation_item_added")
def on_item(ev):
    item = ev.item
    role = getattr(item, "role", "?")
    content = (item.text_content or "")[:60] if hasattr(item, "text_content") else ""
    logger.debug(f"Chat item: {role} - {content}")
```

### `function_tools_executed` — Tools ejecutados

Incluye detalles de la llamada y outputs.

```python
@session.on("function_tools_executed")
def on_tools_executed(ev):
    for call, output in ev.zipped():
        result = str(output.output)[:60] if output else "—"
        logger.info(f"Tool: {call.name}, Result: {result}, Handoff: {ev.has_agent_handoff}")
```

### `agent_false_interruption` — Falsa alarma de interrupción

El usuario empezó a hablar pero no era una interrupción real (VAD false positive). El agente puede retomar.

```python
@session.on("agent_false_interruption")
def on_false_interruption(ev):
    logger.debug(f"False interruption, resumed: {ev.resumed}")
```

### `error` — Error en la sesión

Puede ser recuperable o no.

```python
@session.on("error")
def on_error(ev: ErrorEvent):
    logger.error(f"Session error from {type(ev.source).__name__}: {ev.error}")

    if not ev.error.recoverable:
        session.say(
            "I'm having trouble. Let me transfer your call.",
            allow_interruptions=False,
        )
```

### `close` — Sesión cerrada

```python
@session.on("close")
def on_close(ev: CloseEvent):
    logger.info(f"Session closed, reason: {ev.reason}")
    for item in session.history.items:
        match item.type:
            case "message":
                print(f"{item.role}: {item.text_content}")
            case "function_call":
                print(f"Tool call: {item.name}({item.arguments})")
            case "function_call_output":
                print(f"Tool result: {item.output}")
```

### `metrics_collected` — Métricas de uso disponibles

Token counts, latencia, etc.

```python
usage_collector = metrics.UsageCollector()

@session.on("metrics_collected")
def on_metrics(ev: MetricsCollectedEvent):
    metrics.log_metrics(ev.metrics)
    usage_collector.collect(ev.metrics)
```

---

## 2. Control de Sesión

### `session.interrupt()` — Interrupción global

Para todo lo que esté haciendo el agente (speech, LLM generation).

```python
session.interrupt()
```

Event-driven:

```python
@session.on("user_input_transcribed")
def on_user_speech(ev):
    if "cancel" in ev.transcript.lower():
        session.interrupt()
```

### `session.say()` — Speech de texto predefinido (sin LLM)

```python
handle = session.say("Processing your request...")
await handle
```

Sin añadir al historial (filler):

```python
handle = session.say("Thinking...", add_to_chat_ctx=False)
```

No interrumpible:

```python
handle = session.say("Important message", allow_interruptions=False)
```

Con audio pre-sintetizado (bypass TTS):

```python
from livekit.agents.utils.audio import audio_frames_from_file

await session.say(
    "Your greeting",
    audio=audio_frames_from_file(path, sample_rate=24000, num_channels=1),
)
```

### `session.generate_reply()` — Speech generado por LLM

Instrucción efímera (no queda en historial):

```python
await session.generate_reply(
    instructions="Greet the user warmly and ask how you can help."
)
```

Simular input de usuario (queda en historial):

```python
await session.generate_reply(user_input="What's the weather today?")
```

Control de tools:

```python
await session.generate_reply(tool_choice="none")      # sin tools
await session.generate_reply(tool_choice="auto")       # default
await session.generate_reply(tool_choice="required")   # forzar tool
```

---

## 3. Speech Handle Callbacks

### `add_done_callback()` — Cuando el speech termina o es interrumpido

```python
handle = session.say("Hello, how can I help?")
handle.add_done_callback(lambda _: print("speech ended"))
```

### `wait_for_playout()` — Esperar a que termine la reproducción

```python
handle = session.say("Long explanation...")
await handle.wait_for_playout()

if handle.interrupted:
    logger.info("User interrupted the message")
else:
    logger.info("Message completed")
```

### Propiedades del SpeechHandle

```python
handle.interrupted  # bool — fue interrumpido?
handle.done()       # bool — terminó el playout?
```

---

## 4. Patrón Interrupt + Re-trigger

Patrón canónico para inyectar nueva información mid-conversation:

```python
# 1. Parar speech/generation actual
session.interrupt()

# 2. Inyectar contexto en historial
ctx = self.chat_ctx.copy()
ctx.add_message(role="system", content=f"[result] {data}")
await self.update_chat_ctx(ctx)

# 3. Re-trigger con instrucción efímera
await session.generate_reply(
    instructions="New information arrived. Share the results with the user."
)
```

---

## 5. Hooks a Nivel de Servidor

### `setup_fnc` — Warmup del proceso (una vez por proceso al arrancar)

Cargar modelos pesados, VAD, etc.

```python
server = agents.AgentServer()

def prewarm(proc: agents.JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

server.setup_fnc = prewarm
```

### `on_request` — Nuevo request recibido

```python
@server.rtc_session(
    agent_name="my-agent",
    on_request=lambda ctx: print(f"New request from {ctx.room.name}"),
)
async def entrypoint(ctx: agents.JobContext):
    session = AgentSession(...)
    await session.start(agent=..., room=ctx.room)
    await ctx.connect()
```

### `on_session_end` — Sesión terminada (cleanup, reportes)

```python
async def on_session_end(ctx: agents.JobContext) -> None:
    report = ctx.make_session_report()
    await save_report(json.dumps(report.to_dict(), indent=2))

@server.rtc_session(on_session_end=on_session_end)
async def entrypoint(ctx: agents.JobContext):
    session = AgentSession(...)
    await session.start(agent=..., room=ctx.room)
    await ctx.connect()
```

### `ctx.add_shutdown_callback()` — Cleanup en shutdown graceful

```python
async def log_usage():
    summary = usage_collector.get_summary()
    logger.info(f"Usage: {summary}")

ctx.add_shutdown_callback(log_usage)
```
