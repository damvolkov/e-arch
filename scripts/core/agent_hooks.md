# Agent Hooks

Hooks que se implementan como métodos async en la subclase `Agent`. LiveKit los invoca automáticamente.

---

## 1. Hooks de Ciclo de Vida

### `on_enter()` — Activación del agente

Se ejecuta cuando el agente se activa (carga inicial o target de un handoff).

```python
class GreeterAgent(Agent):
    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="Greet the user warmly and ask how you can help."
        )
```

### `on_exit()` — Desactivación del agente

Se ejecuta cuando el agente va a ser reemplazado por otro.

```python
class GreeterAgent(Agent):
    async def on_exit(self) -> None:
        await self.session.generate_reply(
            instructions="Tell the user a friendly goodbye before you exit."
        )
```

### `on_user_turn_completed()` — Pre-LLM hook

Se ejecuta **ANTES** de que el LLM genere respuesta. Para RAG injection, validación de input, guardrails.

```python
class MyAgent(Agent):
    async def on_user_turn_completed(
        self, turn_ctx: ChatContext, new_message: ChatMessage,
    ) -> None:
        user_text = new_message.text_content or ""

        # RAG injection
        docs = await vector_search(user_text)
        turn_ctx.add_message(role="assistant", content=f"Context: {docs}")
```

Bloquear respuesta con `StopResponse`:

```python
from livekit.agents import StopResponse

class MyAgent(Agent):
    async def on_user_turn_completed(self, turn_ctx, new_message) -> None:
        if not new_message.text_content:
            raise StopResponse()  # Aborta la respuesta del LLM
```

---

## 2. Pipeline Node Overrides

Interceptan el flujo VAD → STT → LLM → TTS.

### `stt_node()` — Preprocesado de audio / filtrado de transcripción

```python
class MyAgent(Agent):
    async def stt_node(self, audio, model_settings):
        async for event in Agent.default.stt_node(self, audio, model_settings):
            yield event
```

### `llm_node()` — Guardrail del output del LLM / moderación

```python
class MyAgent(Agent):
    async def llm_node(self, chat_ctx, tools, model_settings):
        async def process_stream():
            buffer = ""
            async with self.session.llm.chat(chat_ctx=chat_ctx, tools=tools) as stream:
                async for chunk in stream:
                    content = getattr(chunk.delta, "content", None)
                    if content:
                        buffer += content
                        if await is_safe(buffer):
                            yield chunk
                        else:
                            yield "I can't respond to that."
                            return
        return process_stream()
```

### `tts_node()` — Filtrado / reemplazo del speech output

```python
class MyAgent(Agent):
    async def tts_node(self, text, model_settings):
        async def filtered():
            async for chunk in text:
                yield chunk.replace("CompetitorX", "another provider")

        async for frame in Agent.default.tts_node(self, filtered(), model_settings):
            yield frame
```

### `transcription_node()` — Limpieza del transcript antes de mostrarlo al usuario

```python
class MyAgent(Agent):
    async def transcription_node(self, text, model_settings):
        async for delta in text:
            yield delta.replace("😘", "")
```

---

## 3. Instrucciones Dinámicas en Runtime

### `update_instructions()` — Reemplazar prompt base (persistente, NO en historial)

```python
await agent.update_instructions(
    "You are now in support mode. Help troubleshoot technical issues."
)
```

### `update_chat_ctx()` — Inyectar en contexto de conversación (persistente, SÍ en historial)

```python
ctx = self.chat_ctx.copy()
ctx.add_message(role="system", content=f"[knowledge_base]\n{await search_docs(query)}")
await self.update_chat_ctx(ctx)
```

---

## 4. Control de Interrupción desde el Agente

### `StopResponse` — Abortar respuesta del LLM

Solo dentro de `on_user_turn_completed`. Previene que el LLM genere respuesta.

```python
from livekit.agents import StopResponse

async def on_user_turn_completed(self, turn_ctx, new_message) -> None:
    if is_garbage(new_message.text_content):
        raise StopResponse()
```

---

## 5. Handoff / Swap de Agentes

### Tool-based handoff — El LLM decide la transferencia

Retornar un `Agent` (o tupla `(Agent, str)`) desde `@function_tool`.

```python
class GreeterAgent(Agent):
    @function_tool()
    async def route_to_billing(self, context: RunContext):
        """Transfer to billing specialist."""
        return BillingAgent(chat_ctx=self.chat_ctx), "Transferring to billing"
```

Handoff condicional (sin handoff si la condición falla):

```python
@function_tool()
async def confirm_checkout(self, context: RunContext) -> str | tuple[Agent, str]:
    """Confirm the checkout."""
    if not context.userdata.order:
        return "No order found."  # tool result, sin handoff
    return CheckoutAgent(), "Proceeding to checkout."  # handoff
```

### Swap programático — Reemplazo directo sin tool

```python
session.update_agent(NewAgent())
```
