# Double Loop — Arquitectura Reactiva

## El modelo

```
┌──────────────────────────────────────────────────────────┐
│                      OUTER LOOP                           │
│                                                           │
│   La cara del agente. Siempre presente, siempre atento.   │
│   El usuario habla con el outer loop. Nunca sabe que      │
│   hay un inner loop. Como un bibliotecario que te         │
│   atiende, te habla, y a la vez manda a un ayudante       │
│   a buscar algo al almacén remoto.                        │
│                                                           │
│   Componentes:                                            │
│     ReactiveSession  → observa y alimenta el estado       │
│     ReactiveAgent    → conversa y decide intervenciones   │
│     ReactiveState    → estado compartido entre ambos      │
└────────────────────────┬─────────────────────────────────┘
                         │ lanza / monitorea
                         ▼
┌──────────────────────────────────────────────────────────┐
│                      INNER LOOP                           │
│                                                           │
│   Lo que ocurre under the hood. Background tasks,         │
│   heavy ops, búsquedas largas, handoffs a nodos           │
│   especializados. El usuario no espera — el outer         │
│   loop sigue conversando mientras esto procesa.           │
│                                                           │
│   Cuando termina, actualiza ReactiveState.                │
│   El outer loop lo detecta y actúa.                       │
└──────────────────────────────────────────────────────────┘
```

## Monitoreo — dónde se observan los eventos

En `ReactiveSession`, via `@session.on(...)`. Estos hooks se registran en
`__init__` y alimentan el `ReactiveState` con cada cambio. Solo observan.
No pueden intervenir en el pipeline del agente.

```python
class ReactiveSession(AgentSession[ReactiveState]):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.on("agent_state_changed")(self._rs_on_agent_state)   # → AgentState.mode
        self.on("user_state_changed")(self._rs_on_user_state)     # → UserState.mode
        self.on("user_input_transcribed")(self._rs_on_transcript) # → log transcripts
        self.on("function_tools_executed")(self._rs_on_tools)     # → tool call count
        self.on("close")(self._rs_on_close)                       # → session ending
```

Responsabilidad única: mantener `ReactiveState` como snapshot fiel de lo que
está ocurriendo en la sesión. Este estado es la fuente de verdad que consultan
ambos protocolos de intervención.

## Dos protocolos de comunicación outer ↔ inner

### Protocolo 1: Gate

**Dirección:** usuario → outer loop → (¿pasa o no al inner?)

El usuario dice algo. Antes de que el LLM procese, el outer loop evalúa
el `ReactiveState` y el contenido del mensaje. Decide: ¿dejo pasar al
flujo normal, o intercepto con una respuesta forzada?

Es un filtro de entrada. El bibliotecario escucha al usuario y antes de
ponerse a buscar, evalúa si lo que ha dicho requiere un tratamiento especial.

```
usuario habla → STT → [GATE: outer loop evalúa ReactiveState + mensaje]
                             │
                      pasa ──┤── intercepta
                      │             │
                      ▼             ▼
                    LLM          generate_reply (respuesta forzada)
                    (normal)     + raise StopResponse
```

**Cuándo ocurre:** solo cuando el usuario habla (boundary de turno).
El agente estaba escuchando, no hablando. No hay interrupción.

**Hook en LiveKit:** `Agent.on_user_turn_completed()`
Se ejecuta ANTES del LLM. Puede leer el estado, frenar el pipeline
(`StopResponse`) y forzar una respuesta alternativa (`generate_reply`).

**Orden crítico:**
```
generate_reply(instructions=...)   ← lanza la alternativa PRIMERO
raise StopResponse()               ← frena el pipeline normal DESPUÉS

⚠ NO usar session.interrupt() aquí — mata el propio generate_reply
```

**Aplicación en nuestro sistema:**
```python
class Parker(Agent):
    async def on_user_turn_completed(self, turn_ctx, new_message) -> None:
        state: ReactiveState = self.session.userdata

        # Evaluar condición sobre ReactiveState + mensaje
        if condicion_gate(state, new_message):
            self.session.generate_reply(instructions="Respuesta alternativa")
            raise StopResponse()

        # Sin intervención → flujo normal del LLM
        state.register_turn()
```

**Casos de uso:**

| Caso | Qué evalúa | Qué hace |
|------|-----------|----------|
| Wake word | "perico" en el transcript | Fuerza frase predefinida |
| Guardrail de input | Contenido prohibido en mensaje | Bloquea antes del LLM |
| Intent redirect | Palabra clave + estado de sesión | Redirige a flujo alternativo |
| Session expired | `state.session_duration > MAX` | Despedida forzada |
| Sentiment gate | Frustración detectada en historial | Cambia tono/estrategia |
| Cooldown | `state.current.turns > N` sin resultado | Sugiere pausa |

---

### Protocolo 2: Callback

**Dirección:** inner loop → ReactiveState → outer loop → usuario

Una tarea background (inner loop) termina. Actualiza el `ReactiveState`.
El outer loop lo detecta via monitoreo constante e interrumpe lo que esté
haciendo para entregar el resultado al usuario.

Es un callback asíncrono. El ayudante vuelve del almacén con el libro.
El bibliotecario para lo que estaba diciendo y te lo entrega.

```
outer loop conversando normalmente...
         │
         │  inner loop completa tarea
         │  → actualiza ReactiveState
         │  → reactor detecta cambio
         │
         ▼
  session.interrupt()                    ← para al agente
  session.generate_reply(instructions=...) ← entrega resultado
         │
         ▼
  usuario recibe la info
  conversación continúa
```

**Cuándo ocurre:** en cualquier momento, independiente del turno del usuario.
El agente puede estar hablando, pensando o idle.

**Hook en LiveKit:** no es un hook — es código externo que llama directamente
a `session.interrupt()` + `session.generate_reply()`. Típicamente desde un
`asyncio.Task`, un callback de tarea, o una policy del reactor.

**Orden crítico:**
```
session.interrupt()                    ← para lo que haya en curso PRIMERO
session.generate_reply(instructions=...) ← lanza la nueva respuesta DESPUÉS

⚠ Aquí SÍ se usa interrupt() — estamos fuera del agent hook
```

**Aplicación en nuestro sistema:**
```python
# El inner loop completa y actualiza ReactiveState
async def background_search(ctx: SessionContext, query: str) -> None:
    result = await heavy_search(query)
    # Señalizar al outer loop via ReactiveState
    ctx.reactor.emit(Event(
        kind=EventKind.TASK_COMPLETED,
        payload={"message": result},
    ))

# El reactor (policy) detecta TASK_COMPLETED en ReactiveState
# y ejecuta la interrupción:
class TaskCompletedPolicy(Policy):
    def evaluate(self, state, event) -> list[Decision]:
        if event.kind == EventKind.TASK_COMPLETED:
            return [
                Decision(action=Action.INTERRUPT),
                Decision(action=Action.REPLY, payload={
                    "instructions": f"Comparte: {event.payload['message']}",
                }),
            ]
        return []
```

**Casos de uso:**

| Caso | Trigger | Qué hace |
|------|---------|----------|
| Búsqueda pesada | `TASK_COMPLETED` | Entrega resultados de web/DB |
| Cómputo largo | `TASK_COMPLETED` | Comparte resultado de análisis |
| Timeout inactividad | `TICK` + `user.silent_for > N` | Proactive nudge |
| Evento externo | `TASK_COMPLETED` via webhook | Notificación al usuario |
| Handoff de vuelta | `TASK_COMPLETED` de nodo especializado | Entrega resultado del nodo |
| Kill switch | `TICK` + condición crítica | Cierre forzado de sesión |

---

## Comparativa

```
                    GATE                        CALLBACK
                    ────                        ────────
Dirección           usuario → outer             inner → outer → usuario
Trigger             el usuario habla            evento asíncrono (task done)
Momento             boundary de turno           cualquier instante
El agente estaba    escuchando                  hablando / pensando / idle
Interrumpe agente   no (no estaba hablando)     sí (interrupt)
Hook LiveKit        on_user_turn_completed      código externo (reactor/task)
Mecanismo           generate_reply + Stop       interrupt + generate_reply
Qué evalúa          ReactiveState + mensaje     ReactiveState (sin mensaje)
Analogía            bibliotecario escucha y     ayudante vuelve del almacén,
                    decide antes de actuar      bibliotecario para y entrega
```

## Flujo completo — el bibliotecario

```
1. Usuario: "Busca info sobre Tatán Rufino en la web"

2. GATE: on_user_turn_completed evalúa
   → no hay condición especial en ReactiveState → pasa

3. LLM decide usar web_search → lanza tool call
   El tool crea un asyncio.Task (inner loop)
   El tool retorna "Buscando, resultados en unos segundos"
   El agente dice esto al usuario
   El agente sigue disponible — outer loop activo

4. Usuario: "Mientras, cuéntame qué tal el tiempo"
   GATE: evalúa → sin condición especial → pasa
   LLM responde sobre el tiempo normalmente
   (el inner loop sigue procesando en background)

5. Inner loop completa la búsqueda
   → actualiza ReactiveState (TASK_COMPLETED)
   → reactor detecta el cambio

6. CALLBACK: el reactor ejecuta
   → session.interrupt() (corta lo que sea que esté diciendo el agente)
   → session.generate_reply("Comparte estos resultados: ...")
   → el agente entrega los resultados de Tatán Rufino

7. Conversación continúa normalmente

A lo largo de todo el flujo, ReactiveSession estuvo alimentando
ReactiveState con cada cambio de modo, cada turno, cada tool call.
Ese estado es lo que permite tanto al gate como al callback tomar
decisiones informadas.
```

## Resumen

El double loop necesita ambos protocolos:

- **Gate** filtra lo que entra. Protege, redirige, intercepta el input del
  usuario antes de que llegue al LLM. Actúa en el momento del turno.

- **Callback** entrega lo que sale del inner loop. Interrumpe el flujo
  actual para surface resultados de tareas asíncronas. Actúa en cualquier
  momento.

Ambos leen del mismo `ReactiveState`. Ambos terminan en lo mismo: el agente
dice algo al usuario. La diferencia es cuándo, por qué, y desde dónde se
activan. Juntos hacen que la experiencia sea la de hablar con alguien que
siempre está atento y que trabaja en paralelo sin hacerte esperar.
