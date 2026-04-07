# Agent EDA Architecture

## Event-Driven Agent Loop with Hook Lifecycle, Singleton State & Concurrency Safety

Arquitectura para agentes autónomos inspirada en los internals de Claude Code (TypeScript/Bun), portada a Python asyncio. Diseñada para ser **framework-agnostic**: no depende de LangChain, CrewAI, ni AutoGen.

---

## 1. Visión General

```
User Input
    │
    ▼
┌──────────────────────────────────────────────────────┐
│                    AGENT LOOP (Reactor)               │
│                                                       │
│   ┌─────────┐    ┌─────────┐    ┌────────────────┐   │
│   │ HookReg │◄──►│ Global  │◄──►│ Tool Executor  │   │
│   │  istry  │    │ State   │    │ (concurrency)  │   │
│   └────┬────┘    └─────────┘    └───────┬────────┘   │
│        │                                │             │
│        ▼                                ▼             │
│   ┌─────────┐                    ┌────────────┐      │
│   │ 27 Hook │                    │ Tool       │      │
│   │ Events  │                    │ Registry   │      │
│   └─────────┘                    └────────────┘      │
│                                                       │
│   Turn N: LLM Call → Tool Selection → Execution →    │
│           Result Injection → Turn N+1                 │
│                                                       │
│   Exit: final_answer | ask_user | budget_exhausted    │
└──────────────────────────────────────────────────────┘
    │
    ▼
Response
```

El loop NO es request-response. Es un **reactor multi-turn** donde cada turno el LLM decide: ejecutar tools, responder, o pedir clarificación. El loop continúa hasta que el agente termina o se agota el budget.

---

## 2. Capas de la Arquitectura

| Capa | Responsabilidad | Patrón |
|------|----------------|--------|
| **HookRegistry** | Interceptar y controlar flujo en 27 puntos del lifecycle | Observer + Chain of Responsibility |
| **AgentState** | Estado mutable compartido por todos los módulos | Singleton (global mutable) |
| **ToolRegistry** | Catálogo de tools con schema uniforme | Registry + Strategy |
| **ToolExecutor** | Ejecución con concurrency safety y retries | Semaphore mutex + gather |
| **AgentLoop** | Orquestación multi-turn: LLM → tools → inject → repeat | Reactor |

---

## 3. Hook Taxonomy

27 eventos agrupados en 4 dominios de ejecución. Cada hook point corresponde a un momento preciso del agent loop.

### 3.1 Tool Lifecycle

Controla toda la vida de una ejecución de tool, desde el registro hasta el retry.

```
ON_TOOL_REGISTER ──► ON_TOOL_PRE_VALIDATE ──► ON_TOOL_AUTHORIZED ──► ON_TOOL_PRE_EXECUTE
                            │                        │                       │
                            ▼                        ▼                       ▼
                     ON_TOOL_DENIED            ON_TOOL_DENIED          ON_TOOL_POST_EXECUTE
                     (abort)                   (abort)                       │
                                                                            ▼
                                                                     ON_TOOL_ERROR
                                                                            │
                                                                            ▼
                                                                     ON_TOOL_RETRY
```

| Evento | Cuándo dispara | Qué puede hacer un hook |
|--------|---------------|------------------------|
| `ON_TOOL_REGISTER` | Tool añadida al registry | Validar schema, logging |
| `ON_TOOL_PRE_VALIDATE` | Antes del permission check | **Bloquear** commands peligrosos |
| `ON_TOOL_AUTHORIZED` | Permiso concedido | Audit logging, override permisos |
| `ON_TOOL_DENIED` | Permiso denegado | Notificar usuario, logging |
| `ON_TOOL_PRE_EXECUTE` | Justo antes de ejecutar | **Mutar** args, inject context |
| `ON_TOOL_POST_EXECUTE` | Después de ejecutar con resultado | Cache result, transform output |
| `ON_TOOL_ERROR` | Ejecución falló | Error tracking, alerting |
| `ON_TOOL_RETRY` | Reintentando tras fallo | Backoff strategy, circuit breaker |

### 3.2 LLM Lifecycle

Controla las llamadas al modelo. Permite mutación del prompt y transformación de respuestas.

```
ON_LLM_PRE_REQUEST ──► ON_LLM_STREAM_CHUNK (×N) ──► ON_LLM_POST_RESPONSE
       │                                                     │
       ▼                                                     ▼
ON_LLM_ERROR                                        ON_LLM_TOKEN_LIMIT
                                                             │
                                                             ▼
                                                    ON_LLM_COMPACTION
```

| Evento | Cuándo dispara | Qué puede hacer un hook |
|--------|---------------|------------------------|
| `ON_LLM_PRE_REQUEST` | Antes de la API call | **Mutar** prompt, inject system context, **bloquear** |
| `ON_LLM_POST_RESPONSE` | Después de la respuesta | Transform output, token tracking |
| `ON_LLM_STREAM_CHUNK` | Cada chunk en streaming | Progress UI, live logging |
| `ON_LLM_TOKEN_LIMIT` | Context window cerca del límite | Trigger compaction |
| `ON_LLM_ERROR` | API call falló | Retry logic, fallback model |
| `ON_LLM_COMPACTION` | Contexto compactado (resumido) | Logging, notify user |

### 3.3 Agent Loop Lifecycle

Eventos a nivel de turno y planificación.

| Evento | Cuándo dispara | Qué puede hacer un hook |
|--------|---------------|------------------------|
| `ON_TURN_START` | Nuevo turno de razonamiento | Logging, cost tracking |
| `ON_TURN_END` | Turno completado | Metrics, checkpoint state |
| `ON_PLAN_CREATED` | Agente creó plan de ejecución | Validar plan, notify user |
| `ON_PLAN_STEP_COMPLETE` | Un paso del plan terminó | Progress tracking |
| `ON_BUDGET_WARNING` | Budget al 90% | Notify, reduce scope |
| `ON_BUDGET_EXHAUSTED` | Budget agotado → force stop | Graceful shutdown |
| `ON_AGENT_IDLE` | Sin tareas pendientes | Prompt for input, auto-shutdown |

### 3.4 Session Lifecycle

Eventos de bootstrap, memoria, y subagentes.

| Evento | Cuándo dispara | Qué puede hacer un hook |
|--------|---------------|------------------------|
| `ON_SESSION_START` | Bootstrap completo | Init resources, load config |
| `ON_SESSION_RESUME` | Resumed desde checkpoint | Restore state |
| `ON_SESSION_END` | Shutdown graceful | Cleanup, persist state |
| `ON_CONTEXT_INJECTED` | Contexto externo cargado (CLAUDE.md equiv) | Validate, filter |
| `ON_MEMORY_LOADED` | Memoria long-term recuperada | Merge with session |
| `ON_SUBAGENT_SPAWNED` | Agente hijo creado | Inherit permissions, limit tools |
| `ON_SUBAGENT_RETURNED` | Agente hijo terminó | Merge results, cleanup |

---

## 4. Hook System — Flujo de Control

Cada handler retorna un `HookResult` con una acción que controla el flujo:

```
Hook dispatched
    │
    ├── CONTINUE ──► Ejecuta siguiente hook en la cadena
    │
    ├── MUTATE ────► Actualiza el contexto con payload, sigue cadena
    │
    ├── BLOCK ─────► ABORTA operación. No ejecuta más hooks.
    │
    └── SKIP ──────► Salta hooks restantes, procede con contexto actual
```

Los hooks se ejecutan en orden de **prioridad** (menor = primero). Un hook de seguridad en `priority=10` bloquea antes de que el logger en `priority=999` lo vea.

```python
hooks.on(HookEvent.ON_TOOL_PRE_VALIDATE, security_hook, priority=10)   # Evalúa primero
hooks.on(HookEvent.ON_TOOL_PRE_VALIDATE, audit_hook, priority=50)      # Segundo
hooks.on(HookEvent.ON_TOOL_PRE_VALIDATE, log_hook, priority=999)       # Último
```

---

## 5. Global State Singleton

Estado mutable compartido por todos los componentes. Sin DI container — acceso directo por import.

```python
from agent_arch import get_state

state = get_state()          # Siempre la misma instancia
state.turn_count += 1        # Cualquier módulo puede mutar
state.total_tokens_used      # Cualquier módulo puede leer
```

### Campos del AgentState

| Campo | Tipo | Propósito |
|-------|------|-----------|
| `session_id` | `str` | Identificador único de sesión |
| `turn_count` | `int` | Contador de turnos del agent loop |
| `total_tokens_used` | `int` | Tokens consumidos (acumulativo) |
| `token_budget` | `int` | Límite hard de tokens |
| `conversation` | `list[dict]` | Historial completo (user + assistant + tool_result) |
| `tool_results` | `dict[str, Any]` | Cache de últimos resultados por tool |
| `pending_tasks` | `dict[str, Task]` | Tasks asyncio en vuelo |
| `memory` | `dict[str, Any]` | Estado persistente entre sesiones |

### Trade-offs

| Ventaja | Desventaja |
|---------|-----------|
| Zero overhead de bootstrap | Acoplamiento implícito entre módulos |
| Acceso O(1) desde cualquier punto | Race conditions si no se protege |
| Fácil de debuggear (un solo objeto) | Difícil de testear sin `reset_state()` |
| No necesita framework de DI | No escala a multi-process sin Redis |

---

## 6. Tool System

### 6.1 Schema Uniforme

Toda tool expone la misma interfaz. El agent loop no sabe qué hace cada tool — solo conoce su schema.

```python
ToolSchema(
    name="grep",
    description="Search code with ripgrep",
    concurrency_safe=True,       # ← Puede correr en paralelo
    permission=PermissionLevel.READ_ONLY,
    max_retries=2,
    timeout_seconds=30.0,
)
```

### 6.2 Permission Levels

7 niveles de permiso, del más al menos permisivo:

```
UNRESTRICTED ──► AUTO_APPROVE ──► CLASSIFIER ──► WRITE_AUTO ──► INTERACTIVE ──► READ_ONLY ──► DELEGATED
 (testing)       (logged)       (LLM decide)    (edits ok)     (user prompt)   (no writes)   (subagent)
```

### 6.3 Concurrency Safety

Cada tool declara si es safe para ejecución paralela. El executor usa esta declaración para decidir la estrategia:

```
┌─────────────────────────────────────────────────┐
│              execute_batch([...])                │
│                                                  │
│   ┌──────────────┐     ┌──────────────────────┐ │
│   │ Safe tools   │     │ Unsafe tools         │ │
│   │              │     │                      │ │
│   │ grep    ─┐   │     │ file_write ──► sem   │ │
│   │          ├──►│     │                 │    │ │
│   │ search  ─┘   │     │ bash ────────► sem   │ │
│   │              │     │            (serial)  │ │
│   │ (parallel)   │     │                      │ │
│   │ gather()     │     │ await one by one     │ │
│   └──────────────┘     └──────────────────────┘ │
└─────────────────────────────────────────────────┘
```

| Tool | `concurrency_safe` | Estrategia | Por qué |
|------|-------------------|-----------|---------|
| `grep` | `True` | `asyncio.gather()` | Read-only, sin side effects |
| `web_search` | `True` | `asyncio.gather()` | Read-only, API externa |
| `file_write` | `False` | `Semaphore(1)` | Mutación de filesystem |
| `bash` | `False` | `Semaphore(1)` | Ejecución arbitraria, side effects |

---

## 7. Agent Loop — Flujo de un Turno

```
                          ┌─────────────────────┐
                          │   ON_TURN_START      │
                          └──────────┬──────────┘
                                     │
                                     ▼
                          ┌─────────────────────┐
                          │ ON_LLM_PRE_REQUEST   │──── BLOCK? ──► EXIT
                          └──────────┬──────────┘
                                     │
                                     ▼
                          ┌─────────────────────┐
                          │   LLM API Call       │
                          └──────────┬──────────┘
                                     │
                                     ▼
                          ┌─────────────────────┐
                          │ ON_LLM_POST_RESPONSE │
                          └──────────┬──────────┘
                                     │
                          ┌──────────┴──────────┐
                          │  match action:       │
                          └──────────┬──────────┘
                    ┌────────────────┼────────────────┐
                    ▼                ▼                 ▼
             ┌────────────┐  ┌────────────┐   ┌────────────┐
             │ tool_calls │  │final_answer│   │  ask_user  │
             └─────┬──────┘  └─────┬──────┘   └─────┬──────┘
                   │               │                 │
                   ▼               ▼                 ▼
          ┌──────────────┐    STOP LOOP         STOP LOOP
          │ Tool Executor│    return text        return question
          │  (batch)     │
          └──────┬───────┘
                 │
                 ▼
          ┌──────────────┐
          │ Inject results│
          │ into convo    │
          └──────┬───────┘
                 │
                 ▼
          ┌──────────────┐
          │ ON_TURN_END  │──────► NEXT TURN
          └──────────────┘
```

### Condiciones de salida

| Condición | Evento disparado | Resultado |
|-----------|-----------------|-----------|
| LLM dice `final_answer` | `ON_TURN_END` → `ON_SESSION_END` | Respuesta final al usuario |
| LLM dice `ask_user` | `ON_TURN_END` → `ON_AGENT_IDLE` | Pregunta de clarificación |
| Budget ≥ 100% | `ON_BUDGET_EXHAUSTED` → `ON_SESSION_END` | Force stop |
| Hook bloquea LLM | `ON_LLM_PRE_REQUEST` → BLOCK | Abort inmediato |

---

## 8. Ejemplo de Ejecución

Con 4 tools registradas y un LLM simulado que toma 3 turnos:

```
══════════════════════════════════════════════════
  🚀 SESSION a1b2c3d4e5f6 — budget: 50,000 tokens
══════════════════════════════════════════════════

  📡 ON_SESSION_START: session_id=a1b2c3d4e5f6

  🔄 ON_TURN_START: turn=1
  🧠 ON_LLM_PRE_REQUEST
  🧠 ON_LLM_POST_RESPONSE: tokens=1500
  🔧 ON_TOOL_PRE_VALIDATE: tool_name=grep           ← parallel group
  🔧 ON_TOOL_PRE_VALIDATE: tool_name=web_search     ← parallel group
  🔧 ON_TOOL_PRE_EXECUTE: tool_name=grep
  🔧 ON_TOOL_PRE_EXECUTE: tool_name=web_search
  🔧 ON_TOOL_POST_EXECUTE: tool_name=grep            ← both finish ~0.1s
  🔧 ON_TOOL_POST_EXECUTE: tool_name=web_search      ← both finish ~0.8s
  🔄 ON_TURN_END: turn=1, action=tool_calls

  🔄 ON_TURN_START: turn=2
  🧠 ON_LLM_PRE_REQUEST
  🧠 ON_LLM_POST_RESPONSE: tokens=2300
  🔧 ON_TOOL_PRE_VALIDATE: tool_name=file_write     ← serialized (semaphore)
  🔧 ON_TOOL_PRE_EXECUTE: tool_name=file_write
  🔧 ON_TOOL_POST_EXECUTE: tool_name=file_write
  🔄 ON_TURN_END: turn=2, action=tool_calls

  🔄 ON_TURN_START: turn=3
  🧠 ON_LLM_PRE_REQUEST
  🧠 ON_LLM_POST_RESPONSE: tokens=2800
  🔄 ON_TURN_END: turn=3, action=final_answer

  📡 ON_SESSION_END: turns=3

══════════════════════════════════════════════════
  ✅ FINAL: Done. Found 2 async functions, applied fix.
  📊 Turns: 3 | Tokens: 2,800/50,000
══════════════════════════════════════════════════
```

---

## 9. Extensibilidad

### Añadir una tool nueva

Registrar con schema. El agent loop no cambia.

```python
await registry.register(Tool(
    schema=ToolSchema("docker_exec", "Run in container", concurrency_safe=False, permission=PermissionLevel.INTERACTIVE),
    execute=docker_run_fn,
))
```

### Añadir un hook nuevo

Registrar con evento y prioridad. Nada más cambia.

```python
async def rate_limit_hook(event: HookEvent, ctx: dict[str, Any]) -> HookResult:
    if too_many_requests():
        return HookResult(HookAction.BLOCK, reason="Rate limited")
    return HookResult(HookAction.CONTINUE)

hooks.on(HookEvent.ON_LLM_PRE_REQUEST, rate_limit_hook, priority=5)
```

### Escalar a distribuido

Reemplazar `EventBus` in-process por Redis Streams o NATS. El `HookRegistry` pasa a leer de un topic. Las tools publican resultados en el bus. El `AgentState` migra a un hash en Redis con TTL. Cero cambios en el `AgentLoop`.

---

## 10. Comparativa con Frameworks Existentes

| Aspecto | Esta arquitectura | LangGraph | CrewAI | AutoGen |
|---------|------------------|-----------|--------|---------|
| Execution model | Reactor multi-turn | DAG secuencial | Sequential/hierarchical | Round-robin chat |
| Hook system | 27 eventos, 4 dominios | Callbacks limitados | No | No |
| Concurrency | Opt-in per tool | No nativo | No | No |
| State management | Global singleton | Checkpointer | Shared memory | Chat history |
| Tool permissions | 7 niveles granulares | No | No | No |
| Framework lock-in | Ninguno | LangChain | CrewAI | AutoGen |