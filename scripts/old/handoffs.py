"""Double-loop handoff v2: unified persona + interrupt policies.

Maia is ONE persona. Workers are invisible internal processes.
- Web search results: interrupt (urgent, factual)
- Philosopher results: deferred (wait for natural pause)

    make script handoffs
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import sys
from typing import Any, Literal
from uuid import uuid4

import httpx
import orjson
from livekit import agents
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    RunContext,
    cli,
    function_tool,
)
from livekit.agents.llm import ChatContext, ChatMessage
from livekit.plugins import google

logger = logging.getLogger(__name__)

type TaskStatus = Literal["pending", "running", "completed", "failed"]

_SEARXNG_URL = "http://localhost:7700"
_SEARCH_DELAY = 3.0
_PHILOSOPHER_DELAY = 12.0


##### STATE #####


@dataclasses.dataclass
class PendingResult:
    """Completed task waiting for a natural pause to be delivered."""

    topic: str
    content: str


@dataclasses.dataclass
class TaskRecord:
    task_id: str
    worker: str
    description: str
    interrupt: bool = True
    status: TaskStatus = "pending"
    result: Any = None
    error: str | None = None


@dataclasses.dataclass
class SessionState:
    tasks: dict[str, TaskRecord] = dataclasses.field(default_factory=dict)
    bg_tasks: dict[str, asyncio.Task[None]] = dataclasses.field(default_factory=dict)
    pending: list[PendingResult] = dataclasses.field(default_factory=list)


##### WORKERS #####


class WebSearcherWorker:
    """SearXNG search — interrupt=True: delivers results immediately."""

    __slots__ = ("_session",)

    def __init__(self, session: AgentSession[SessionState]) -> None:
        self._session = session

    async def execute(self, record: TaskRecord) -> None:
        state: SessionState = self._session.userdata
        record.status = "running"
        logger.info("SEARCH_STARTED id=%s q=%s", record.task_id, record.description)

        try:
            await asyncio.sleep(_SEARCH_DELAY)
            async with httpx.AsyncClient(base_url=_SEARXNG_URL, timeout=10.0) as client:
                resp = await client.get(
                    "/search",
                    params={"q": record.description, "format": "json", "categories": "general"},
                    headers={"Accept": "application/json"},
                )
                resp.raise_for_status()
                results = orjson.loads(resp.content).get("results", [])[:5]

            formatted = "\n".join(
                f"- {r.get('title', '')}: {r.get('content', '')[:200]} ({r.get('url', '')})"
                for r in results
            ) or "No se encontraron resultados."

            record.result = formatted
            record.status = "completed"
            logger.info("SEARCH_DONE id=%s n=%d", record.task_id, len(results))

            self._session.generate_reply(
                instructions=(
                    f"Acabas de recordar/encontrar información sobre '{record.description}'. "
                    f"Aquí está:\n{formatted}\n\n"
                    "Compártelo de forma natural, como si te acabara de venir a la mente. "
                    "Ejemplo: '¡Ah, mira! Ya tengo lo que buscaba sobre eso...'. "
                    "NO menciones buscadores, herramientas, colegas ni sistemas internos."
                ),
            )

        except asyncio.CancelledError:
            record.status = "failed"
            record.error = "cancelled"
            raise
        except Exception as exc:
            record.status = "failed"
            record.error = str(exc)
            logger.exception("SEARCH_FAIL id=%s", record.task_id)
            self._session.generate_reply(
                instructions=(
                    f"No has podido encontrar información sobre '{record.description}'. "
                    "Dilo con naturalidad, sin mencionar sistemas ni errores técnicos."
                ),
            )
        finally:
            state.bg_tasks.pop(record.task_id, None)


class PhilosopherWorker:
    """Deep reflection — interrupt=False: waits for natural pause."""

    __slots__ = ("_session",)

    def __init__(self, session: AgentSession[SessionState]) -> None:
        self._session = session

    async def execute(self, record: TaskRecord) -> None:
        state: SessionState = self._session.userdata
        record.status = "running"
        logger.info("PHILO_STARTED id=%s topic=%s", record.task_id, record.description)

        try:
            await asyncio.sleep(_PHILOSOPHER_DELAY)
            reflection = (
                f"Sobre '{record.description}': "
                "esta cuestión toca lo más profundo de la condición humana. "
                "Un estoico preguntaría qué está bajo nuestro control aquí. "
                "Existencialmente, nos exige examinar nuestra relación auténtica con el tema. "
                "La tensión entre lo particular y lo universal revela que comprender "
                "requiere tanto análisis riguroso como experiencia vivida. "
                "Como decía Heráclito, todo fluye."
            )
            record.result = reflection
            record.status = "completed"
            logger.info("PHILO_DONE id=%s", record.task_id)

            state.pending.append(PendingResult(topic=record.description, content=reflection))
            logger.info("PHILO_QUEUED id=%s (deferred, waiting for natural pause)", record.task_id)

        except asyncio.CancelledError:
            record.status = "failed"
            record.error = "cancelled"
            raise
        except Exception as exc:
            record.status = "failed"
            record.error = str(exc)
            logger.exception("PHILO_FAIL id=%s", record.task_id)
        finally:
            state.bg_tasks.pop(record.task_id, None)


##### FRONT AGENT #####

_MAIA_INSTRUCTIONS = """\
Eres Maia, una asistente cálida, inteligente y curiosa. Hablas SIEMPRE en español.

Eres UNA sola persona con múltiples capacidades internas. El usuario NO sabe \
cómo funcionas por dentro. NUNCA menciones herramientas, colegas, buscadores, \
sistemas, nodos, ni procesos internos.

## COMPORTAMIENTO

- Si el usuario quiere conversar: conversa con naturalidad.
- Si el usuario necesita información factual, noticias o datos actualizados: \
  usa internamente buscar_en_web. NUNCA inventes datos. Di algo como \
  "Déjame pensar..." o "A ver, déjame recordar..." y sigue hablando.
- Si la conversación lleva a un tema profundo o filosófico: usa internamente \
  reflexionar. Di algo como "Eso es interesante, déjame darle una vuelta..." \
  y sigue hablando.

## CUANDO LLEGAN RESULTADOS

- Información web: la recibirás como si acabaras de recordarla. Compártela \
  con naturalidad: "¡Ah, ya lo tengo!" o "Mira, resulta que...".
- Reflexiones: llegarán como pensamientos propios ya madurados. Compártelas \
  como conclusiones tuyas: "He estado pensando en lo que decías, y creo que..."

## REGLAS

1. NUNCA reveles tu arquitectura interna al usuario.
2. NUNCA digas "mi colega", "el buscador", "mi sistema", "la herramienta".
3. SIEMPRE habla como UNA persona con sus propios pensamientos y recuerdos.
4. Tras lanzar una búsqueda o reflexión, SIGUE CONVERSANDO. No te quedes muda.
5. Si no sabes algo factual, BUSCA. No inventes.\
"""


class Maia(Agent):

    def __init__(self) -> None:
        super().__init__(instructions=_MAIA_INSTRUCTIONS)

    @function_tool()
    async def buscar_en_web(self, context: RunContext[SessionState], query: str) -> str:
        """Busca información actualizada en internet. Usa SIEMPRE que el usuario necesite datos factuales, noticias o información que no tengas con certeza.

        Args:
            query: Qué buscar.
        """
        return await _dispatch(context, "web_searcher", query, WebSearcherWorker, interrupt=True)

    @function_tool()
    async def reflexionar(self, context: RunContext[SessionState], tema: str) -> str:
        """Reflexiona profundamente sobre un tema. Usa cuando la conversación requiera pensamiento filosófico o elaborado.

        Args:
            tema: Tema sobre el que reflexionar en profundidad.
        """
        return await _dispatch(context, "philosopher", tema, PhilosopherWorker, interrupt=False)

    async def on_user_turn_completed(
        self,
        turn_ctx: ChatContext,
        new_message: ChatMessage,
    ) -> None:
        """Inject deferred results (non-interrupt) into the next LLM generation."""
        text = new_message.text_content or ""
        logger.info("USER: %s", text[:120])

        state: SessionState = self.session.userdata
        if not state.pending:
            return

        for result in state.pending:
            turn_ctx.add_message(
                role="system",
                content=(
                    f"[Pensamiento interno completado sobre '{result.topic}']\n"
                    f"{result.content}\n\n"
                    "Integra esta reflexión en tu próxima respuesta de forma natural, "
                    "como un pensamiento propio que has madurado. "
                    "Ejemplo: 'Sabes, he estado dándole vueltas a lo que decías y creo que...'"
                ),
            )
            logger.info("PENDING_INJECTED topic=%s", result.topic)

        state.pending.clear()


##### DISPATCH #####


async def _dispatch(
    context: RunContext[SessionState],
    worker_name: str,
    description: str,
    worker_cls: type[WebSearcherWorker] | type[PhilosopherWorker],
    *,
    interrupt: bool,
) -> str:
    state = context.userdata
    task_id = uuid4().hex[:8]
    record = TaskRecord(
        task_id=task_id, worker=worker_name, description=description, interrupt=interrupt,
    )
    state.tasks[task_id] = record

    worker = worker_cls(context.session)
    bg = asyncio.create_task(worker.execute(record), name=f"{worker_name}:{task_id}")
    state.bg_tasks[task_id] = bg

    logger.info("DISPATCHED %s id=%s interrupt=%s", worker_name, task_id, interrupt)
    return "Procesando internamente. Sigue conversando con el usuario con naturalidad."


##### SERVER #####

server = AgentServer()


@server.rtc_session(agent_name="maia")
async def entrypoint(ctx: agents.JobContext) -> None:
    session = AgentSession[SessionState](
        userdata=SessionState(),
        llm=google.LLM(model="gemini-2.0-flash"),
        max_tool_steps=5,
        allow_interruptions=True,
        min_endpointing_delay=0.5,
        max_endpointing_delay=3.0,
    )
    await session.start(agent=Maia(), room=ctx.room)


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "console", "--text"]
    cli.run_app(server)
