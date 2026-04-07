"""Double-loop PoC — Front + Journalist + Thinker brain nodes.

    make script dsolver

Architecture::

    ┌─────────────────────────────────────────────────────────────────┐
    │  AgentSession (userdata=ReactiveState)                          │
    │                                                                 │
    │  ┌──────────────┐  submit_task   ┌─────────────────────────┐   │
    │  │  FrontAgent   │──────────────▶│ JournalistNode          │   │
    │  │  (outer,      │               │ (search + fetch, async) │   │
    │  │   reactive,   │◁── push_event ├─────────────────────────┘   │
    │  │   user-facing)│               │                             │
    │  │               │  submit_task   ┌─────────────────────────┐   │
    │  │               │──────────────▶│ ThinkerNode             │   │
    │  │               │               │ (LLM reflect, async)    │   │
    │  │               │◁── push_event ├─────────────────────────┘   │
    │  └──────┬───────┘                                              │
    │         │  monitor_loop: wait_event → push_thread →            │
    │         │                generate_reply (LLM synthesis)        │
    │         ▼                                                       │
    │       [User]                                                    │
    └─────────────────────────────────────────────────────────────────┘

    Front is the ONLY LiveKit agent.
    Brain nodes are async tasks that update ReactiveState.
    Front reacts via monitor loop → enriches thread → LLM synthesizes.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

logging.getLogger("google_genai.models").setLevel(logging.WARNING)
logging.getLogger("trafilatura").setLevel(logging.WARNING)

import httpx
import trafilatura
from google import genai
from livekit import agents
from livekit.agents import AgentServer, AgentSession, RunContext, cli, function_tool
from livekit.plugins import google

from e_agents.arch.livekit import LiveKitReactiveAgent
from e_agents.arch.models import EventEffect, Priority, TaskConfig
from e_agents.arch.state import ReactiveState
from e_agents.shared.adapters.searxng import SearXNGAdapter
from e_agents.shared.models import SearchCategory

logger = logging.getLogger("dsolver")


##### BRAIN NODES #####


class JournalistNode:
    """Web researcher — searches SearXNG + fetches top articles. Not a LiveKit agent."""

    __slots__ = ()

    @staticmethod
    async def investigate(*, query: str, category: str = "news") -> dict[str, Any]:
        """Search + fetch top results, return compiled findings."""
        logger.info("📰 NODE_SEARCH query=%r category=%r", query, category, extra={"tags": "DLOOP"})

        try:
            cat = SearchCategory(category)
        except ValueError:
            cat = SearchCategory.NEWS

        results = await SearXNGAdapter.query(query, category=cat, max_results=5)
        logger.info("📰 NODE_SEARCH_OK results=%d", len(results), extra={"tags": "DLOOP"})

        articles: list[str] = []
        for r in results[:2]:
            logger.info("📰 NODE_FETCH url=%r", r.url[:80], extra={"tags": "DLOOP"})
            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(12.0),
                    follow_redirects=True,
                    headers={"User-Agent": "Mozilla/5.0 (compatible; e-agents/1.0)"},
                ) as client:
                    resp = await client.get(r.url)
                    resp.raise_for_status()
                text = trafilatura.extract(resp.text, favor_recall=True, deduplicate=True)
                if text:
                    articles.append(f"## {r.title}\nFuente: {r.url}\n{text[:2500]}")
            except Exception:
                logger.debug("📰 NODE_FETCH_FAIL url=%r", r.url[:80], extra={"tags": "DLOOP"})

        if articles:
            findings = "\n\n---\n\n".join(articles)
        else:
            findings = "\n".join(f"- {r.title}: {r.snippet}" for r in results)

        logger.info("📰 NODE_COMPLETE type=journalist chars=%d", len(findings), extra={"tags": "DLOOP"})
        return {"message": findings, "source": "journalist", "query": query}


class ThinkerNode:
    """Philosopher — analyzes with LLM. Not a LiveKit agent."""

    __slots__ = ("_client",)

    def __init__(self) -> None:
        self._client = genai.Client()

    async def reflect(self, *, question: str) -> dict[str, Any]:
        """Generate philosophical/analytical perspective via direct LLM call."""
        logger.info("🧠 NODE_LLM question=%r", question[:80], extra={"tags": "DLOOP"})

        response = await self._client.aio.models.generate_content(
            model="gemini-2.0-flash",
            contents=(
                "Eres un filósofo y pensador crítico. Analiza este tema desde múltiples "
                "perspectivas: ética, geopolítica, histórica y humanista. Sé profundo, "
                "provocador y matizado. Escribe en español. Máximo 800 palabras.\n\n"
                f"Tema: {question}"
            ),
        )
        analysis = response.text or "No se pudo generar análisis."

        logger.info("🧠 NODE_COMPLETE type=thinker chars=%d", len(analysis), extra={"tags": "DLOOP"})
        return {"message": analysis, "source": "thinker", "question": question}


##### FRONT AGENT #####


_JOURNALIST = JournalistNode()
_THINKER = ThinkerNode()


class FrontAgent(LiveKitReactiveAgent):
    """User-facing agent — dispatches to brain nodes, reacts to their findings."""

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "Eres Alda, una asistente de investigación inteligente. "
                "Hablas SIEMPRE en español.\n\n"
                "Tu personalidad:\n"
                "- Directa, informativa y analítica.\n"
                "- Fusionas datos concretos con perspectiva crítica.\n"
                "- Citas fuentes, nombres, fechas y cifras cuando están disponibles.\n"
                "- Nunca dices 'no puedo' — siempre actúas.\n\n"
                "Cuando el usuario pregunte algo que requiera investigación, "
                "usa las herramientas disponibles.\n"
                "Cuando recibas información nueva en tu contexto, "
                "sintetízala y compártela de forma completa y detallada."
            ),
        )

    # ── Tools ──────────────────────────────────────────────────────────

    @function_tool()
    async def investigate(
        self,
        context: RunContext[ReactiveState],
        query: str,
    ) -> str:
        """Dispatch journalist and thinker to investigate a topic in parallel. Use for any factual question, current events, or research request."""
        state = context.userdata

        logger.info("🚀 NODE_DISPATCH query=%r nodes=[journalist, thinker]", query, extra={"tags": "DLOOP"})

        await state.submit_task(
            TaskConfig(
                name="journalist",
                source="journalist",
                priority=Priority.HIGH,
                effect=EventEffect.INTERRUPT,
            ),
            _JOURNALIST.investigate,
            query=query,
            category="news",
        )

        await state.submit_task(
            TaskConfig(
                name="thinker",
                source="thinker",
                priority=Priority.NORMAL,
                effect=EventEffect.INTERRUPT,
            ),
            _THINKER.reflect,
            question=query,
        )

        logger.info(
            "🚀 NODE_DISPATCH_OK tasks=%d running=%d",
            2, state.running_count,
            extra={"tags": "DLOOP"},
        )
        return "Investigación en marcha. Te comparto los resultados en cuanto los tenga."

    @function_tool()
    async def ask_journalist(
        self,
        context: RunContext[ReactiveState],
        query: str,
        category: str = "news",
    ) -> str:
        """Dispatch only the journalist for web research. Use for quick factual lookups."""
        state = context.userdata

        logger.info("🚀 NODE_DISPATCH query=%r nodes=[journalist]", query, extra={"tags": "DLOOP"})

        await state.submit_task(
            TaskConfig(
                name="journalist",
                source="journalist",
                priority=Priority.HIGH,
                effect=EventEffect.INTERRUPT,
            ),
            _JOURNALIST.investigate,
            query=query,
            category=category,
        )
        return "Buscando información. Te comparto en cuanto la tenga."

    @function_tool()
    async def ask_thinker(
        self,
        context: RunContext[ReactiveState],
        question: str,
    ) -> str:
        """Dispatch only the thinker for philosophical analysis. Use when the user wants deep reflection."""
        state = context.userdata

        logger.info("🚀 NODE_DISPATCH question=%r nodes=[thinker]", question, extra={"tags": "DLOOP"})

        await state.submit_task(
            TaskConfig(
                name="thinker",
                source="thinker",
                priority=Priority.NORMAL,
                effect=EventEffect.INTERRUPT,
            ),
            _THINKER.reflect,
            question=question,
        )
        return "Analizando el tema. Te comparto mi perspectiva en breve."

    @function_tool()
    async def check_status(
        self,
        context: RunContext[ReactiveState],
    ) -> str:
        """Check status of running background tasks."""
        state = context.userdata
        tasks = state.running_tasks
        if not tasks:
            return "No hay tareas en ejecución."
        lines = [f"- {cfg.name} (prioridad: {cfg.priority.name})" for cfg in tasks.values()]
        return f"Tareas activas ({len(tasks)}):\n" + "\n".join(lines)


##### SERVER #####

server = AgentServer()


@server.rtc_session(agent_name="dsolver")
async def entrypoint(ctx: agents.JobContext) -> None:
    state = ReactiveState(max_concurrency=4)

    session = AgentSession[ReactiveState](
        userdata=state,
        llm=google.LLM(model="gemini-2.0-flash"),
        max_tool_steps=8,
        allow_interruptions=True,
        min_endpointing_delay=0.5,
        max_endpointing_delay=3.0,
    )
    await session.start(agent=FrontAgent(), room=ctx.room)


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "console", "--text"]
    cli.run_app(server)
