"""Handoff experiment — AssistantAgent delegates search to SearchAgent via native handoff.

State flows through RunContext.userdata, shared across agents.

    make script experiment
"""

from __future__ import annotations

import dataclasses
import logging
import sys
from enum import StrEnum, auto
from typing import Literal

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
from livekit.plugins import google

logger = logging.getLogger(__name__)

_SEARXNG_URL = "http://localhost:7700"
_MAX_RESULTS = 5


##### ENUMS #####


class SearchCategory(StrEnum):
    GENERAL = auto()
    NEWS = auto()
    IT = auto()
    SCIENCE = auto()


##### STATE #####


@dataclasses.dataclass
class HandoffResponse:
    status: Literal["pending", "finished"] = "pending"
    response: str = ""
    agent: str = ""


@dataclasses.dataclass
class State:
    handoff_search_agent: list[HandoffResponse] = dataclasses.field(default_factory=list)


##### SEARCH AGENT #####


class SearchAgent(Agent):

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "Eres un agente de búsqueda interno. "
                "SIEMPRE usa search_web con query y category. NUNCA respondas sin buscar."
            ),
        )

    async def on_enter(self) -> None:
        state: State = self.session.userdata
        state.handoff_search_agent.append(HandoffResponse(agent="search_agent"))
        logger.info("SEARCH_AGENT entered")

    @function_tool()
    async def search_web(
        self, context: RunContext[State], query: str, category: SearchCategory,
    ) -> str:
        """Busca información en internet. SIEMPRE proporciona query y category.

        Args:
            query: Consulta de búsqueda.
            category: Categoría de búsqueda.
        """
        state = context.userdata
        current = next(
            (h for h in reversed(state.handoff_search_agent) if h.status == "pending"), None,
        )

        try:
            async with httpx.AsyncClient(base_url=_SEARXNG_URL, timeout=10.0) as client:
                resp = await client.get(
                    "/search",
                    params={"q": query, "format": "json", "categories": category.value},
                    headers={"Accept": "application/json"},
                )
                resp.raise_for_status()

            raw = orjson.loads(resp.content).get("results", [])[:_MAX_RESULTS]
            formatted = "\n".join(
                f"- {r.get('title', '')}: {r.get('content', '')[:200]}" for r in raw
            ) or f"Sin resultados para '{query}'."

            logger.info("SEARCH_DONE query=%r cat=%s n=%d", query, category, len(raw))
        except Exception as exc:
            formatted = f"Error buscando '{query}': {exc}"
            logger.exception("SEARCH_FAIL query=%r", query)

        if current:
            current.response = formatted

        return formatted


##### ASSISTANT AGENT #####


class AssistantAgent(Agent):

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "Eres Maia. Hablas SIEMPRE en español. Directa, concisa, ejecutiva.\n\n"
                "REGLAS:\n"
                "- Si necesitas buscar, BUSCA. Si necesitas actuar, ACTÚA. No pidas permiso.\n"
                "- Siempre confirma brevemente lo que vas a hacer: 'Ok, busco eso.' y actúa.\n"
                "- Respuestas cortas, datos directos, sin adornos.\n"
                "- Si no sabes algo factual, busca. No especules."
            ),
        )

    async def on_enter(self) -> None:
        state: State = self.session.userdata
        pending = [h for h in state.handoff_search_agent if h.status == "pending"]

        if pending:
            for h in pending:
                h.status = "finished"
                logger.info("HANDOFF_FINISHED agent=%s", h.agent)

    @function_tool()
    async def transferir_a_buscador(self, context: RunContext[State], query: str) -> Agent:
        """Transfiere al agente buscador cuando el usuario necesita información de internet.

        Args:
            query: Lo que el usuario quiere buscar.
        """
        logger.info("HANDOFF_TO_SEARCH query=%r", query)
        return SearchAgent()


##### SERVER #####

server = AgentServer()


@server.rtc_session(agent_name="experiment")
async def entrypoint(ctx: agents.JobContext) -> None:
    session = AgentSession[State](
        userdata=State(),
        llm=google.LLM(model="gemini-2.0-flash"),
        max_tool_steps=5,
        allow_interruptions=True,
        min_endpointing_delay=0.5,
        max_endpointing_delay=3.0,
    )
    await session.start(agent=AssistantAgent(), room=ctx.room)


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "console", "--text"]
    cli.run_app(server)
