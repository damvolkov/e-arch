"""Agente buscador — un solo agente con herramientas web_search + web_fetch.

    make script search
"""

from __future__ import annotations

import logging
import sys

logging.getLogger("google_genai.models").setLevel(logging.WARNING)

from livekit import agents
from livekit.agents import Agent, AgentServer, AgentSession, cli
from livekit.plugins import google

from e_agents.rtc.models.state import SessionState
from e_agents.rtc.tools.fetch import web_fetch
from e_agents.rtc.tools.search import web_search
from e_agents.shared.adapters.searxng import SearXNGAdapter
from e_agents.shared.state import State

logger = logging.getLogger(__name__)


##### AGENT #####


class SearchAgent(Agent):

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "Eres un periodista e investigador experto. Hablas SIEMPRE en español.\n\n"
                "HERRAMIENTAS:\n"
                "- web_search: Busca en internet. Devuelve títulos, URLs y snippets.\n"
                "- web_fetch: Lee el contenido completo de una URL. Úsala para profundizar.\n\n"
                "FLUJO DE TRABAJO:\n"
                "1. SIEMPRE empieza con web_search para encontrar fuentes.\n"
                "2. Si el usuario pide profundidad o los snippets no bastan, usa web_fetch\n"
                "   en las 2-3 URLs más relevantes para obtener el artículo completo.\n"
                "3. SINTETIZA toda la información en una respuesta detallada y completa.\n\n"
                "REGLAS:\n"
                "1. NUNCA respondas sin buscar primero.\n"
                "2. Basa tus respuestas EXCLUSIVAMENTE en los resultados obtenidos.\n"
                "3. Incluye datos concretos: nombres, fechas, cifras, lugares, eventos.\n"
                "4. NUNCA digas 'revisa los enlaces' ni 'no puedo' — TÚ informas.\n"
                "5. Si los resultados son insuficientes, haz otra búsqueda con query diferente.\n"
                "6. Sé directo, informativo y exhaustivo. Eres un analista, no un intermediario.\n"
            ),
            tools=[web_search, web_fetch],
        )


##### SERVER #####

server = AgentServer()


@server.rtc_session(agent_name="searcher")
async def entrypoint(ctx: agents.JobContext) -> None:
    state = State()
    state.register_adapter(SearXNGAdapter())
    session_state = SessionState(shared=state)

    session = AgentSession[SessionState](
        userdata=session_state,
        llm=google.LLM(model="gemini-2.0-flash"),
        max_tool_steps=10,
        allow_interruptions=True,
        min_endpointing_delay=0.5,
        max_endpointing_delay=3.0,
    )
    await session.start(agent=SearchAgent(), room=ctx.room)


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "console", "--text"]
    cli.run_app(server)
