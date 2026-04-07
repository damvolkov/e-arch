"""Duo agent system: Attendant (outer) + Reasoner (inner).

Attendant: Always-on front agent. Handles conversation, delegates complex tasks.
Reasoner:  Deep-thinking agent. Two modes:
  - Background: Attendant submits a task, keeps talking, result arrives via state.
  - Handoff:    For multi-turn reasoning where the user interacts directly.

    make script duo
"""

from __future__ import annotations

import logging
import sys
from typing import Any

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

from e_agents.arch.livekit import LiveKitReactiveAgent
from e_agents.arch.models import (
    Event,
    EventEffect,
    EventPolicy,
    EventStrategy,
    Priority,
    TaskConfig,
    TaskStatus,
)
from e_agents.arch.state import ReactiveState

logger = logging.getLogger("rtc.agents.duo")


# ─── State ───────────────────────────────────────────────────────────────────

class DuoState(ReactiveState):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.user_name: str | None = None
        self.reasoning_depth: int = 0


# ─── Prompts ─────────────────────────────────────────────────────────────────

_ATTENDANT_PROMPT = (
    "Eres un asistente conversacional inteligente. Eres el punto de contacto principal.\n"
    "Tu trabajo:\n"
    "- Responder preguntas sencillas directamente.\n"
    "- Para análisis complejos, comparaciones, decisiones con múltiples factores,\n"
    "  problemas técnicos o preguntas que requieran razonamiento profundo:\n"
    "  usa la herramienta 'razonar_en_fondo' para lanzar el análisis en segundo plano\n"
    "  mientras sigues conversando con el usuario.\n"
    "- Si el usuario necesita una sesión interactiva de razonamiento paso a paso\n"
    "  (debugging, diseño de arquitectura, brainstorming), transfiérelo al Razonador.\n"
    "- Cuando recibas resultados de análisis en segundo plano, preséntalos\n"
    "  de forma clara y conversacional.\n"
    "Eres cálido, conciso y profesional. Hablas SIEMPRE en español."
)

_REASONER_PROMPT = (
    "Eres un agente de Razonamiento Profundo. Te especializas en:\n"
    "- Análisis paso a paso de problemas complejos\n"
    "- Evaluación de pros y contras con criterios explícitos\n"
    "- Debugging y resolución técnica interactiva\n"
    "- Diseño de soluciones con el usuario\n"
    "Cuando termines el razonamiento o el usuario esté satisfecho,\n"
    "transfiérelo de vuelta al asistente principal.\n"
    "Estructura tu pensamiento: enumera supuestos, evalúa alternativas,\n"
    "y presenta conclusiones claras.\n"
    "Hablas SIEMPRE en español."
)


# ─── Background Reasoning Handler ────────────────────────────────────────────

async def _reason_in_background(
    *,
    query: str,
    context_summary: str,
    llm_model: str = "gemini-2.0-flash",
) -> dict[str, Any]:
    """Background reasoning — runs outside the voice pipeline."""
    llm = google.LLM(model=llm_model)

    system = (
        "Eres un motor de razonamiento. Analiza la consulta en profundidad.\n"
        "Estructura tu respuesta:\n"
        "1. Reformulación del problema\n"
        "2. Factores clave\n"
        "3. Análisis\n"
        "4. Conclusión concisa (máx 2 frases, esto se leerá en voz alta)\n"
        "Responde SOLO en español."
    )

    chat_ctx = google.LLM.ChatContext()
    chat_ctx.add_message(role="system", content=system)
    if context_summary:
        chat_ctx.add_message(
            role="system",
            content=f"Contexto previo de la conversación:\n{context_summary}",
        )
    chat_ctx.add_message(role="user", content=query)

    response = await llm.chat(chat_ctx=chat_ctx)

    full_text = response.text
    conclusion = full_text.split("4.")[-1].strip() if "4." in full_text else full_text[:200]

    return {
        "message": f"He analizado tu consulta. {conclusion}",
        "full_analysis": full_text,
        "query": query,
    }


# ─── Attendant (outer — always facing the user) ─────────────────────────────

class AttendantAgent(LiveKitReactiveAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions=_ATTENDANT_PROMPT,
            monitor_interval=0.5,
        )

    def format_event(self, event: Event) -> str | None:
        match event.status:
            case TaskStatus.COMPLETED if event.source == "reasoner":
                return event.payload.get("message")
            case TaskStatus.COMPLETED:
                return event.payload.get("message")
            case TaskStatus.FAILED:
                return f"No pude completar el análisis: {event.payload.get('error', 'desconocido')}"
            case _:
                return None

    async def push_thread(self, payload: dict[str, Any]) -> None:
        """Override: push full analysis into thread, not just the message."""
        ctx = self.chat_ctx.copy()

        if full := payload.get("full_analysis"):
            ctx.add_message(
                role="system",
                content=(
                    f"[reasoning_result] Análisis completado para: {payload.get('query', '?')}\n"
                    f"{full}"
                ),
            )
        else:
            ctx.add_message(
                role="system",
                content=f"[background_result] {payload}",
            )

        await self.update_chat_ctx(ctx)

    # ── Tools ────────────────────────────────────────────────────────────

    @function_tool()
    async def razonar_en_fondo(
        self,
        context: RunContext[DuoState],
        consulta: str,
    ) -> None:
        """Lanza un análisis en segundo plano. No bloquea la conversación.

        Args:
            consulta: La pregunta o problema a analizar en profundidad.
        """
        recent = self.chat_ctx.items[-4:]
        summary = " | ".join(
            getattr(item, "text_content", "")
            for item in recent
            if hasattr(item, "text_content")
        )

        await self.submit_task(
            TaskConfig(
                name="background_reasoning",
                priority=Priority.NORMAL,
                effect=EventEffect.INTERRUPT,
                source="reasoner",
            ),
            handler=_reason_in_background,
            query=consulta,
            context_summary=summary,
        )

    @function_tool()
    async def transferir_a_razonador(
        self,
        context: RunContext[DuoState],
    ) -> Agent:
        """Transfiere al Razonador para sesión interactiva de análisis profundo."""
        self.session.say("Entendido, te conecto con el modo de razonamiento profundo.")
        return await self.transfer_to("reasoner")


# ─── Reasoner (inner — deep thinking, interactive) ───────────────────────────

class ReasonerAgent(LiveKitReactiveAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions=_REASONER_PROMPT,
            monitor_interval=1.0,
        )

    @function_tool()
    async def devolver_a_asistente(
        self,
        context: RunContext[DuoState],
    ) -> Agent:
        """Devuelve al usuario al asistente principal."""
        self.session.say("Perfecto, te devuelvo con el asistente principal.")
        return await self.transfer_to("attendant")

    @function_tool()
    async def analizar_en_fondo(
        self,
        context: RunContext[DuoState],
        consulta: str,
    ) -> None:
        """Lanza un sub-análisis en segundo plano mientras sigue la sesión.

        Args:
            consulta: Sub-problema a analizar en paralelo.
        """
        recent = self.chat_ctx.items[-6:]
        summary = " | ".join(
            getattr(item, "text_content", "")
            for item in recent
            if hasattr(item, "text_content")
        )

        await self.submit_task(
            TaskConfig(
                name="sub_analysis",
                priority=Priority.LOW,
                effect=EventEffect.ENRICH,
                source="reasoner_sub",
            ),
            handler=_reason_in_background,
            query=consulta,
            context_summary=summary,
        )


# ─── Server ─────────────────────────────────────────────────────────────────

server = AgentServer()



_DUO_POLICY = EventPolicy(
    rules={
        Priority.CRITICAL: (EventStrategy.IMMEDIATE, EventEffect.INTERRUPT),
        Priority.HIGH: (EventStrategy.IMMEDIATE, EventEffect.INTERRUPT),
        Priority.NORMAL: (EventStrategy.TURN_BOUNDARY, EventEffect.INTERRUPT),
        Priority.LOW: (EventStrategy.NATURAL_PAUSE, EventEffect.ENRICH),
        Priority.BACKGROUND: (EventStrategy.ENQUEUE, EventEffect.NOOP),
    },
    idle_timeout_seconds=2.5,
)


@server.rtc_session(agent_name="duo")
async def entrypoint(ctx: agents.JobContext) -> None:
    attendant = AttendantAgent()
    reasoner = ReasonerAgent()

    state = DuoState(policy=_DUO_POLICY, max_concurrency=3)
    state.register_agents({
        "attendant": attendant,
        "reasoner": reasoner,
    })

    session = AgentSession[DuoState](
        userdata=state,
        llm=google.LLM(model="gemini-2.0-flash"),
        max_tool_steps=5,
        allow_interruptions=True,
        min_endpointing_delay=0.5,
        max_endpointing_delay=3.0,
    )
    await session.start(agent=attendant, room=ctx.room)


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "console", "--text"]
    cli.run_app(server)
