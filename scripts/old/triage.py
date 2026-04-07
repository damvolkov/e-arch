"""Medical Office Triage — double-loop multi-agent transfer.

Outer loop: triage routes patients to specialists based on need.
Inner loop: specialists handle domain queries and can route back.
LiveKit's agent handoff (function_tool returning Agent) carries context
and fires lifecycle hooks (on_enter/on_exit) automatically.

    make script triage
"""

from __future__ import annotations

import dataclasses
import logging
import sys
from typing import Any

from livekit import agents
from livekit.agents import (
    NOT_GIVEN,
    Agent,
    AgentServer,
    AgentSession,
    RunContext,
    cli,
    function_tool,
)
from livekit.plugins import google

logger = logging.getLogger(__name__)


##### STATE #####


@dataclasses.dataclass
class TriageState:
    """Shared session state — accessible via RunContext[TriageState].userdata."""

    transfer_count: int = 0


##### PROMPTS #####

_TRIAGE_PROMPT = (
    "Eres el agente de Triaje de una clínica médica. Eres el primer punto de contacto.\n"
    "Tu trabajo es entender la necesidad del paciente y derivarlo:\n"
    "- Servicios médicos, citas, síntomas → transferir a soporte\n"
    "- Seguros, facturación, pagos, reclamaciones → transferir a facturación\n"
    "Haz preguntas de clarificación si es necesario. Sé cálido, profesional y conciso.\n"
    "Hablas SIEMPRE en español."
)

_SUPPORT_PROMPT = (
    "Eres el agente de Soporte al Paciente. Te encargas de:\n"
    "- Agendamiento y reagendamiento de citas\n"
    "- Pre-evaluación de síntomas y derivación médica\n"
    "- Consultas generales sobre servicios médicos\n"
    "Si el paciente necesita ayuda con facturación, transfiere a facturación.\n"
    "Si la consulta está fuera de tu alcance, transfiere de vuelta a triaje.\n"
    "Hablas SIEMPRE en español."
)

_BILLING_PROMPT = (
    "Eres el agente de Facturación Médica. Te encargas de:\n"
    "- Verificación de seguros y estado de reclamaciones\n"
    "- Planes de pago y saldos pendientes\n"
    "- Disputas y ajustes de facturación\n"
    "Si el paciente necesita servicios médicos, transfiere a soporte.\n"
    "Si la consulta está fuera de tu alcance, transfiere de vuelta a triaje.\n"
    "Hablas SIEMPRE en español."
)

_TRIAGE_GREETING = (
    "Saluda al paciente cálidamente como agente de Triaje. "
    "Pregunta en qué puedes ayudarle. Sé breve y profesional."
)


##### AGENTS #####


class TriageAgent(Agent):
    """Outer loop — routes patients to the correct specialist."""

    def __init__(self, *, chat_ctx: Any = NOT_GIVEN) -> None:
        super().__init__(
            instructions=_TRIAGE_PROMPT,
            chat_ctx=chat_ctx,
        )

    @function_tool()
    async def transferir_a_soporte(self, context: RunContext[TriageState]) -> Agent:
        """Transfiere al paciente a Soporte para servicios médicos, citas y síntomas."""
        context.userdata.transfer_count += 1
        logger.info("HANDOFF triage→soporte (n=%d)", context.userdata.transfer_count)
        return SupportAgent(chat_ctx=context.session.current_agent.chat_ctx)

    @function_tool()
    async def transferir_a_facturacion(self, context: RunContext[TriageState]) -> Agent:
        """Transfiere al paciente a Facturación para seguros, pagos y reclamaciones."""
        context.userdata.transfer_count += 1
        logger.info("HANDOFF triage→facturación (n=%d)", context.userdata.transfer_count)
        return BillingAgent(chat_ctx=context.session.current_agent.chat_ctx)


class SupportAgent(Agent):
    """Inner loop — handles medical queries, appointments, symptoms."""

    def __init__(self, *, chat_ctx: Any = NOT_GIVEN) -> None:
        super().__init__(
            instructions=_SUPPORT_PROMPT,
            chat_ctx=chat_ctx,
        )

    @function_tool()
    async def transferir_a_triaje(self, context: RunContext[TriageState]) -> Agent:
        """Transfiere de vuelta a triaje para re-derivación."""
        context.userdata.transfer_count += 1
        logger.info("HANDOFF soporte→triaje (n=%d)", context.userdata.transfer_count)
        return TriageAgent(chat_ctx=context.session.current_agent.chat_ctx)

    @function_tool()
    async def transferir_a_facturacion(self, context: RunContext[TriageState]) -> Agent:
        """Transfiere a Facturación para seguros, pagos y reclamaciones."""
        context.userdata.transfer_count += 1
        logger.info("HANDOFF soporte→facturación (n=%d)", context.userdata.transfer_count)
        return BillingAgent(chat_ctx=context.session.current_agent.chat_ctx)


class BillingAgent(Agent):
    """Inner loop — handles insurance, billing, payment plans."""

    def __init__(self, *, chat_ctx: Any = NOT_GIVEN) -> None:
        super().__init__(
            instructions=_BILLING_PROMPT,
            chat_ctx=chat_ctx,
        )

    @function_tool()
    async def transferir_a_triaje(self, context: RunContext[TriageState]) -> Agent:
        """Transfiere de vuelta a triaje para re-derivación."""
        context.userdata.transfer_count += 1
        logger.info("HANDOFF facturación→triaje (n=%d)", context.userdata.transfer_count)
        return TriageAgent(chat_ctx=context.session.current_agent.chat_ctx)

    @function_tool()
    async def transferir_a_soporte(self, context: RunContext[TriageState]) -> Agent:
        """Transfiere a Soporte para servicios médicos, citas y síntomas."""
        context.userdata.transfer_count += 1
        logger.info("HANDOFF facturación→soporte (n=%d)", context.userdata.transfer_count)
        return SupportAgent(chat_ctx=context.session.current_agent.chat_ctx)


##### SERVER #####

server = AgentServer()



@server.rtc_session(agent_name="triage")
async def entrypoint(ctx: agents.JobContext) -> None:
    session = AgentSession[TriageState](
        userdata=TriageState(),
        llm=google.LLM(model="gemini-2.0-flash"),
        max_tool_steps=5,
        allow_interruptions=True,
        min_endpointing_delay=0.5,
        max_endpointing_delay=3.0,
    )
    await session.start(agent=TriageAgent(), room=ctx.room)


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "console", "--text"]
    cli.run_app(server)
