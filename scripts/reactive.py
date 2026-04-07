"""ReactiveSession prototype — voice agent with policy-driven orchestration.

    make download          # descargar modelos (una vez)
    make script reactive   # voz con evoice STT/TTS + ReactiveSession
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent.parent
_ENV_PATH = _BASE_DIR / ".env"
if _ENV_PATH.exists():
    for line in _ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key and key not in os.environ:
            os.environ[key] = value

_TURN_CACHE = _BASE_DIR / "data" / "models" / "turn"
_VAD_MODEL = _BASE_DIR / "data" / "models" / "vad" / "silero_vad.onnx"
os.environ.setdefault("HF_HUB_CACHE", str(_TURN_CACHE))

logging.getLogger("google_genai.models").setLevel(logging.WARNING)

from livekit.agents import (
    Agent,
    AgentServer,
    JobContext,
    RunContext,
    TurnHandlingOptions,
    cli,
    function_tool,
)
from livekit.agents.voice import AgentSession
from livekit.plugins import google, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from e_agents.arch.models import Event, EventKind, ReactiveState
from e_agents.arch.policies import AwayPolicy, TaskCompletedPolicy
from e_agents.arch.protocols import Policy
from e_agents.arch.session import ReactiveSession
from e_agents.rtc.adapters.stt.evoice import EVoiceSTT
from e_agents.rtc.adapters.tts.evoice import EVoiceTTS

##### LOGGING #####


_CYAN = "\033[36m"
_YELLOW = "\033[33m"
_MAGENTA = "\033[35m"
_GREEN = "\033[32m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_RESET = "\033[0m"

_TAG_COLORS: dict[str, str] = {
    "react": _YELLOW,
    "policy": _MAGENTA,
    "action": f"{_BOLD}{_GREEN}",
    "state": _DIM,
    "agent": _CYAN,
    "system": _BOLD,
    "tick": f"{_DIM}{_YELLOW}",
}


def _log(tag: str, msg: str) -> None:
    color = _TAG_COLORS.get(tag, "")
    print(f"  {color}[{tag.upper():<8}]{_RESET} {msg}")


##### SESSION CONTEXT #####


@dataclass
class SessionContext:
    """Stored in session.userdata — bridges agent tools to reactive system."""

    state: ReactiveState
    reactor: ReactiveSession | None = None


##### LIVEKIT SESSION HANDLE #####


class LiveKitSession:
    """SessionHandle wrapping a real LiveKit AgentSession (voice mode)."""

    __slots__ = ("_session",)

    def __init__(self, session: AgentSession[SessionContext]) -> None:
        self._session = session

    async def interrupt(self) -> None:
        _log("action", "interrupt()")
        self._session.interrupt()

    async def say(self, *, text: str) -> None:
        _log("action", f"say({text!r})")
        self._session.say(text, add_to_chat_ctx=True)

    async def generate_reply(self, *, instructions: str) -> None:
        preview = instructions[:80] + ("..." if len(instructions) > 80 else "")
        _log("action", f"generate_reply({preview!r})")
        self._session.generate_reply(instructions=instructions)

    async def update_instructions(self, *, instructions: str) -> None:
        _log("action", f"update_instructions({instructions[:50]!r}) (stub)")

    async def swap_agent(self, *, agent_id: str) -> None:
        _log("action", f"swap_agent({agent_id!r}) (stub)")


##### BACKGROUND TASKS #####


async def _bg_weather(ctx: SessionContext, city: str) -> None:
    """Simulate weather API — emits TASK_COMPLETED after delay."""
    _log("agent", f"⏳ bg_weather({city!r}) started — 3.5s")
    await asyncio.sleep(3.5)
    result = f"Clima en {city}: Soleado, 22°C, humedad 45%, viento suave del norte"
    _log("agent", f"✅ bg_weather({city!r}) done → TASK_COMPLETED")
    ctx.reactor.emit(Event(
        kind=EventKind.TASK_COMPLETED,
        payload={"message": result},
    ))


async def _bg_news(ctx: SessionContext, topic: str) -> None:
    """Simulate news API — emits TASK_COMPLETED after delay."""
    _log("agent", f"⏳ bg_news({topic!r}) started — 8s")
    await asyncio.sleep(8.0)
    result = (
        f"Noticias sobre {topic}: "
        "(1) Gran avance en investigación con IA generativa. "
        "(2) Nuevo acuerdo comercial entre UE y Mercosur. "
        "(3) Descubrimiento de exoplaneta habitable a 40 años luz."
    )
    _log("agent", f"✅ bg_news({topic!r}) done → TASK_COMPLETED")
    ctx.reactor.emit(Event(
        kind=EventKind.TASK_COMPLETED,
        payload={"message": result},
    ))


##### AGENT #####


_ANA_INSTRUCTIONS = """\
Eres Ana, una asistente amable y directa. Hablas SIEMPRE en español.
Conversa con naturalidad. Respuestas cortas y claras.
No uses emojis, asteriscos ni markdown.

IMPORTANTE: Cuando el usuario pregunte por el clima de una ciudad, SIEMPRE usa \
la herramienta check_weather. Cuando pregunte por noticias, SIEMPRE usa lookup_news. \
No inventes datos — usa las herramientas.

Cuando lances una herramienta, dile al usuario que estás buscando y sigue conversando. \
Cuando recibas resultados (llegarán como instrucciones del sistema), compártelos \
de forma natural.\
"""


class Ana(Agent):
    """Agent with background tools — ReactiveSession handles task completion."""

    def __init__(self) -> None:
        super().__init__(instructions=_ANA_INSTRUCTIONS)

    async def on_enter(self) -> None:
        self.session.generate_reply(instructions="Saluda al usuario brevemente.")

    @function_tool()
    async def check_weather(
        self, context: RunContext[SessionContext], city: str,
    ) -> str:
        """Consulta el clima actual de una ciudad.

        Args:
            city: Ciudad para consultar el clima.
        """
        _log("agent", f"🔧 check_weather({city!r}) → bg task launched")
        asyncio.create_task(_bg_weather(context.userdata, city))
        return f"Buscando el clima en {city}. Resultados en unos segundos."

    @function_tool()
    async def lookup_news(
        self, context: RunContext[SessionContext], topic: str,
    ) -> str:
        """Busca noticias recientes sobre un tema.

        Args:
            topic: Tema a buscar.
        """
        _log("agent", f"🔧 lookup_news({topic!r}) → bg task launched")
        asyncio.create_task(_bg_news(context.userdata, topic))
        return f"Buscando noticias sobre {topic}. Resultados en unos segundos."


##### SERVER #####


server = AgentServer()


@server.rtc_session(agent_name="reactive")
async def entrypoint(ctx: JobContext) -> None:
    """Wire ReactiveSession + LiveKit voice agent together."""
    state = ReactiveState()
    policies: tuple[Policy, ...] = (
        TaskCompletedPolicy(),
        AwayPolicy(timeout=30.0),
    )

    session_ctx = SessionContext(state=state)
    vad_kwargs: dict = {}
    if _VAD_MODEL.exists():
        vad_kwargs["onnx_file_path"] = _VAD_MODEL

    session = AgentSession[SessionContext](
        userdata=session_ctx,
        llm=google.LLM(model="gemini-2.0-flash"),
        stt=EVoiceSTT(language="es"),
        tts=EVoiceTTS(voice="ef_dora"),
        vad=silero.VAD.load(**vad_kwargs),
        turn_handling=TurnHandlingOptions(
            turn_detection=MultilingualModel(),
            aec_warmup_duration=3.0,
        ),
        preemptive_generation=True,
        max_tool_steps=5,
    )

    lk_handle = LiveKitSession(session)
    reactor = ReactiveSession(lk_handle, state, policies, tick_interval=2.0, log=_log)
    session_ctx.reactor = reactor

    # Wire LiveKit session events → ReactiveSession events
    def _on_user_state(ev: object) -> None:
        new = getattr(ev, "new_state", None)
        match new:
            case "speaking":
                reactor.emit(Event(kind=EventKind.USER_SPEAKING))
            case "listening":
                reactor.emit(Event(kind=EventKind.USER_SILENT))

    def _on_agent_state(ev: object) -> None:
        match getattr(ev, "new_state", None):
            case "speaking":
                reactor.emit(Event(kind=EventKind.AGENT_SPEAKING))
            case "thinking":
                reactor.emit(Event(kind=EventKind.AGENT_THINKING))
            case "listening":
                reactor.emit(Event(kind=EventKind.AGENT_IDLE))

    session.on("user_state_changed", _on_user_state)
    session.on("agent_state_changed", _on_agent_state)

    asyncio.create_task(reactor.run())
    await session.start(agent=Ana(), room=ctx.room)


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "console"]
    cli.run_app(server)
