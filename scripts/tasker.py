"""Minimal LiveKit AgentTask demo.

Shows: AgentTask lifecycle, typed result, tool-driven completion,
       task → agent handoff.

    make script task_demo
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
if _ENV_PATH.exists():
    for line in _ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key and key not in os.environ:
            os.environ[key] = value

_BASE_DIR = Path(__file__).resolve().parent.parent
_TURN_CACHE = _BASE_DIR / "data" / "models" / "turn"
_VAD_MODEL = _BASE_DIR / "data" / "models" / "vad" / "silero_vad.onnx"
os.environ.setdefault("HF_HUB_CACHE", str(_TURN_CACHE))

logging.getLogger("google_genai.models").setLevel(logging.WARNING)

from livekit.agents import (
    Agent,
    AgentServer,
    AgentTask,
    JobContext,
    RunContext,
    TurnHandlingOptions,
    cli,
    function_tool,
)
from livekit.agents.voice import AgentSession
from livekit.plugins import google, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from e_agents.rtc.adapters.stt.evoice import EVoiceSTT
from e_agents.rtc.adapters.tts.evoice import EVoiceTTS


##### TASK — COLLECT NAME #####


class CollectName(AgentTask[str]):
    """Task that asks the user's name and completes with it."""

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "Tu ÚNICO objetivo es averiguar el nombre del usuario. "
                "Pregúntale cómo se llama. Cuando te diga su nombre, "
                "usa la herramienta 'record_name' para registrarlo. "
                "No hables de nada más hasta tener el nombre."
            ),
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions="Saluda brevemente y pregunta al usuario cómo se llama.",
        )

    @function_tool
    async def record_name(self, context: RunContext, name: str) -> None:
        """Registra el nombre del usuario cuando lo haya dicho.

        Args:
            name: El nombre que el usuario ha proporcionado.
        """
        print(f"\n✓ TASK COMPLETE — name captured: {name!r}\n")
        self.complete(name)


##### AGENT #####


class Greeter(Agent):
    """Agent that runs CollectName task first, then chats normally."""

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "Eres un asistente amable. Hablas SIEMPRE en español. "
                "Respuestas cortas y claras. No uses emojis ni markdown."
            ),
        )

    async def on_enter(self) -> None:
        # Run the task — blocks until complete, returns typed str
        user_name = await CollectName()
        print(f"\n→ AGENT RESUMED — user_name={user_name!r}\n")

        # Now continue with the name in context
        self.session.generate_reply(
            instructions=(
                f"Ya sabes que el usuario se llama {user_name}. "
                f"Dile '¡Encantado, {user_name}!' y pregúntale en qué puedes ayudarle."
            ),
        )


##### SERVER #####

server = AgentServer()


@server.rtc_session(agent_name="task_demo")
async def entrypoint(ctx: JobContext) -> None:
    vad_kwargs: dict = {}
    if _VAD_MODEL.exists():
        vad_kwargs["onnx_file_path"] = _VAD_MODEL

    session = AgentSession(
        stt=EVoiceSTT(language="es"),
        llm=google.LLM(model="gemini-2.0-flash"),
        tts=EVoiceTTS(voice="ef_dora"),
        vad=silero.VAD.load(**vad_kwargs),
        turn_handling=TurnHandlingOptions(
            turn_detection=MultilingualModel(),
            aec_warmup_duration=3.0,
        ),
        preemptive_generation=True,
    )
    await session.start(agent=Greeter(), room=ctx.room)


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "console"]
    cli.run_app(server)
