"""Agente simple — voz con evoice STT/TTS.

    make download      # descargar modelos (una vez)
    make script simple
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

# Redirect HuggingFace cache to data/models/turn/ for turn detector
_BASE_DIR = Path(__file__).resolve().parent.parent
_TURN_CACHE = _BASE_DIR / "data" / "models" / "turn"
_VAD_MODEL = _BASE_DIR / "data" / "models" / "vad" / "silero_vad.onnx"
os.environ.setdefault("HF_HUB_CACHE", str(_TURN_CACHE))

logging.getLogger("google_genai.models").setLevel(logging.WARNING)

from livekit.agents import Agent, AgentServer, JobContext, TurnHandlingOptions, cli
from livekit.agents.voice import AgentSession
from livekit.plugins import google, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from e_agents.rtc.adapters.stt.evoice import EVoiceSTT
from e_agents.rtc.adapters.tts.evoice import EVoiceTTS

##### AGENT #####


class Ana(Agent):

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "Eres Ana, una asistente amable y directa. Hablas SIEMPRE en español.\n"
                "Conversa con naturalidad. Respuestas cortas y claras.\n"
                "No uses emojis, asteriscos ni markdown."
            ),
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(instructions="Saluda al usuario brevemente.")


##### SERVER #####

server = AgentServer()


@server.rtc_session(agent_name="simple")
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
    await session.start(agent=Ana(), room=ctx.room)


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "console"]
    cli.run_app(server)
