"""Parker — researcher agent with ReactiveState tracking.

make script prototype
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from typing import Any

import httpx
import trafilatura
from core.models import (
    STATE_COLOR,
    AgentMode,
    AgentState,
    ReactiveState,
    SessionMode,
    Signal,
    UserMode,
)
from livekit.agents import (
    Agent,
    AgentServer,
    ChatContext,
    ChatMessage,
    JobContext,
    RunContext,
    StopResponse,
    TurnHandlingOptions,
    cli,
    function_tool,
)
from livekit.agents.voice import AgentSession
from livekit.agents.voice.events import (
    AgentStateChangedEvent,
    CloseEvent,
    FunctionToolsExecutedEvent,
    UserStateChangedEvent,
)
from livekit.plugins import google, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from e_agents.rtc.adapters.stt.evoice import EVoiceSTT
from e_agents.rtc.adapters.tts.evoice import EVoiceTTS
from e_agents.shared.adapters.searxng import SearXNGAdapter
from e_agents.shared.core.settings import settings as st

logging.getLogger("google_genai.models").setLevel(logging.WARNING)

_RST = "\033[0m"
_DIM = "\033[2m"


##### LOG #####


def _emit(color: str, tag: str, event: str, **kw: Any) -> None:
    """Colored log — color comes from model class attributes."""
    extra = " ".join(f"{k}={v}" for k, v in kw.items())
    detail = f" {_DIM}{extra}{_RST}" if extra else ""
    print(f"{_DIM}{time.strftime('%H:%M:%S')}{_RST} {color}[{tag}]{_RST} {event}{detail}")


##### DISPATCHERS #####

_AGENT_MODE_MAP: dict[str, AgentMode] = {
    "initializing": AgentMode.IDLE,
    "idle": AgentMode.IDLE,
    "listening": AgentMode.LISTENING,
    "thinking": AgentMode.THINKING,
    "speaking": AgentMode.SPEAKING,
}

_AGENT_TIMESTAMP: dict[str, str] = {
    "thinking": "thought_at",
    "speaking": "spoke_at",
    "idle": "finished_at",
}

_USER_MODE_MAP: dict[str, UserMode] = {
    "speaking": UserMode.SPEAKING,
    "listening": UserMode.SILENT,
    "away": UserMode.AWAY,
}

_USER_TIMESTAMP: dict[str, tuple[str, ...]] = {
    "speaking": ("spoke_at", "active_at"),
    "listening": ("stopped_at",),
}

_USER_SIGNAL: dict[str, Signal] = {
    "speaking": Signal.USER_SPOKE,
    "listening": Signal.USER_STOPPED,
    "away": Signal.USER_LEFT,
}


##### SESSION #####


class ReactiveSession(AgentSession[ReactiveState]):
    """AgentSession that auto-registers ReactiveState tracking hooks."""

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.on("agent_state_changed")(self._rs_on_agent_state)
        self.on("user_state_changed")(self._rs_on_user_state)
        self.on("user_input_transcribed")(self._rs_on_transcript)
        self.on("function_tools_executed")(self._rs_on_tools)
        self.on("close")(self._rs_on_close)

    @property
    def state(self) -> ReactiveState:
        return self.userdata

    def _rs_on_agent_state(self, ev: AgentStateChangedEvent) -> None:
        self.state.current.mode = _AGENT_MODE_MAP[ev.new_state]
        if ts_field := _AGENT_TIMESTAMP.get(ev.new_state):
            setattr(self.state.current, ts_field, time.monotonic())
        _emit(AgentMode._color, "agent", f"{ev.old_state} → {ev.new_state}", mode=self.state.current.mode)

    def _rs_on_user_state(self, ev: UserStateChangedEvent) -> None:
        now = time.monotonic()
        self.state.user.mode = _USER_MODE_MAP[ev.new_state]
        for attr in _USER_TIMESTAMP.get(ev.new_state, ()):
            setattr(self.state.user, attr, now)
        sig = _USER_SIGNAL[ev.new_state]
        kw: dict[str, Any] = {}
        if ev.new_state == "speaking" and self.state.current.mode == AgentMode.SPEAKING:
            self.state.user.interrupts += 1
            self.state.current.interruptions += 1
            sig = Signal.USER_BARGED_IN
            kw["interrupts"] = self.state.user.interrupts
        _emit(sig.color, "user", sig, **kw)

    def _rs_on_transcript(self, ev: Any) -> None:
        is_final = getattr(ev, "is_final", False)
        transcript = getattr(ev, "transcript", "")
        if not is_final:
            _emit(_DIM, "stt", f"interim: {transcript[:50]!r}")
            return
        self.state.user.transcript_at = time.monotonic()
        _emit(Signal.USER_SPOKE.color, "stt", f"final: {transcript[:60]!r}")

    def _rs_on_tools(self, ev: FunctionToolsExecutedEvent) -> None:
        self.state.current.tool_calls += 1
        self.state.current.tool_called_at = time.monotonic()
        sig = Signal.AGENT_TOOL_RESULT
        for call, output in ev.zipped():
            result = str(output.output)[:40] if output else "—"
            _emit(sig.color, "tool", sig, tool=call.name, result=repr(result))
        _emit(STATE_COLOR, "state", "tool_calls", total=self.state.current.tool_calls)

    def _rs_on_close(self, ev: CloseEvent) -> None:
        self.state.session = SessionMode.ENDING
        sig = Signal.SESSION_ENDING
        _emit(
            sig.color,
            "session",
            sig,
            reason=ev.reason,
            duration=f"{self.state.session_duration:.0f}s",
            turns=self.state.turn_count,
            tools=self.state.current.tool_calls,
            interrupts=self.state.user.interrupts,
        )


##### GATE #####

_GATE_COOLDOWN = 10.0  # seconds to suppress turns after gate fires
_GATE_DELAY = 2.0  # seconds to wait for STT to settle before generate_reply


async def _delayed_gate_reply(
    session: AgentSession[ReactiveState], instructions: str,
) -> None:
    """Wait for STT FINALs to settle, then fire generate_reply."""
    _emit(_INNER_COLOR, "gate", f"reply scheduled in {_GATE_DELAY}s")
    await asyncio.sleep(_GATE_DELAY)
    _emit(_INNER_COLOR, "gate", "CALLBACK → generate_reply")
    session.interrupt()
    session.generate_reply(instructions=instructions)


##### INNER LOOP — DEEP FETCH #####

_DEEP_FETCH_TOP = 2
_DEEP_FETCH_MAX_CHARS = 3000
_INNER_COLOR = "\033[93m"  # bright yellow


_DEEP_FETCH_DELAY = 10.0  # TODO(temp): artificial delay to observe outer/inner separation


async def _deep_fetch(session: AgentSession[ReactiveState], urls: list[str]) -> None:
    """Inner loop: fetch full content from top URLs, then callback to outer loop."""
    state: ReactiveState = session.userdata
    state.tasks_running += 1
    _emit(_INNER_COLOR, "inner", f"DEEP_FETCH started (delay={_DEEP_FETCH_DELAY}s)", urls=len(urls))
    await asyncio.sleep(_DEEP_FETCH_DELAY)

    contents: list[str] = []
    for url in urls[:_DEEP_FETCH_TOP]:
        try:
            _emit(_INNER_COLOR, "inner", f"fetching {url[:60]}")
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(15.0),
                follow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0 (compatible; e-agents/1.0)"},
            ) as client:
                resp = await client.get(url)
                resp.raise_for_status()
            text = trafilatura.extract(
                resp.text, include_links=True, include_tables=True,
                favor_recall=True, deduplicate=True,
            )
            if text:
                contents.append(f"[{url}]\n{text[:_DEEP_FETCH_MAX_CHARS]}")
                _emit(_INNER_COLOR, "inner", f"FETCH_OK {len(text)} chars", url=url[:40])
            else:
                _emit(_INNER_COLOR, "inner", "FETCH_EMPTY", url=url[:40])
        except Exception as exc:
            _emit(_INNER_COLOR, "inner", f"FETCH_ERROR {type(exc).__name__}", url=url[:40])

    state.tasks_running -= 1

    if not contents:
        _emit(_INNER_COLOR, "inner", "DEEP_FETCH done — no content extracted")
        return

    merged = "\n\n---\n\n".join(contents)
    _emit(_INNER_COLOR, "inner", f"DEEP_FETCH done — {len(contents)} sources, CALLBACK →")

    # Callback protocol: interrupt + generate_reply
    session.interrupt()
    session.generate_reply(
        instructions=(
            "Has leído en profundidad las fuentes de la búsqueda anterior. "
            "Comparte con el usuario los detalles más relevantes que has encontrado. "
            "Sé específico con datos, nombres y hechos. "
            "No digas que leíste fuentes — simplemente amplía tu respuesta anterior "
            "con la información nueva.\n\n"
            f"Contenido de las fuentes:\n{merged}"
        ),
    )


##### AGENT #####


class Parker(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "Eres Parker, un investigador experto y solucionador de problemas. "
                "Hablas SIEMPRE en español.\n"
                "Tu fortaleza es descomponer problemas complejos en pasos claros, "
                "buscar información actualizada en la web y sintetizar respuestas precisas.\n"
                "Cuando el usuario plantea un problema, primero analiza la raíz, "
                "luego investiga si necesitas datos frescos usando tus herramientas de búsqueda.\n"
                "Respuestas directas, estructuradas y sin rodeos. "
                "No uses emojis, asteriscos ni markdown."
            ),
        )

    @function_tool()
    async def research(
        self, context: RunContext[ReactiveState], query: str,
    ) -> str:
        """Busca información en la web sobre un tema.

        Args:
            query: La consulta de búsqueda.
        """
        _emit(_INNER_COLOR, "inner", f"SEARCH query={query!r}")
        results = await SearXNGAdapter.query(query, max_results=5)

        if not results:
            return "No se encontraron resultados. Intenta con otra consulta."

        # Extract URLs for background deep fetch (inner loop)
        urls = [r.url for r in results if r.url]
        if urls:
            asyncio.create_task(_deep_fetch(context.session, urls))
            _emit(_INNER_COLOR, "inner", f"DEEP_FETCH launched → {len(urls)} URLs")

        # Return snippets immediately to LLM (outer loop continues)
        body = "\n\n".join(str(r) for r in results)
        return (
            f"<search_result query='{query}' count='{len(results)}'>\n"
            f"{body}\n"
            "</search_result>\n"
            "Sintetiza TODOS los resultados en una respuesta completa. "
            "Incluye datos específicos, nombres y hechos. "
            "En unos segundos tendrás más detalles de las fuentes — "
            "el usuario ya lo sabe, no hace falta avisarle."
        )

    async def on_enter(self) -> None:
        state: ReactiveState = self.session.userdata
        state.current = AgentState(name="parker", entered_at=time.monotonic(), mode=AgentMode.LISTENING)
        state.session = SessionMode.ACTIVE
        sig = Signal.SESSION_STARTED
        _emit(sig.color, "session", sig, agent="parker")
        self.session.generate_reply(
            instructions="Preséntate brevemente como Parker y pregunta en qué puedes ayudar.",
        )

    async def on_exit(self) -> None:
        state: ReactiveState = self.session.userdata
        state.session = SessionMode.ENDING
        sig = Signal.SESSION_ENDING
        _emit(sig.color, "session", sig)

    async def on_user_turn_completed(
        self,
        turn_ctx: ChatContext,
        new_message: ChatMessage,
    ) -> None:
        state: ReactiveState = self.session.userdata

        # Gate cooldown: suppress turns while a forced reply is settling
        if time.monotonic() < state.data.get("_gate_cooldown", 0.0):
            raise StopResponse()

        # TODO(temp): PoC wake-word "perico" — settle-based gate
        # WORKAROUND: _wake_pos needed until e-voice server deploys VAD segmentation
        full = (new_message.text_content or "").lower()
        checked_pos = state.data.get("_wake_pos", 0)
        state.data["_wake_pos"] = len(full)
        if "perico" in full[checked_pos:]:
            _emit(Signal.USER_SPOKE.color, "wake", "PERICO detected — delayed reply")
            state.data["_gate_cooldown"] = time.monotonic() + _GATE_COOLDOWN
            asyncio.create_task(_delayed_gate_reply(
                self.session,
                f"Dile al usuario exactamente esto: {state.fake_reactive_sentence}",
            ))
            raise StopResponse()

        state.user.messages += 1
        state.register_turn()
        text = (new_message.text_content or "")[:60]
        sig = Signal.USER_SPOKE
        _emit(sig.color, "user", sig, messages=state.user.messages, text=repr(text))
        _emit(STATE_COLOR, "state", "turn", count=state.turn_count, agent_turns=state.current.turns)


##### SERVER #####

server = AgentServer()


@server.rtc_session(agent_name="proto")
async def entrypoint(ctx: JobContext) -> None:
    state = ReactiveState(current=AgentState(name="parker"))

    session = ReactiveSession(
        userdata=state,
        stt=EVoiceSTT(language="es"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=EVoiceTTS(voice="em_alex"),
        vad=silero.VAD.load(onnx_file_path=st.VAD_MODEL_PATH),
        turn_handling=TurnHandlingOptions(
            turn_detection=MultilingualModel(),
            aec_warmup_duration=3.0,
        ),
        preemptive_generation=True,
    )
    await session.start(agent=Parker(), room=ctx.room)


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "console"]
    cli.run_app(server)
