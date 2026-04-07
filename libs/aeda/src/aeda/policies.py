"""AEDA — built-in policies.

Sync (fast evaluation, no I/O). Each returns zero or more Decisions.

Taxonomy (N1):
  Entities:  IdlePolicy, TaskCompletedPolicy, EscalationPolicy, BudgetPolicy
  Verb:      evaluate (the single method on Policy protocol)
"""

from __future__ import annotations

from aeda.models import Command, Decision, Event, Signal
from aeda.state import ReactiveState


class IdlePolicy:
    """On TICK, if no activity for `timeout` seconds → SAY prompt.

    Resets when any meaningful activity signal is observed.
    """

    __slots__ = ("_timeout", "_text", "_notified")

    def __init__(self, timeout: float = 30.0, text: str = "Are you still there?") -> None:
        self._timeout = timeout
        self._text = text
        self._notified = False

    def evaluate(self, state: ReactiveState, event: Event) -> tuple[Decision, ...]:
        match event.signal:
            case (
                Signal.AGENT_RESPONDING
                | Signal.TASK_COMPLETED
                | Signal.MODEL_RESPONDED
                | Signal.TURN_ENDED
                | Signal.STATE_CHANGED
            ):
                self._notified = False
                return ()

        if event.signal != Signal.TICK:
            return ()

        if state.inactive_for > self._timeout and not self._notified:
            self._notified = True
            return (Decision(command=Command.SAY, payload={"text": self._text}),)
        return ()


class TaskCompletedPolicy:
    """On TASK_COMPLETED/FAILED, interrupt current output and share results.

    Cooldown prevents infinite loops when LLM re-triggers the same tool.
    """

    __slots__ = ("_cooldown", "_last_handled")

    def __init__(self, cooldown: float = 5.0) -> None:
        self._cooldown = cooldown
        self._last_handled: float = 0.0

    def evaluate(self, state: ReactiveState, event: Event) -> tuple[Decision, ...]:
        match event.signal:
            case Signal.TASK_COMPLETED:
                if event.timestamp - self._last_handled < self._cooldown:
                    return ()
                self._last_handled = event.timestamp
                msg = event.payload.get("message", event.payload.get("result", "Task completed."))
                if isinstance(msg, dict):
                    msg = str(msg)
                return (
                    Decision(command=Command.INTERRUPT),
                    Decision(
                        command=Command.RESPOND,
                        payload={"instructions": f"Share this result naturally: {msg}"},
                    ),
                )
            case Signal.TASK_FAILED:
                if event.timestamp - self._last_handled < self._cooldown:
                    return ()
                self._last_handled = event.timestamp
                error = event.payload.get("error", "Unknown error.")
                return (
                    Decision(
                        command=Command.RESPOND,
                        payload={"instructions": f"A background task failed: {error}. Inform the user."},
                    ),
                )
            case _:
                return ()


class EscalationPolicy:
    """After N turns → SET_PROMPT + SWAP to escalation agent."""

    __slots__ = ("_threshold", "_target", "_escalated")

    def __init__(self, threshold: int = 10, target_agent: str = "escalation") -> None:
        self._threshold = threshold
        self._target = target_agent
        self._escalated = False

    def evaluate(self, state: ReactiveState, event: Event) -> tuple[Decision, ...]:
        if self._escalated:
            return ()

        if event.signal not in {Signal.AGENT_IDLE, Signal.TURN_ENDED}:
            return ()

        if state.turn_count >= self._threshold:
            self._escalated = True
            return (
                Decision(
                    command=Command.SET_PROMPT,
                    payload={"prompt": "Conversation is taking too long. Wrap up or escalate."},
                ),
                Decision(command=Command.SWAP, payload={"agent_id": self._target}),
            )
        return ()


class BudgetPolicy:
    """Emit BUDGET_WARNING at threshold, force HOLD at exhaustion.

    Evaluates on MODEL_RESPONDED (when tokens_used is updated) and on TICK.
    """

    __slots__ = ("_warn_ratio", "_warned", "_exhausted")

    def __init__(self, warn_ratio: float = 0.9) -> None:
        self._warn_ratio = warn_ratio
        self._warned = False
        self._exhausted = False

    def evaluate(self, state: ReactiveState, event: Event) -> tuple[Decision, ...]:
        if self._exhausted or state.token_budget == 0:
            return ()

        if event.signal not in {Signal.MODEL_RESPONDED, Signal.TICK}:
            return ()

        ratio = state.budget_ratio

        if ratio >= 1.0 and not self._exhausted:
            self._exhausted = True
            return (
                Decision(
                    command=Command.SAY,
                    payload={"text": "Token budget exhausted. Stopping."},
                ),
                Decision(command=Command.HOLD),
            )

        if ratio >= self._warn_ratio and not self._warned:
            self._warned = True
            return (
                Decision(
                    command=Command.SET_PROMPT,
                    payload={"prompt": f"Budget at {ratio:.0%}. Be concise, wrap up soon."},
                ),
            )

        return ()
