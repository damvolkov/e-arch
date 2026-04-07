"""Unit test conftest — fake tools and agent responses."""

from __future__ import annotations

import asyncio

import pytest


##### FAKE TOOLS #####


async def fake_fast_tool(query: str = "test") -> dict[str, str]:
    """50ms latency."""
    await asyncio.sleep(0.05)
    return {"query": query, "result": "fast_result"}


async def fake_slow_tool(topic: str = "research") -> dict[str, str]:
    """200ms latency."""
    await asyncio.sleep(0.2)
    return {"topic": topic, "result": "slow_result", "items": ["a", "b", "c"]}


async def fake_failing_tool() -> None:
    """Always raises."""
    await asyncio.sleep(0.05)
    msg = "Connection timeout"
    raise TimeoutError(msg)


##### FAKE AGENT RESPONSES #####

AGENT_RESPONSES: list[str] = [
    "I'll look into that for you.",
    "Let me check the details.",
    "Here's what I found.",
    "Based on my analysis, the answer is 42.",
    "I need to run a deeper search.",
]


@pytest.fixture()
def agent_responses() -> list[str]:
    return list(AGENT_RESPONSES)
