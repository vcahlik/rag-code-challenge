import asyncio

from brainsoft_code_challenge.config import DEFAULT_FREQUENCY_PENALTY, DEFAULT_MODEL, DEFAULT_PRESENCE_PENALTY, DEFAULT_TEMPERATURE, DEFAULT_TOP_P
from shell_assistant import run


def test_smoke() -> None:
    asyncio.run(run(DEFAULT_MODEL, DEFAULT_TEMPERATURE, DEFAULT_FREQUENCY_PENALTY, DEFAULT_PRESENCE_PENALTY, DEFAULT_TOP_P))
