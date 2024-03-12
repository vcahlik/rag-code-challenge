import asyncio
import os

import pytest

from brainsoft_code_challenge.config import DEFAULT_FREQUENCY_PENALTY, DEFAULT_MODEL, DEFAULT_PRESENCE_PENALTY, DEFAULT_TEMPERATURE, DEFAULT_TOP_P
from brainsoft_code_challenge.constants import PYTEST_USER_INPUT_ENV_VAR
from shell_assistant import run


def initialize_test_with_user_input(user_inputs: list[str]) -> int:
    for i, user_input in enumerate(user_inputs):
        os.environ[f"{PYTEST_USER_INPUT_ENV_VAR}_{i}"] = user_input
    return len(user_inputs)


def finalize_test_with_user_input(n_inputs: int) -> None:
    for i in range(n_inputs):
        del os.environ[f"{PYTEST_USER_INPUT_ENV_VAR}_{i}"]


def test_chat() -> None:
    n_inputs = initialize_test_with_user_input(["Who are you?", "quit"])
    with pytest.raises(SystemExit) as e:
        asyncio.run(run(DEFAULT_MODEL, DEFAULT_TEMPERATURE, DEFAULT_FREQUENCY_PENALTY, DEFAULT_PRESENCE_PENALTY, DEFAULT_TOP_P))
    assert e.value.code == 0  # noqa: S101
    finalize_test_with_user_input(n_inputs)


def test_chat_with_csv_file() -> None:
    n_inputs = initialize_test_with_user_input(
        ["load data/pytest/test_file.csv", "What information do you have about the file, exactly as it was provided to you?", "quit"]
    )
    with pytest.raises(SystemExit) as e:
        asyncio.run(run(DEFAULT_MODEL, DEFAULT_TEMPERATURE, DEFAULT_FREQUENCY_PENALTY, DEFAULT_PRESENCE_PENALTY, DEFAULT_TOP_P))
    assert e.value.code == 0  # noqa: S101
    finalize_test_with_user_input(n_inputs)


def test_chat_with_pdf_file() -> None:
    n_inputs = initialize_test_with_user_input(
        ["load data/pytest/test_file.pdf", "What information do you have about the file, exactly as it was provided to you?", "quit"]
    )
    with pytest.raises(SystemExit) as e:
        asyncio.run(run(DEFAULT_MODEL, DEFAULT_TEMPERATURE, DEFAULT_FREQUENCY_PENALTY, DEFAULT_PRESENCE_PENALTY, DEFAULT_TOP_P))
    assert e.value.code == 0  # noqa: S101
    finalize_test_with_user_input(n_inputs)


def test_chat_with_files() -> None:
    n_inputs = initialize_test_with_user_input(
        ["load data/pytest/test_file.csv data/pytest/test_file.pdf", "What information do you have about the files, exactly as it was provided to you?", "quit"]
    )
    with pytest.raises(SystemExit) as e:
        asyncio.run(run(DEFAULT_MODEL, DEFAULT_TEMPERATURE, DEFAULT_FREQUENCY_PENALTY, DEFAULT_PRESENCE_PENALTY, DEFAULT_TOP_P))
    assert e.value.code == 0  # noqa: S101
    finalize_test_with_user_input(n_inputs)
