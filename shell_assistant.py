from brainsoft_code_challenge.utils import load_environment

load_environment()

import argparse  # noqa: E402
import asyncio  # noqa: E402
import contextlib  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402
from collections.abc import Sequence  # noqa: E402
from typing import TYPE_CHECKING  # noqa: E402

from prompt_toolkit import PromptSession  # noqa: E402
from prompt_toolkit.styles import Style  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.live import Live  # noqa: E402
from rich.markdown import Markdown  # noqa: E402

from brainsoft_code_challenge.config import (  # noqa: E402
    DEFAULT_FREQUENCY_PENALTY,
    DEFAULT_MODEL,
    DEFAULT_PRESENCE_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    MAX_FREQUENCY_PENALTY,
    MAX_PRESENCE_PENALTY,
    MAX_TEMPERATURE,
    MAX_TOP_P,
    MIN_FREQUENCY_PENALTY,
    MIN_PRESENCE_PENALTY,
    MIN_TEMPERATURE,
    MIN_TOP_P,
    MODEL_CHOICES,
)

if TYPE_CHECKING:
    from langchain.agents import AgentExecutor

    from brainsoft_code_challenge.files import InputFile

console = Console()
session = PromptSession()  # type: ignore


def display_intro() -> None:
    intro_text = """
# Generative AI Python SDK Assistant

Type **load [file ..]** to include the specified file(s) with the next message.
Type **quit** to exit the application.
    """
    markdown = Markdown(intro_text)
    console.print(markdown)
    console.print()


def read_attached_files(file_names: Sequence[str]) -> list["InputFile"]:
    from brainsoft_code_challenge.files import InputFile, UnsupportedFileTypeError, process_csv, read_pdf_file  # noqa: E402

    input_files = []
    for file_name in file_names:
        try:
            if file_name.endswith(".csv"):
                with open(file_name) as file:
                    content = process_csv(file)
            elif file_name.endswith(".pdf"):
                content = read_pdf_file(file_name)
            else:
                raise UnsupportedFileTypeError("Unsupported file type")
        except Exception as e:
            message = f"Could not load file {file_name}: {e}"
            console.print(Markdown(f"**System:** {message}"))
            return []
        input_file = InputFile(name=file_name, content=content)
        input_files.append(input_file)
    return input_files


async def conversation_loop(agent_executor: "AgentExecutor", user_input: str, prompt_style: Style) -> None:
    from brainsoft_code_challenge.agent import build_agent_input

    input_files = []
    while True:
        if user_input.strip().lower() == "quit":
            sys.exit(0)
        split_input = user_input.lower().split()
        if split_input and split_input[0] == "load":
            file_names = split_input[1:]
            input_files = read_attached_files(file_names)
            if not input_files:
                message = "No files will be attached with the next message."
            elif len(input_files) == 1:
                message = "1 file will be attached with the next message."
            else:
                message = f"{len(input_files)} files will be attached with the next message."
            console.print(Markdown(f"**System:** {message}"))
        else:
            full_response = "**Assistant:** "
            response_markdown = Markdown(full_response)
            with Live(response_markdown, console=console, auto_refresh=True) as live:
                async for event in agent_executor.astream_events(build_agent_input(user_input, input_files), version="v1"):
                    if event["event"] == "on_chat_model_stream":
                        full_response += event["data"]["chunk"].content
                        live.update(Markdown(full_response))
            input_files = []
        user_input = await session.prompt_async("User: ", style=prompt_style)


async def run(model: str, temperature: float, frequency_penalty: float, presence_penalty: float, top_p: float) -> None:
    display_intro()

    console.print("How can I help you?")

    prompt_style = Style.from_dict({"prompt": "bold"})
    user_input = await session.prompt_async("User: ", style=prompt_style)

    from brainsoft_code_challenge.agent import get_agent_executor

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        agent_executor = get_agent_executor(model, temperature, frequency_penalty, presence_penalty, top_p, verbose=False)
        await conversation_loop(agent_executor, user_input, prompt_style)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the assistant in the shell")
    parser.add_argument("--model", type=str, help="Model to use", choices=MODEL_CHOICES, default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, help="Temperature", default=DEFAULT_TEMPERATURE)
    parser.add_argument("--frequency-penalty", type=float, help="Frequency penalty", default=DEFAULT_FREQUENCY_PENALTY)
    parser.add_argument("--presence-penalty", type=float, help="Presence penalty", default=DEFAULT_PRESENCE_PENALTY)
    parser.add_argument("--top-p", type=float, help="Top-p", default=DEFAULT_TOP_P)
    args = parser.parse_args()

    if not MIN_TEMPERATURE <= args.temperature <= MAX_TEMPERATURE:
        print(f"Temperature must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}")
        sys.exit(1)
    if not MIN_FREQUENCY_PENALTY <= args.frequency_penalty <= MAX_FREQUENCY_PENALTY:
        print(f"Frequency penalty must be between {MIN_FREQUENCY_PENALTY} and {MAX_FREQUENCY_PENALTY}")
        sys.exit(1)
    if not MIN_PRESENCE_PENALTY <= args.presence_penalty <= MAX_PRESENCE_PENALTY:
        print(f"Presence penalty must be between {MIN_PRESENCE_PENALTY} and {MAX_PRESENCE_PENALTY}")
        sys.exit(1)
    if not MIN_TOP_P <= args.top_p <= MAX_TOP_P:
        print(f"Top-p must be between {MIN_TOP_P} and {MAX_TOP_P}")
        sys.exit(1)

    with contextlib.suppress(KeyboardInterrupt, EOFError):
        asyncio.run(run(args.model, args.temperature, args.frequency_penalty, args.presence_penalty, args.top_p))
