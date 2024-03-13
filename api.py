from brainsoft_code_challenge.utils import load_environment

load_environment()

import base64  # noqa: E402
import os  # noqa: E402
import tempfile  # noqa: E402
from collections.abc import Mapping, Sequence  # noqa: E402
from enum import Enum  # noqa: E402
from typing import Any  # noqa: E402

from fastapi import FastAPI, HTTPException  # noqa: E402
from langchain.agents import AgentExecutor  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from brainsoft_code_challenge.agent import MemoryContextType, build_agent_input, get_agent_executor  # noqa: E402
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
from brainsoft_code_challenge.files import InputFile, UnsupportedFileTypeError, process_csv, read_pdf_file  # noqa: E402
from brainsoft_code_challenge.tokenizer import count_tokens, get_memory_token_limit  # noqa: E402

app = FastAPI()


class InvalidInputError(ValueError):
    pass


class MessageType(Enum):
    HUMAN = "human"
    AI = "ai"


class MessagePayload(BaseModel):
    type: str
    content: str


class FilePayload(BaseModel):
    file_name: str
    content: str


class ChatRequestPayload(BaseModel):
    user_input: str
    history: list[MessagePayload] | None = None
    files: list[FilePayload] | None = None
    model: str = DEFAULT_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    frequency_penalty: float = DEFAULT_FREQUENCY_PENALTY
    presence_penalty: float = DEFAULT_PRESENCE_PENALTY
    top_p: float = DEFAULT_TOP_P
    return_history: bool = False

    class Config:
        extra = "forbid"


def __parse_history(history: Sequence[Mapping[str, str]], model: str) -> list[MemoryContextType]:
    """
    Parses the history into LangChain memory contexts. The total length of the history is checked as well, as initial summarization
    (before the agent even outputs its response) could take a very long time.

    :param history: The history from the request payload.
    :return: LangChain memory contexts.
    """
    total_token_length = sum([count_tokens(message["content"]) for message in history])
    if total_token_length > get_memory_token_limit(model):
        raise InvalidInputError(f"The history's length of {total_token_length} tokens exceeds the maximum length of {get_memory_token_limit(model)} tokens.")
    contexts = []
    if history and history[0]["type"] == MessageType.AI.value:
        history = [{"type": MessageType.HUMAN.value, "content": ""}] + list(history)
    for i in range(0, len(history), 2):
        odd_message = history[i]
        even_message = history[i + 1]
        if odd_message["type"] != MessageType.HUMAN.value or even_message["type"] != MessageType.AI.value:
            raise InvalidInputError('The history must be composed of alternating "human" and "ai" messages, with the last message being of type "ai".')
        context = ({"input": str(odd_message["content"])}, {"output": str(even_message["content"])})
        contexts.append(context)
    return contexts


def __validate_config(payload: Mapping[str, Any]) -> None:
    """
    Validates the model configuration from the request payload dictionary. Raises a InvalidInputError upon failure.

    :param payload: The request payload dictionary.
    """
    if payload["model"] not in MODEL_CHOICES:
        raise InvalidInputError(f"Model must be one of {MODEL_CHOICES}")
    if not MIN_TEMPERATURE <= payload["temperature"] <= MAX_TEMPERATURE:
        raise InvalidInputError(f"Temperature must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}")
    if not MIN_FREQUENCY_PENALTY <= payload["frequency_penalty"] <= MAX_FREQUENCY_PENALTY:
        raise InvalidInputError(f"Frequency penalty must be between {MIN_FREQUENCY_PENALTY} and {MAX_FREQUENCY_PENALTY}")
    if not MIN_PRESENCE_PENALTY <= payload["presence_penalty"] <= MAX_PRESENCE_PENALTY:
        raise InvalidInputError(f"Presence penalty must be between {MIN_PRESENCE_PENALTY} and {MAX_PRESENCE_PENALTY}")
    if not MIN_TOP_P <= payload["top_p"] <= MAX_TOP_P:
        raise InvalidInputError(f"Top-p must be between {MIN_TOP_P} and {MAX_TOP_P}")


def __read_attached_files(file_payloads: list[Mapping[str, str]]) -> list[InputFile]:
    """
    Reads the attached base64-encoded files from the request payload.

    :param file_payloads: The file payloads from the request payload.
    :return: A list of InputFile objects.
    """
    if file_payloads is None:
        return []
    input_files = []
    for file_payload in file_payloads:
        file_name = file_payload["file_name"]
        error = None
        try:
            raw_content = base64.b64decode(file_payload["content"])
            if file_name.endswith(".csv"):
                parsed_content = process_csv(content=raw_content.decode("utf-8"))
            elif file_name.endswith(".pdf"):
                fd, tmp_filename = tempfile.mkstemp(suffix=".pdf")
                with os.fdopen(fd, "wb") as tmpfile:
                    tmpfile.write(raw_content)
                    tmpfile.flush()
                    parsed_content = read_pdf_file(tmp_filename)
            else:
                raise UnsupportedFileTypeError("Unsupported file type")
        except UnsupportedFileTypeError as e:
            raise e
        except Exception as e:
            raise InvalidInputError("An error occurred while reading the file contents.") from e
        input_file = InputFile(name=file_name, content=parsed_content, error=error)
        input_files.append(input_file)
    return input_files


@app.get("/")
def home() -> dict[str, str]:
    return {
        "Name": "Generative AI Python SDK Assistant API",
        "Status": "ok",
    }


def __get_history_from_agent_executor(agent_executor: AgentExecutor) -> list[dict[str, str]]:
    """
    Retrieves the chat history from the agent executor. As ConversationSummaryBufferMemory does not support initialization with a
    system message, the (potential) system message would be converted to a regular human or AI message.

    :param agent_executor: The agent executor holding the memory.
    :return: The chat history.
    """
    if agent_executor.memory is None:
        return []
    memory_variables = agent_executor.memory.load_memory_variables({})
    history = []
    summary = None
    for message in memory_variables["chat_history"]:
        if isinstance(message, HumanMessage):
            history.append({"type": MessageType.HUMAN.value, "content": message.content})
        elif isinstance(message, AIMessage):
            history.append({"type": MessageType.AI.value, "content": message.content})
        elif isinstance(message, SystemMessage):
            if summary is not None:
                continue  # This should never happen, there should only be a single system message
            summary = message.content
        else:
            raise InvalidInputError(f"Unknown message type: {str(type(message))}")
    if summary is not None:
        summary_message_type = MessageType.AI
        if history and history[0]["type"] == MessageType.AI.value:
            summary_message_type = MessageType.HUMAN
        summary_message = {"type": summary_message_type.value, "content": summary}
        history = [summary_message] + history
    return history  # type: ignore


@app.post("/chat")
def get_chat_response(payload: ChatRequestPayload) -> dict[str, Any]:
    """
    Get a response from the AI model using a POST request.

    :param payload: The request payload.
    :return: API response.
    """
    payload_dict = payload.model_dump()
    history = payload_dict["history"]
    if history is None:
        history = []

    try:
        __validate_config(payload_dict)
        input_files = __read_attached_files(payload_dict["files"])
        contexts = __parse_history(history, payload_dict["model"])
    except InvalidInputError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    agent_executor = get_agent_executor(
        payload_dict["model"],
        payload_dict["temperature"],
        payload_dict["frequency_penalty"],
        payload_dict["presence_penalty"],
        payload_dict["top_p"],
        verbose=False,
        memory_contexts=contexts,
    )
    agent_input, input_was_cut_off = build_agent_input(payload_dict["user_input"], input_files, payload_dict["model"])
    try:
        output = agent_executor.invoke(agent_input)
    except Exception as e:
        # In case of a public API, we should not expose the exception message
        raise HTTPException(status_code=500, detail=f"An error occurred while obtaining the agent response: {e}") from e

    response = {"input": agent_input["input"], "output": output["output"]}
    if payload_dict["return_history"]:
        response["history"] = __get_history_from_agent_executor(agent_executor)
    if input_was_cut_off:
        response["warning"] = "The input was too long and therefore was cut off."
    return response
