from brainsoft_code_challenge.utils import load_environment

load_environment()

import base64  # noqa: E402
import os  # noqa: E402
import tempfile  # noqa: E402
from collections.abc import Mapping, Sequence  # noqa: E402
from typing import Any  # noqa: E402

from fastapi import FastAPI, HTTPException  # noqa: E402
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
from brainsoft_code_challenge.files import InputFile, process_csv, read_pdf_file  # noqa: E402

app = FastAPI()


@app.get("/")
def home() -> dict[str, str]:
    return {
        "Name": "Generative AI Python SDK Assistant API",
        "Status": "ok",
    }


class MessagePayload(BaseModel):
    type: str
    content: str


class FilePayload(BaseModel):
    file_name: str
    content: str


class QueryPayload(BaseModel):
    user_input: str
    messages: list[MessagePayload] | None = None
    files: list[FilePayload] | None = None
    model: str = DEFAULT_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    frequency_penalty: float = DEFAULT_FREQUENCY_PENALTY
    presence_penalty: float = DEFAULT_PRESENCE_PENALTY
    top_p: float = DEFAULT_TOP_P


def parse_messages(messages: Sequence[Mapping[str, str]]) -> list[MemoryContextType]:
    contexts = []
    if not len(messages) % 2 == 0:
        raise ValueError("The number of messages must be even or zero.")
    for odd_message, even_message in zip(messages, messages[1:], strict=False):
        if odd_message["type"] != "human":
            raise ValueError('Every odd message must be of type "human".')
        if even_message["type"] != "ai":
            raise ValueError('Every even message must be of type "ai".')
        context = ({"input": str(odd_message["content"])}, {"output": str(even_message["content"])})
        contexts.append(context)
    return contexts


def validate_config(payload: Mapping[str, Any]) -> None:
    if payload["model"] not in MODEL_CHOICES:
        raise ValueError(f"Model must be one of {MODEL_CHOICES}")
    if not MIN_TEMPERATURE <= payload["temperature"] <= MAX_TEMPERATURE:
        raise ValueError(f"Temperature must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}")
    if not MIN_FREQUENCY_PENALTY <= payload["frequency_penalty"] <= MAX_FREQUENCY_PENALTY:
        raise ValueError(f"Frequency penalty must be between {MIN_FREQUENCY_PENALTY} and {MAX_FREQUENCY_PENALTY}")
    if not MIN_PRESENCE_PENALTY <= payload["presence_penalty"] <= MAX_PRESENCE_PENALTY:
        raise ValueError(f"Presence penalty must be between {MIN_PRESENCE_PENALTY} and {MAX_PRESENCE_PENALTY}")
    if not MIN_TOP_P <= payload["top_p"] <= MAX_TOP_P:
        raise ValueError(f"Top-p must be between {MIN_TOP_P} and {MAX_TOP_P}")


def read_attached_files(file_payloads: list[Mapping[str, str]]) -> list[InputFile]:
    if file_payloads is None:
        return []
    input_files = []
    for file_payload in file_payloads:
        file_name = file_payload["file_name"]
        raw_content = base64.b64decode(file_payload["content"])
        error = None

        try:
            if file_name.endswith(".csv"):
                parsed_content = process_csv(content=raw_content.decode("utf-8"))
            elif file_name.endswith(".pdf"):
                fd, tmp_filename = tempfile.mkstemp(suffix=".pdf")
                with os.fdopen(fd, "wb") as tmpfile:
                    tmpfile.write(raw_content)
                    tmpfile.flush()
                    parsed_content = read_pdf_file(tmp_filename)
            else:
                raise ValueError("Unsupported file type")
        except Exception as e:
            raise ValueError("An error occurred while reading the file contents.") from e
        input_file = InputFile(name=file_name, content=parsed_content, error=error)
        input_files.append(input_file)
    return input_files


@app.get("/chat")
def get_chat_response(payload: QueryPayload) -> dict[str, str]:
    payload_dict = payload.dict()
    messages = payload_dict["messages"]
    if messages is None:
        messages = []

    try:
        validate_config(payload_dict)
        input_files = read_attached_files(payload_dict["files"])
        contexts = parse_messages(messages)
    except ValueError as e:
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
    output = agent_executor.invoke(build_agent_input(payload_dict["user_input"], input_files))

    return {"output": output["output"]}
