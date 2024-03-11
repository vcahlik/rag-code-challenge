import os
import tempfile
from collections.abc import Mapping, Sequence
from typing import Any

import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from streamlit.runtime.uploaded_file_manager import UploadedFile

from brainsoft_code_challenge.agent import build_agent_input, get_agent_executor
from brainsoft_code_challenge.config import (
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
from brainsoft_code_challenge.constants import ACTION_HINTS
from brainsoft_code_challenge.files import InputFile, process_csv, read_pdf_file


def initialize_chat() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "model_config" not in st.session_state:
        st.session_state.model_config = {
            "model": DEFAULT_MODEL,
            "temperature": DEFAULT_TEMPERATURE,
            "frequency_penalty": DEFAULT_FREQUENCY_PENALTY,
            "presence_penalty": DEFAULT_PRESENCE_PENALTY,
            "top_p": DEFAULT_TOP_P,
        }
    if "agent_executor" not in st.session_state:
        st.session_state.agent_executor = get_agent_executor(
            DEFAULT_MODEL, DEFAULT_TEMPERATURE, DEFAULT_FREQUENCY_PENALTY, DEFAULT_PRESENCE_PENALTY, DEFAULT_TOP_P, verbose=True
        )


def reset_chat(model: str, temperature: float, frequency_penalty: float, presence_penalty: float, top_p: float) -> None:
    st.session_state.messages = []
    st.session_state.model_config = {
        "model": model,
        "temperature": temperature,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "top_p": top_p,
    }
    st.session_state.agent_executor = get_agent_executor(model, temperature, frequency_penalty, presence_penalty, top_p, verbose=True)


def prepare_page() -> None:
    st.set_page_config(layout="wide", page_title="Generative AI Python SDK Assistant")
    initialize_chat()

    with st.sidebar:
        model_config = st.session_state.model_config
        st.title("Configuration")
        model = st.selectbox("Model", MODEL_CHOICES, index=MODEL_CHOICES.index(model_config["model"]))
        temperature = st.slider("Temperature", MIN_TEMPERATURE, MAX_TEMPERATURE, model_config["temperature"])
        frequency_penalty = st.slider("Frequency penalty", MIN_FREQUENCY_PENALTY, MAX_FREQUENCY_PENALTY, model_config["frequency_penalty"])
        presence_penalty = st.slider("Presence penalty", MIN_PRESENCE_PENALTY, MAX_PRESENCE_PENALTY, model_config["presence_penalty"])
        top_p = st.slider("Top-p", MIN_TOP_P, MAX_TOP_P, model_config["top_p"])
        col1, col2 = st.columns(2)
        reset_chat_args = (model, temperature, frequency_penalty, presence_penalty, top_p)
        if (
            model_config["model"] != model
            or model_config["temperature"] != temperature
            or model_config["frequency_penalty"] != frequency_penalty
            or model_config["presence_penalty"] != presence_penalty
            or model_config["top_p"] != top_p
        ):
            col1.button("Apply & reset chat", type="primary", on_click=reset_chat, args=reset_chat_args)
        else:
            col1.button("Reset chat", on_click=reset_chat, args=reset_chat_args)


def render_actions(actions: Sequence[Mapping[str, str]], element: DeltaGenerator | None = None) -> None:
    for action in actions:
        hint = ACTION_HINTS[action["tool"]]
        query = action["query"]
        expander = st.expander(f"{hint}: {query}", expanded=False) if element is None else element.expander(f"{hint}: {query}", expanded=False)
        with expander:
            st.text(action["output"])


def read_attached_files(buffers: Sequence[UploadedFile] | None) -> list[InputFile]:
    if buffers is None:
        return []
    input_files = []
    for buffer in buffers:
        content = ""
        error = None
        buffer.seek(0)
        file_name = buffer.name
        try:
            if file_name.endswith(".csv"):
                content = process_csv(buffer)
            elif file_name.endswith(".pdf"):
                fd, tmp_filename = tempfile.mkstemp(suffix=".pdf")
                with os.fdopen(fd, "wb") as tmpfile:
                    tmpfile.write(buffer.read())
                    tmpfile.flush()
                    content = read_pdf_file(tmp_filename)
            else:
                raise ValueError("Unsupported file type")
        except Exception:
            error = "An error occurred while reading the file contents."
        input_file = InputFile(name=file_name, content=content, error=error)
        input_files.append(input_file)
    return input_files


def render_attached_files(message: Mapping[str, Any], element: DeltaGenerator | None = None) -> None:
    if "input_files" in message:
        file_statuses = []
        for input_file in message["input_files"]:
            if input_file.error is not None:
                file_statuses.append(f"❌ {input_file.name} - {input_file.error}")
            else:
                file_statuses.append(f"✅ {input_file.name}")
        file_names_string = "\n\n".join(file_statuses)
        if element is not None:
            element.info(f"Attached files:\n\n{file_names_string}", icon="ℹ️")
        else:
            st.info(f"Attached files:\n\n{file_names_string}", icon="ℹ️")


async def render_streamlit_ui() -> None:
    st.title("Generative AI Python SDK Assistant")

    with st.expander("Attach files", expanded=False):
        st.warning("These files will be included with each subsequent query. Make sure to clear them after you submit your message.", icon="⚠️")
        attached_files = st.file_uploader("Upload a file", type=["csv", "pdf"], accept_multiple_files=True, label_visibility="collapsed")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            render_actions(message.get("actions", []))
            render_attached_files(message)
            st.markdown(message["content"])

    attached_files_note = f" ({len(attached_files)} file{'s' if len(attached_files) != 1 else ''} attached)" if attached_files else ""
    if user_input := st.chat_input(f"How can I help you?{attached_files_note}"):
        input_files = read_attached_files(attached_files)
        message = {"role": "user", "content": user_input}
        if attached_files:
            message["input_files"] = input_files
        st.session_state.messages.append(message)
        with st.chat_message("user"):
            st.markdown(user_input)

        assistant_message = st.chat_message("assistant")
        actions_container = assistant_message.container()
        render_attached_files(message, element=assistant_message)
        stream_text = ""
        stream_container = assistant_message.empty()
        actions = []
        async for event in st.session_state.agent_executor.astream_events(build_agent_input(user_input, input_files), version="v1"):
            if event["event"] == "on_tool_end":
                if "query" in event["data"]["input"]:
                    query = event["data"]["input"]["query"]
                elif "python_code" in event["data"]["input"]:
                    query = event["data"]["input"]["python_code"]
                else:
                    query = ""
                action = {
                    "tool": event["name"],
                    "query": query,
                    "output": event["data"]["output"],
                }
                actions.append(action)
                render_actions([action], element=actions_container)
            elif event["event"] == "on_chat_model_stream":
                stream_text += event["data"]["chunk"].content
                stream_container.markdown(stream_text)
        # TODO this code does not run if the user interrupts the model
        st.session_state.messages.append({"role": "assistant", "content": stream_text, "actions": actions})
