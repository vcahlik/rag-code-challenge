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
from brainsoft_code_challenge.files import InputFile, UnsupportedFileTypeError, process_csv, read_pdf_file


def __initialize_chat() -> None:
    """
    Initializes the chat session state.
    """
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


def __reset_chat(model: str, temperature: float, frequency_penalty: float, presence_penalty: float, top_p: float) -> None:
    """
    Resets the chat session state with the given parameters. This clears the chat history and sets the model configuration.
    """
    st.session_state.messages = []
    st.session_state.model_config = {
        "model": model,
        "temperature": temperature,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "top_p": top_p,
    }
    st.session_state.agent_executor = get_agent_executor(model, temperature, frequency_penalty, presence_penalty, top_p, verbose=True)
    if "current_response" in st.session_state:
        del st.session_state.current_response


def __prepare_page() -> None:
    """
    Prepares the Streamlit page by initializing the chat and displaying the sidebar.
    """
    st.set_page_config(layout="wide", page_title="Generative AI Python SDK Assistant")
    __initialize_chat()

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
            col1.button("Apply & reset chat", type="primary", on_click=__reset_chat, args=reset_chat_args)
        else:
            col1.button("Reset chat", on_click=__reset_chat, args=reset_chat_args)


def __render_actions(actions: Sequence[Mapping[str, str]], element: DeltaGenerator | None = None) -> None:
    """
    Displays the agent actions in the given container.

    :param actions: The actions to display.
    :param element: The container to display the actions in.
    """
    for action in actions:
        hint = ACTION_HINTS[action["tool"]]
        query = action["query"]
        expander = st.expander(f"{hint}: {query}", expanded=False) if element is None else element.expander(f"{hint}: {query}", expanded=False)
        with expander:
            st.text(action["output"])


def __read_attached_files(buffers: Sequence[UploadedFile] | None) -> list[InputFile]:
    """
    Reads the contents of the attached files.

    :param buffers: The uploaded file buffers.
    :return: The InputFile objects.
    """
    if buffers is None:
        return []
    input_files = []
    for buffer in buffers:
        content = ""
        error = None
        file_name = buffer.name
        try:
            buffer.seek(0)
            if file_name.endswith(".csv"):
                content = process_csv(buffer)
            elif file_name.endswith(".pdf"):
                fd, tmp_filename = tempfile.mkstemp(suffix=".pdf")
                with os.fdopen(fd, "wb") as tmpfile:
                    tmpfile.write(buffer.read())
                    tmpfile.flush()
                    content = read_pdf_file(tmp_filename)
            else:
                raise UnsupportedFileTypeError()
        except UnsupportedFileTypeError:
            error = "Unsupported file type"
        except Exception:
            error = "An error occurred while reading the file contents."
        input_file = InputFile(name=file_name, content=content, error=error)
        input_files.append(input_file)
    return input_files


def __render_attached_files(message: Mapping[str, Any], element: DeltaGenerator | None = None) -> None:
    """
    Renders the attached files in the given container.

    :param message: The message containing the attached files.
    :param element: The container to display the attached files in.
    """
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


def __save_last_agent_output() -> None:
    """
    Saves the last agent output to the chat history.
    """
    if "current_response" in st.session_state:
        st.session_state.messages.append(
            {"role": "assistant", "content": st.session_state.current_response["stream_text"], "actions": st.session_state.current_response["actions"]}
        )
        del st.session_state.current_response


def __render_conversation_history() -> None:
    """
    Renders the conversation history.
    """
    __save_last_agent_output()
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            __render_actions(message.get("actions", []))
            __render_attached_files(message)
            st.markdown(message["content"])


async def __process_user_input(user_input: str, attached_files: Sequence[UploadedFile] | None) -> None:
    """
    Processes the user input and renders the assistant's response.

    :param user_input: The input submitted by the user.
    :param attached_files: The files attached with the input.
    """
    input_files = __read_attached_files(attached_files)
    message: dict[str, Any] = {"role": "user", "content": user_input}
    if attached_files:
        message["input_files"] = input_files
    st.session_state.messages.append(message)
    with st.chat_message("user"):
        st.markdown(user_input)

    assistant_message = st.chat_message("assistant")
    actions_container = assistant_message.container()
    __render_attached_files(message, element=assistant_message)
    stream_container = assistant_message.empty()
    st.session_state.current_response = {"stream_text": "", "actions": []}
    agent_input, input_was_cut_off = build_agent_input(user_input, input_files, model=st.session_state.model_config["model"])
    if input_was_cut_off:
        st.toast("The input was too long and therefore was cut off.", icon="⚠️")
    async for event in st.session_state.agent_executor.astream_events(agent_input, version="v1"):
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
            st.session_state.current_response["actions"].append(action)
            __render_actions([action], element=actions_container)
        elif event["event"] == "on_chat_model_stream":
            st.session_state.current_response["stream_text"] += event["data"]["chunk"].content
            stream_container.markdown(st.session_state.current_response["stream_text"])
    __save_last_agent_output()


async def render_streamlit_ui() -> None:
    """
    Renders the whole Streamlit app.
    """
    st.title("Generative AI Python SDK Assistant")

    with st.expander("Attach files", expanded=False):
        st.warning("These files will be included with each subsequent query. Make sure to clear them after you submit your message.", icon="⚠️")
        attached_files = st.file_uploader("Upload a file", type=["csv", "pdf"], accept_multiple_files=True, label_visibility="collapsed")

    __render_conversation_history()

    attached_files_note = f" ({len(attached_files)} file{'s' if len(attached_files) != 1 else ''} attached)" if attached_files else ""
    if user_input := st.chat_input(f"How can I help you?{attached_files_note}"):
        await __process_user_input(user_input, attached_files)
