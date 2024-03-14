import os
import tempfile
from collections.abc import Mapping, Sequence
from enum import Enum
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


class StreamlitMessageData:
    """
    A class to manage the data for a single message in the Streamlit chat. As we want to visualize whenever the agent uses a tool,
    it keeps track of the mix of "message" (a single part of the stream) and "action" (the use of a tool) interactions, together with the attached files.
    """

    class MessageRole(Enum):
        USER = "user"
        ASSISTANT = "assistant"

    class InteractionType(Enum):
        MESSAGE = "message"
        ACTION = "action"

    def __init__(self, role: MessageRole, attached_files: Sequence[InputFile] | None = None):
        self.role = role
        self.interactions: list[Mapping[str, Any]] = []
        self.attached_files = attached_files or []
        self.current_stream = ""
        self.current_stream_container: DeltaGenerator | None = None

    def attach_files(self, files: Sequence[InputFile], render_element: DeltaGenerator | None = None) -> None:
        """
        Attaches the given files to the message, and visualizes them.

        :param files: The InputFile objects attached to the message.
        :param render_element: Streamlit element to render the attached files in (use None to render in the main element).
        """
        self.attached_files = files
        self._render_attached_files(element=render_element)

    def append_to_current_stream_and_render(self, content: str, render_element: DeltaGenerator) -> None:
        """
        Appends a string to the current stream and renders it in the given container.

        :param content: Text to be appended.
        :param render_element: Streamlit element to render the appended text in.
        """
        self.current_stream += content
        if self.current_stream_container is None:
            self.current_stream_container = render_element.empty()
        self._render_message_content(self.current_stream, element=self.current_stream_container)

    def register_and_render_message(self, content: str, render_element: DeltaGenerator | None = None) -> None:
        """
        Registers a "message" interaction and renders it in the given container.

        :param content: Message text.
        :param render_element: Streamlit element to render the message in (use None to render in the main element).
        """
        interaction = {"type": self.InteractionType.MESSAGE, "content": content}
        self.interactions.append(interaction)
        self._render_message_content(content, element=render_element)

    def register_and_render_action(self, action: Mapping[str, str], render_element: DeltaGenerator | None = None) -> None:
        """
        Registers an "action" interaction and renders it in the given container.

        :param action: Dictionary representing the action data.
        :param render_element: Streamlit element to render the action in (use None to render in the main element).
        """
        self.register_and_reset_current_stream()
        interaction = {"type": self.InteractionType.ACTION, "action": action}
        self.interactions.append(interaction)
        self._render_action(action, element=render_element)

    def register_and_reset_current_stream(self) -> None:
        """
        Registers the current stream as a "message" interaction and resets it.
        """
        self.current_stream_container = None
        if self.current_stream == "":
            return
        interaction = {"type": self.InteractionType.MESSAGE, "content": self.current_stream}
        self.interactions.append(interaction)
        self.current_stream = ""

    def _render_attached_files(self, element: DeltaGenerator | None = None) -> None:
        """
        Renders the attached files in the given container.

        :param element: Streamlit element to render the action in (use None to render in the main element).
        """
        if not self.attached_files:
            return
        file_statuses = []
        for attached_file in self.attached_files:
            if attached_file.error is not None:
                file_statuses.append(f"❌ {attached_file.name} - {attached_file.error}")
            else:
                file_statuses.append(f"✅ {attached_file.name}")
        file_names_string = "\n\n".join(file_statuses)
        if element is not None:
            element.info(f"Attached files:\n\n{file_names_string}", icon="ℹ️")
        else:
            st.info(f"Attached files:\n\n{file_names_string}", icon="ℹ️")

    @staticmethod
    def _render_message_content(content: str, element: DeltaGenerator | None = None) -> None:
        """
        Renders the content of a "message" interaction in the given container.

        :param content: The message content.
        :param element: Streamlit element to render the message in (use None to render in the main element).
        """
        if element is None:
            st.markdown(content)
        else:
            element.markdown(content)

    @staticmethod
    def _render_action(action: Mapping[str, str], element: DeltaGenerator | None = None) -> None:
        """
        Displays an "action" interaction in the given container.

        :param action: The action interaction to display.
        :param element: Streamlit element to render the message in (use None to render in the main element).
        """
        hint = ACTION_HINTS[action["tool"]]
        query = action["query"]
        expander = st.expander(f"{hint}: {query}", expanded=False) if element is None else element.expander(f"{hint}: {query}", expanded=False)
        with expander:
            st.text(action["output"])

    def render(self) -> None:
        """
        Renders the message in the Streamlit chat.
        """
        chat_message = st.chat_message(self.role.value)
        stream_container = None
        self._render_attached_files(element=chat_message)
        for interaction in self.interactions:
            if interaction["type"] == self.InteractionType.MESSAGE:
                if stream_container is None:
                    stream_container = chat_message.empty()
                self._render_message_content(interaction["content"], element=stream_container)
            elif interaction["type"] == self.InteractionType.ACTION:
                self._render_action(interaction["action"], element=chat_message)
                stream_container = None
            else:
                raise ValueError(f"Unknown interaction type: {interaction['type']}")


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


def __save_last_agent_output() -> None:
    """
    Saves the last agent output to the chat history.
    """
    if "current_message_data" in st.session_state:
        current_message_data = st.session_state.current_message_data
        current_message_data.register_and_reset_current_stream()
        st.session_state.messages.append(current_message_data)
        del st.session_state.current_message_data


def __render_conversation_history() -> None:
    """
    Renders the conversation history.
    """
    __save_last_agent_output()
    for message in st.session_state.messages:
        message.render()


async def __process_user_input(user_input: str, attached_files: Sequence[UploadedFile] | None) -> None:
    """
    Processes the user input and renders the assistant's response.

    :param user_input: The input submitted by the user.
    :param attached_files: The files attached with the input.
    """
    user_message = st.chat_message("user")
    user_message_data = StreamlitMessageData(StreamlitMessageData.MessageRole.USER)
    if input_files := __read_attached_files(attached_files):
        user_message_data.attach_files(input_files, render_element=user_message)
    user_message_data.register_and_render_message(user_input, render_element=user_message)
    st.session_state.messages.append(user_message_data)

    assistant_message = st.chat_message("assistant")
    st.session_state.current_message_data = StreamlitMessageData(StreamlitMessageData.MessageRole.ASSISTANT)

    agent_input, input_was_cut_off = build_agent_input(user_input, input_files, model=st.session_state.model_config["model"])
    if input_was_cut_off:
        st.toast("The input was too long and therefore was cut off.", icon="⚠️")
    try:
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
                st.session_state.current_message_data.register_and_render_action(action, render_element=assistant_message)
            elif event["event"] == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                st.session_state.current_message_data.append_to_current_stream_and_render(content, render_element=assistant_message)
    except Exception as e:
        st.toast(f"An error occurred while obtaining the response: {e}", icon="⚠️")
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
