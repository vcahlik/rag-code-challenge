import streamlit as st

from brainsoft_code_challenge.agent import get_agent_executor
from brainsoft_code_challenge.config import DEFAULT_MODEL, DEFAULT_TEMPERATURE, DEFAULT_FREQUENCY_PENALTY, DEFAULT_PRESENCE_PENALTY, DEFAULT_TOP_P
from brainsoft_code_challenge.constants import ACTION_HINTS


def initialize_chat():
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


def reset_chat(model: str, temperature: float, frequency_penalty: float, presence_penalty: float, top_p: float):
    st.session_state.messages = []
    st.session_state.model_config = {
        "model": model,
        "temperature": temperature,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "top_p": top_p,
    }
    st.session_state.agent_executor = get_agent_executor(model, temperature, frequency_penalty, presence_penalty, top_p, verbose=True)


def prepare_page():
    st.set_page_config(layout="wide", page_title=f"Generative AI Python SDK Assistant")
    initialize_chat()

    with st.sidebar:
        model_config = st.session_state.model_config
        st.title("Configuration")
        model_options = ("gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview")
        model = st.selectbox("Model", model_options, index=model_options.index(model_config["model"]))
        temperature = st.slider("Temperature", 0.0, 2.0, model_config["temperature"])
        frequency_penalty = st.slider("Frequency penalty", -2.0, 2.0, model_config["frequency_penalty"])
        presence_penalty = st.slider("Presence penalty", -2.0, 2.0, model_config["presence_penalty"])
        top_p = st.slider("Top-p", 0.0, 1.0, model_config["top_p"])
        col1, col2 = st.columns(2)
        reset_chat_args = [model, temperature, frequency_penalty, presence_penalty, top_p]
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


def render_actions(actions, element=None):
    if element is None:
        element = st
    for action in actions:
        hint = ACTION_HINTS[action["tool"]]
        query = action["query"]
        with element.expander(f"{hint}: {query}", expanded=False):
            st.text(action["output"])


async def render_streamlit_ui():
    st.title("Generative AI Python SDK Assistant")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            render_actions(message.get("actions", []))
            st.markdown(message["content"])

    if user_input := st.chat_input("How can I help you?"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        assistant_message = st.chat_message("assistant")
        actions_container = assistant_message.container()
        stream_text = ""
        stream_container = assistant_message.empty()
        actions = []
        async for event in st.session_state.agent_executor.astream_events({"input": user_input}, version="v1"):
            if event["event"] == "on_tool_end":
                action = {
                    "tool": event["name"],
                    "query": event["data"]["input"]["query"],
                    "output": event["data"]["output"],
                }
                actions.append(action)
                render_actions([action], element=actions_container)
            elif event["event"] == "on_chat_model_stream":
                stream_text += event["data"]["chunk"].content
                stream_container.markdown(stream_text)
        # TODO this code does not run if the user interrupts the model
        st.session_state.messages.append({"role": "assistant", "content": stream_text, "actions": actions})
