import streamlit as st
import asyncio

from brainsoft_code_challenge.agent import get_agent_executor
from brainsoft_code_challenge.constants import ACTION_HINTS


def initialize():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent_executor" not in st.session_state:
        st.session_state.agent_executor = get_agent_executor(verbose=True)


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
        st.session_state.messages.append({"role": "assistant", "content": stream_text, "actions": actions})


if __name__ == "__main__":
    initialize()
    asyncio.run(render_streamlit_ui())
