import streamlit as st
import asyncio

from brainsoft_code_challenge.agent import get_agent_executor


def initialize():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent_executor" not in st.session_state:
        st.session_state.agent_executor = get_agent_executor(verbose=True)


async def render_streamlit_ui():
    st.title("Generative AI Python SDK Assistant")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("How can I help you?"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        chat_message = st.chat_message("assistant")
        chat_stream_container = chat_message.empty()
        output = ""
        async for event in st.session_state.agent_executor.astream_events({"input": user_input}, version="v1"):
            if event["event"] == "on_chat_model_stream":
                partial_output = event["data"]["chunk"].content
                output += partial_output
                chat_stream_container.markdown(output)
        st.session_state.messages.append({"role": "assistant", "content": output})


if __name__ == "__main__":
    initialize()
    asyncio.run(render_streamlit_ui())
