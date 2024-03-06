import asyncio

from brainsoft_code_challenge.streamlit_interface import prepare_page, render_streamlit_ui

if __name__ == "__main__":
    prepare_page()
    asyncio.run(render_streamlit_ui())
