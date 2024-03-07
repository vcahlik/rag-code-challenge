from brainsoft_code_challenge.utils import load_environment

load_environment()

import asyncio  # noqa: E402

from brainsoft_code_challenge.streamlit_interface import prepare_page, render_streamlit_ui  # noqa: E402

if __name__ == "__main__":
    prepare_page()
    asyncio.run(render_streamlit_ui())
