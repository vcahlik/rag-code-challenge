import os

from langchain_community.tools import BearlyInterpreterTool

from brainsoft_code_challenge.constants import BEARLY_CODE_INTERPRETER_DESCRIPTION


def get_code_interpreter_tool():  # type: ignore
    code_interpreter_tool = BearlyInterpreterTool(api_key=os.getenv("BEARLY_API_KEY")).as_tool()
    code_interpreter_tool.description = BEARLY_CODE_INTERPRETER_DESCRIPTION
    return code_interpreter_tool
