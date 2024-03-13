ACTION_HINTS = {
    "search_documentation": "Query to documentation",
    "search_google": "Query to Google Search",
    "bearly_interpreter": "Request to code interpreter",
}

CONTEXT_WINDOW_SIZE_IN_TOKENS_BY_MODEL = {
    "gpt-3.5-turbo": 16385,
    "gpt-4": 8192,
    "gpt-4-turbo-preview": 128000,
}

TOOLS_AND_SYSTEM_PROMPT_LENGTH_TOKENS = 1000  # An upper bound estimate
OUTPUT_TOKEN_LIMIT = 4096

PYTEST_USER_INPUT_ENV_VAR = "PYTEST_USER_INPUT"

BEARLY_CODE_INTERPRETER_DESCRIPTION = """Evaluates Python code in a sandboxed environment. The environment resets on every execution. You must send the whole script every time and print your outputs. The script must be pure Python code that can be evaluated. It must be in Python format, NOT markdown. The code must NOT be wrapped in backticks. All common Python packages including requests, matplotlib, scipy, numpy, pandas, etc. are available, but the IBM Generative AI Python SDK Assistant is not available and can't be installed! Do not use features like plot.show() as you won't be able to see the output! Use print() to print any results so you can capture the output. If you get empty stdout in the response, add print() statements to your code and try again!"""  # noqa: E501
