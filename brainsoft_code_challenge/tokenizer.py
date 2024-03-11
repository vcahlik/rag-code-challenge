import tiktoken

from brainsoft_code_challenge.constants import CONTEXT_WINDOW_SIZE_IN_TOKENS_BY_MODEL, OUTPUT_TOKEN_LIMIT, TOOLS_AND_SYSTEM_PROMPT_LENGTH_TOKENS

tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)


def __get_universal_token_limit(model: str) -> int:
    return (CONTEXT_WINDOW_SIZE_IN_TOKENS_BY_MODEL[model] - TOOLS_AND_SYSTEM_PROMPT_LENGTH_TOKENS - OUTPUT_TOKEN_LIMIT) // 2


def get_input_token_limit(model: str) -> int:
    return __get_universal_token_limit(model)


def get_memory_token_limit(model: str) -> int:
    return __get_universal_token_limit(model)


def shorten_input_text_for_model(text: str, model: str) -> tuple[str, bool]:
    token_limit = get_input_token_limit(model)
    tokens = tokenizer.encode(text, disallowed_special=())
    if len(tokens) <= token_limit:
        return text, False
    text = tokenizer.decode(tokens[:token_limit])
    return text, True
