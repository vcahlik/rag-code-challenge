import tiktoken

from brainsoft_code_challenge.constants import CONTEXT_WINDOW_SIZE_IN_TOKENS_BY_MODEL, OUTPUT_TOKEN_LIMIT, TOOLS_AND_SYSTEM_PROMPT_LENGTH_TOKENS

tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """
    Counts the number of tokens in the given text.
    """
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)


def __get_universal_token_limit(model: str) -> int:
    """
    Get the token limit for input or memory for a given model.
    """
    return (CONTEXT_WINDOW_SIZE_IN_TOKENS_BY_MODEL[model] - TOOLS_AND_SYSTEM_PROMPT_LENGTH_TOKENS - OUTPUT_TOKEN_LIMIT) // 2


def get_input_token_limit(model: str) -> int:
    """
    Get the token limit for input for a given model. Currently, the input token limit is the same as the memory token limit.
    """
    return __get_universal_token_limit(model)


def get_memory_token_limit(model: str) -> int:
    """
    Get the token limit for memory for a given model. Currently, the memory token limit is the same as the input token limit.
    """
    return __get_universal_token_limit(model)


def shorten_input_text_for_model(text: str, model: str) -> tuple[str, bool]:
    """
    Shorten the input text to fit the token limit for a given model.

    :param text: The input text.
    :param model: The model to use.
    :return: The shortened text and a boolean indicating whether the text was shortened.
    """
    token_limit = get_input_token_limit(model)
    tokens = tokenizer.encode(text, disallowed_special=())
    if len(tokens) <= token_limit:
        return text, False
    text = tokenizer.decode(tokens[:token_limit])
    return text, True
