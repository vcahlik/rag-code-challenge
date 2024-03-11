import tiktoken

tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)
