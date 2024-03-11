from brainsoft_code_challenge.config import DEFAULT_MODEL
from brainsoft_code_challenge.tokenizer import count_tokens, get_input_token_limit, shorten_input_text_for_model


def test_shorten_text_for_model() -> None:
    input_text = "   \nthis is some long input text  \n"
    shortened_input_text, was_shortened = shorten_input_text_for_model(input_text, DEFAULT_MODEL)
    assert was_shortened is False  # noqa: S101
    assert input_text == shortened_input_text  # noqa: S101

    token_limit = get_input_token_limit(DEFAULT_MODEL)
    input_text = input_text * 10000
    assert count_tokens(input_text) > token_limit  # noqa: S101
    shortened_input_text, was_shortened = shorten_input_text_for_model(input_text, DEFAULT_MODEL)
    assert was_shortened is True  # noqa: S101
    assert count_tokens(shortened_input_text) == token_limit  # noqa: S101
