from streamlit.testing.v1 import AppTest


def test_chat() -> None:
    at = AppTest.from_file("Assistant.py", default_timeout=30).run()
    at.slider[0].set_value(0.0).run()
    at.button[0].click().run()
    at.chat_input[0].set_value("Who are you?").run()
    assert len(at.chat_message) == 2  # noqa: S101, PLR2004
    assert at.chat_message[0].name == "user"  # noqa: S101
    assert at.chat_message[0].children[0].value == "Who are you?"  # type: ignore  # noqa: S101
    assert at.chat_message[1].name == "assistant"  # noqa: S101
    assert (  # noqa: S101
        at.chat_message[1].children[1].value  # type: ignore
        == "I am a helpful assistant here to answer any general knowledge questions you may have. How can I assist you today?"
    )  # noqa: E501
