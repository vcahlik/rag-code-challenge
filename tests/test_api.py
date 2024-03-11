from fastapi.testclient import TestClient

from api import app
from brainsoft_code_challenge.config import DEFAULT_MODEL  # noqa: E402
from brainsoft_code_challenge.tokenizer import count_tokens, get_memory_token_limit, shorten_text  # noqa: E402

client = TestClient(app)


def test_home() -> None:
    response = client.get("/")
    assert response.status_code == 200  # noqa: S101, PLR2004
    data = response.json()
    assert data["Status"] == "ok"  # noqa: S101


def test_basic_chat() -> None:
    response = client.post("/chat", json={"user_input": "Who are you?", "temperature": 0.0})
    assert response.status_code == 200  # noqa: S101, PLR2004
    data = response.json()
    assert data.keys() == {"input", "output"}  # noqa: S101

    response = client.post("/chat", json={"user_input": "Who are you?", "return_history": True})
    assert response.status_code == 200  # noqa: S101, PLR2004
    data = response.json()
    assert data.keys() == {"input", "output", "history"}  # noqa: S101
    assert data["history"] == [{"type": "human", "content": "Who are you?"}, {"type": "ai", "content": data["output"]}]  # noqa: S101

    response = client.post("/chat", json={"user_input": "Who are you?", "temperature": 10})
    assert response.status_code == 400  # noqa: S101, PLR2004


def test_chat_with_history() -> None:
    response = client.post(
        "/chat", json={"user_input": "Who are you?", "history": [{"type": "human", "content": "Hello"}, {"type": "ai", "content": "Hi! How can I help you?"}]}
    )
    assert response.status_code == 200  # noqa: S101, PLR2004
    data = response.json()
    assert data.keys() == {"input", "output"}  # noqa: S101

    response = client.post("/chat", json={"user_input": "Who are you?", "history": [{"type": "ai", "content": "Hi! How can I help you?"}]})
    assert response.status_code == 200  # noqa: S101, PLR2004
    data = response.json()
    assert data.keys() == {"input", "output"}  # noqa: S101


def test_chat_with_file() -> None:
    response = client.post("/chat", json={"user_input": "Who are you?", "files": [{"file_name": "test.csv", "content": "QSxCCjEsMgozLDQKMTAsMjAKMzAsNDAK"}]})
    assert response.status_code == 200  # noqa: S101, PLR2004
    data = response.json()
    assert data.keys() == {"input", "output"}  # noqa: S101

    response = client.post("/chat", json={"user_input": "Who are you?", "files": [{"file_name": "test.pdf", "content": "QSxCCjEsMgozLDQKMTAsMjAKMzAsNDAK"}]})
    assert response.status_code == 400  # noqa: S101, PLR2004


def test_chat_with_very_long_history() -> None:
    human_text = (
        """Isaac Asimov was an American writer and professor of biochemistry at Boston University. During his lifetime, Asimov was considered one of the "Big Three" science fiction writers, along with Robert A. Heinlein and Arthur C. Clarke. A prolific writer, he wrote or edited more than 500 books. He also wrote an estimated 90,000 letters and postcards. Best known for his hard science fiction, Asimov also wrote mysteries and fantasy, as well as popular science and other non-fiction."""  # noqa: E501
        * 200
    )  # noqa: E501
    ai_text = "Thank you for your input, what can I help you with?"
    history = [{"type": "human", "content": human_text}, {"type": "ai", "content": ai_text}]
    response = client.post("/chat", json={"user_input": "What was the conversation about?", "history": history})
    assert response.status_code == 400  # noqa: S101, PLR2004

    # Test of summarization
    memory_token_limit = get_memory_token_limit(DEFAULT_MODEL)
    ai_text_token_size = count_tokens(ai_text)
    human_text_token_limit = memory_token_limit - ai_text_token_size
    human_text, _ = shorten_text(human_text, human_text_token_limit)
    assert ai_text_token_size + count_tokens(human_text) == memory_token_limit  # noqa: S101
    history = [{"type": "human", "content": human_text}, {"type": "ai", "content": ai_text}]
    response = client.post("/chat", json={"user_input": "What was the conversation about?", "history": history, "return_history": True})
    assert response.status_code == 200  # noqa: S101, PLR2004
    data = response.json()
    assert data.keys() == {"input", "output", "history"}  # noqa: S101
    assert sum([count_tokens(message["content"]) for message in data["history"]]) <= memory_token_limit  # noqa: S101
