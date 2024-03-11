from fastapi.testclient import TestClient

from api import app

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

    response = client.post("/chat", json={"user_input": "Who are you?", "temperature": 10})
    assert response.status_code == 400  # noqa: S101, PLR2004


def test_chat_with_history() -> None:
    response = client.post(
        "/chat", json={"user_input": "Who are you?", "history": [{"type": "human", "content": "Hello"}, {"type": "ai", "content": "Hi! How can I help you?"}]}
    )
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
