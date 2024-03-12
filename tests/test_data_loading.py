from brainsoft_code_challenge.utils import load_environment

load_environment()

import json  # noqa: E402

from brainsoft_code_challenge.data_loading.scraping import scrape_all  # noqa: E402
from brainsoft_code_challenge.data_loading.splitting import __split_long_document, split_document  # noqa: E402


def test_scraping() -> None:
    results = scrape_all()
    assert len(results) > 0  # noqa: S101


def test_splitting() -> None:
    with open("data/pytest/scraped_docs.json") as f:
        data = json.load(f)
    for document in data:
        split_parts = split_document(document)
        assert len(split_parts) > 0  # noqa: S101
        assert "".join(part["content"] for part in split_parts) == document["content"]  # noqa: S101


def test_split_long_document() -> None:
    def __test_split_long_document(content: str, min_length: int | None, verify_n_sections: int | None = None) -> None:
        results = __split_long_document({"content": content}, min_length=min_length)
        if verify_n_sections is not None:
            assert len(results) == verify_n_sections  # noqa: S101
        assert "".join(result["content"] for result in results) == content  # noqa: S101

    content_a = """Changelog
=========


v2.2.0
-------------------

🐛 Bug Fixes
^^^^^^^^^^^
- fix

v2.1.1
-------------------

🐛 Bug Fixes
^^^^^^^^^^^
- fix


v2.1.0 (2024-01-30)
-------------------

Change"""
    __test_split_long_document(content_a, min_length=None, verify_n_sections=4)
    __test_split_long_document(content_a, min_length=10)
    __test_split_long_document(content_a, min_length=50)
    __test_split_long_document(content_a, min_length=100)

    content_b = """v2.2.0
    -------------------

    🐛 Bug Fixes
    ^^^^^^^^^^^
    - fix

    v2.1.1
    -------------------

    🐛 Bug Fixes
    ^^^^^^^^^^^
    - fix


    v2.1.0 (2024-01-30)
    -------------------

    Change"""
    __test_split_long_document(content_b, min_length=None, verify_n_sections=3)
    __test_split_long_document(content_b, min_length=1)
    __test_split_long_document(content_b, min_length=50)
    __test_split_long_document(content_b, min_length=100)
