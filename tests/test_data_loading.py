import json

from brainsoft_code_challenge.scraping import scrape_all
from brainsoft_code_challenge.splitting import split_long_document, split_document


def test_scraping():
    results = scrape_all()
    assert len(results) > 0


def test_splitting():
    with open("data/test/scraped_docs.json") as f:
        data = json.load(f)
    for document in data:
        split_parts = split_document(document)
        assert len(split_parts) > 0
        assert "".join(part["content"] for part in split_parts) == document["content"]


def test_split_long_document():
    def __test_split_long_document(content, min_length: int | None, verify_n_sections: int | None = None):
        results = split_long_document({"content": content}, min_length=min_length)
        if verify_n_sections is not None:
            assert len(results) == verify_n_sections
        assert "".join(result["content"] for result in results) == content

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
