from brainsoft_code_challenge.utils import load_environment

load_environment()

from brainsoft_code_challenge.agent import get_unique_results  # noqa: E402
from brainsoft_code_challenge.vector_store import MetadataType  # noqa: E402


def test_get_unique_results() -> None:
    results: list[MetadataType] = [
        {"source_url": "url1", "split_part": 0, "content": "content1_0", "chunk": "0"},
        {"source_url": "url1", "split_part": 0, "content": "content1_0", "chunk": "1"},
        {"source_url": "url1", "split_part": 0, "content": "content1_0", "chunk": "2"},
        {"source_url": "url1", "split_part": 1, "content": "content1_1", "chunk": "0"},
        {"source_url": "url1", "split_part": 1, "content": "content1_1", "chunk": "1"},
        {"source_url": "url1", "split_part": 1, "content": "content1_1", "chunk": "2"},
        {"source_url": "url2", "split_part": 0, "content": "content2_0", "chunk": "0"},
        {"source_url": "url2", "split_part": 0, "content": "content2_0", "chunk": "1"},
        {"source_url": "url2", "split_part": 0, "content": "content2_0", "chunk": "2"},
        {"source_url": "url2", "split_part": 1, "content": "content2_1", "chunk": "0"},
        {"source_url": "url2", "split_part": 1, "content": "content2_1", "chunk": "1"},
        {"source_url": "url2", "split_part": 1, "content": "content2_1", "chunk": "2"},
    ]
    unique_results = get_unique_results(results, 3)
    assert unique_results == [  # noqa: S101
        {"source_url": "url1", "split_part": 0, "content": "content1_0", "chunk": "0"},
        {"source_url": "url1", "split_part": 1, "content": "content1_1", "chunk": "0"},
        {"source_url": "url2", "split_part": 0, "content": "content2_0", "chunk": "0"},
    ]
