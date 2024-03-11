from collections.abc import Mapping, Sequence
from typing import Any

from brainsoft_code_challenge.config import MIN_SPLIT_LENGTH_CHARS, SPLIT_DOCUMENTS_LONGER_THAN_N_CHARS
from brainsoft_code_challenge.utils import is_pytest_running


def __merge_small_splits(splits: Sequence[str], min_length: int) -> list[str]:
    """
    Merge small splits into larger ones.

    :param splits: The contents of all splits of the document.
    :param min_length: The desired minimum length of each split (in characters).
    :return: The merged splits.
    """
    results = []
    merged_split = []
    for split in splits:
        merged_split.append(split)
        if len("".join(merged_split)) >= min_length:
            results.append(merged_split)
            merged_split = []
    if merged_split:
        if len("".join(merged_split)) < min_length and results:
            results[-1].extend(merged_split)
        else:
            results.append(merged_split)
    return ["".join(merged_section) for merged_section in results]


def __split_long_document(document: Mapping[str, Any], min_length: int | None) -> list[dict[str, Any]]:
    """
    Split a long document into smaller splits.

    :param document: The document to split.
    :param min_length: The desired minimum length of each section in characters.
    :return: The split documents.
    """
    lines = document["content"].splitlines(True)
    subsection_start_indices = sorted({0} | {i - 1 for i, line in enumerate(lines) if set(line.strip()) == {"-"}})
    contents = ["".join(lines[i:j]) for i, j in zip(subsection_start_indices, subsection_start_indices[1:] + [None], strict=False)]
    if min_length is not None:
        contents = __merge_small_splits(contents, min_length)
    splits = []
    for i, content in enumerate(contents):
        split = dict(document).copy()
        split["split_part"] = i
        split["content"] = content
        splits.append(split)
    return splits


def split_document(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    """
    Split a document into smaller "splits" if it is too long.

    :param document: The document to split.
    :return: The split documents.
    """
    document = dict(document)
    if len(document["content"]) > SPLIT_DOCUMENTS_LONGER_THAN_N_CHARS:
        if is_pytest_running():
            assert document["type"] == "documentation" and document["source_path"] in (  # noqa: S101
                "documentation/source/v2_migration_guide.rst",
                "documentation/source/changelog.rst",
            )
        return __split_long_document(document, min_length=MIN_SPLIT_LENGTH_CHARS)
    return [document]
