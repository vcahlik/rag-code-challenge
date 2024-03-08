from collections.abc import Mapping, Sequence

from brainsoft_code_challenge.config import MIN_SPLIT_LENGTH_CHARS, SPLIT_DOCUMENTS_LONGER_THAN_N_CHARS
from brainsoft_code_challenge.utils import is_pytest_running


def merge_small_sections(contents: Sequence[str], min_length: int) -> list[str]:
    results = []
    merged_section = []
    for content in contents:
        merged_section.append(content)
        if len("".join(merged_section)) >= min_length:
            results.append(merged_section)
            merged_section = []
    if merged_section:
        if len("".join(merged_section)) < min_length and results:
            results[-1].extend(merged_section)
        else:
            results.append(merged_section)
    return ["".join(merged_section) for merged_section in results]


def split_long_document(document: Mapping[str, str], min_length: int | None) -> list[dict[str, str]]:
    lines = document["content"].splitlines(True)
    subsection_start_indices = sorted({0} | {i - 1 for i, line in enumerate(lines) if set(line.strip()) == {"-"}})
    contents = ["".join(lines[i:j]) for i, j in zip(subsection_start_indices, subsection_start_indices[1:] + [None], strict=False)]
    if min_length is not None:
        contents = merge_small_sections(contents, min_length)
    results = []
    for content in contents:
        result = dict(document).copy()
        result["content"] = content
        results.append(result)
    return results


def split_document(document: Mapping[str, str]) -> list[dict[str, str]]:
    document = dict(document)
    if len(document["content"]) > SPLIT_DOCUMENTS_LONGER_THAN_N_CHARS:
        if is_pytest_running():
            assert document["type"] == "documentation" and document["source_path"] in (  # noqa: S101
                "documentation/source/v2_migration_guide.rst",
                "documentation/source/changelog.rst",
            )
        return split_long_document(document, min_length=MIN_SPLIT_LENGTH_CHARS)
    return [document]
