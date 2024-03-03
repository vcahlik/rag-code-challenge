def merge_small_sections(contents, min_length: int | None):
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


def split_long_document(document, min_length: int | None):
    lines = document["content"].splitlines(True)
    subsection_start_indices = sorted({0} | {i - 1 for i, line in enumerate(lines) if set(line.strip()) == {'-'}})
    contents = ["".join(lines[i:j]) for i, j in zip(subsection_start_indices, subsection_start_indices[1:] + [None])]
    if min_length is not None:
        contents = merge_small_sections(contents, min_length)
    results = []
    for content in contents:
        result = document.copy()
        result["content"] = content
        results.append(result)
    return results


def split_document(document):
    if len(document["content"]) > 8000:
        assert document["type"] == "documentation" and document["source_path"] in (
            "documentation/source/v2_migration_guide.rst",
            "documentation/source/changelog.rst"
        )
        return split_long_document(document, min_length=5000)
    return [document]
