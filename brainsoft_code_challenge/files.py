from dataclasses import dataclass

import fitz


@dataclass
class InputFile:
    name: str
    content: str
    error: str | None = None


class UnsupportedFileTypeError(ValueError):
    pass


def process_csv(file=None, content: str | None = None) -> str:  # type: ignore
    if (file is None) == (content is None):
        raise ValueError("Exactly one of file and content must be provided.")
    if file is not None:
        content = file.read()
        if isinstance(content, bytes):
            return content.decode("utf-8")
        return content  # type: ignore
    return content  # type: ignore


def read_pdf_file(filename: str) -> str:
    document = fitz.open(filename, filetype="pdf")
    try:
        text = ""
        for page in document:
            text += page.get_text()
        return text
    finally:
        document.close()
