from dataclasses import dataclass

import fitz


@dataclass
class InputFile:
    name: str
    content: str
    error: str | None = None


def read_csv_file(file) -> str:  # type: ignore
    return file.read()  # type: ignore


def read_pdf_file(filename: str) -> str:
    document = fitz.open(filename, filetype="pdf")
    try:
        text = ""
        for page in document:
            text += page.get_text()
        return text
    finally:
        document.close()
