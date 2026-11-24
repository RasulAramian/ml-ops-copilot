
from pathlib import Path
from pypdf import PdfReader


def load_document(file_path):
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    extension = path.suffix.lower()

    if extension in [".md", ".txt"]:
        text = path.read_text(encoding="utf-8")
        page_count = None

    elif extension == ".pdf":
        reader = PdfReader(str(path))

        pages = []

        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""

            pages.append(
                f"\n[Page {page_number}]\n{page_text}"
            )

        text = "\n".join(pages)
        page_count = len(reader.pages)

    else:
        raise ValueError(
            f"Unsupported file format: {extension}"
        )

    if not text.strip():
        raise ValueError("Document contains no extractable text")

    return {
        "filename": path.name,
        "file_type": extension,
        "content": text,
        "page_count": page_count
    }
