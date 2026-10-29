from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def _clean_text(text: str) -> str:
    return " ".join(text.split()).strip()


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf_file(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def load_documents(data_dir: Path) -> List[Dict[str, str]]:
    documents: List[Dict[str, str]] = []

    for file_path in sorted(data_dir.rglob("*")):
        if not file_path.is_file():
            continue

        extension = file_path.suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            continue

        if extension == ".pdf":
            raw_text = _read_pdf_file(file_path)
        else:
            raw_text = _read_text_file(file_path)

        text = _clean_text(raw_text)
        if not text:
            continue

        documents.append(
            {
                "source": file_path.name,
                "path": str(file_path).replace("\\", "/"),
                "text": text,
            }
        )

    return documents
