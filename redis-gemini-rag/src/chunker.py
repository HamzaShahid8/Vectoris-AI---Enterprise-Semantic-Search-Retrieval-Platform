from __future__ import annotations

from typing import List


def _normalize_spaces(text: str) -> str:
    return " ".join(text.split()).strip()


def _find_chunk_end(text: str, start: int, max_end: int) -> int:
    if max_end >= len(text):
        return len(text)

    split_at = text.rfind(" ", start, max_end)
    if split_at == -1 or split_at <= start:
        return max_end
    return split_at


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> List[str]:
    cleaned = _normalize_spaces(text)
    if not cleaned:
        return []

    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 4)

    chunks: List[str] = []
    start = 0
    while start < len(cleaned):
        max_end = min(start + chunk_size, len(cleaned))
        end = _find_chunk_end(cleaned, start, max_end)
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(cleaned):
            break

        start = max(end - overlap, start + 1)

    return chunks
