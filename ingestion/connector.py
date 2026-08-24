from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from app.models import Document, SourceType

if TYPE_CHECKING:
    from collections.abc import Iterable


class SourceConnector(Protocol):
    def iter_documents(self) -> Iterable[Document]: ...


class MarkdownConnector:
    def __init__(
        self,
        corpus_dir: str | Path,
        *,
        base_url: str = "",
        source_type: SourceType = SourceType.HELP_ARTICLE,
    ) -> None:
        self._corpus_dir = Path(corpus_dir)
        self._base_url = base_url.rstrip("/")
        self._source_type = source_type

    def iter_documents(self) -> Iterable[Document]:
        if not self._corpus_dir.is_dir():
            return

        for path in sorted(self._corpus_dir.rglob("*.md")):
            content = path.read_text(encoding="utf-8")
            relative = path.relative_to(self._corpus_dir)
            slug = str(relative.with_suffix(""))
            source_uri = f"{self._base_url}/{slug}" if self._base_url else slug
            doc_id = hashlib.sha256(source_uri.encode()).hexdigest()[:16]
            title = self._extract_title(content, fallback=slug)
            content_hash = hashlib.sha256(content.encode()).hexdigest()

            yield Document(
                id=doc_id,
                source_uri=source_uri,
                title=title,
                source_type=self._source_type,
                content_raw=content,
                content_hash=content_hash,
                metadata={"file_path": str(relative)},
            )

    @staticmethod
    def _extract_title(content: str, fallback: str) -> str:
        match = re.match(r"^#\s+(.+)", content.strip())
        return match.group(1).strip() if match else fallback
