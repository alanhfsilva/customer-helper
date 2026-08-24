from __future__ import annotations

import hashlib
import re


def normalize_content(raw: str) -> str:
    text = re.sub(r"<!--.*?-->", "", raw, flags=re.DOTALL)
    text = re.sub(r"<nav[\s\S]*?</nav>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<footer[\s\S]*?</footer>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def compute_content_hash(content: str) -> str:
    normalized = " ".join(content.split())
    return hashlib.sha256(normalized.encode()).hexdigest()
