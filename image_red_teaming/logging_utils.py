from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


_LOG_LOCK = threading.Lock()


def append_jsonl(path: Path, record: Any) -> None:
    """
    Append a JSON-serializable record to a JSONL file.
    """
    ensure_parent(path)
    payload = json.dumps(record, ensure_ascii=False)
    with _LOG_LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(payload)
            f.write("\n")


def append_text(path: Path, text: str) -> None:
    """
    Append plain text (with trailing newline) to a file.
    """
    ensure_parent(path)
    if not text.endswith("\n"):
        text = text + "\n"
    with _LOG_LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(text)
