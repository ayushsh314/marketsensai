"""Append-only JSONL cache for per-item LLM results.

Entries are keyed by a namespace (model + prompt version) and an item key, so changing either
invalidates old results automatically. Writing after every batch means an interrupted run resumes
where it stopped, and a rerun on a refreshed corpus only processes new articles.
"""

import json
import os
from typing import Dict, Iterable, Optional


class JsonlCache:
    def __init__(self, path: Optional[str], namespace: str):
        self.path = path
        self.namespace = namespace
        self._data: Dict[str, dict] = {}
        if path and os.path.exists(path):
            with open(path) as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # a torn last line from an interrupted write
                    if entry.get("ns") == namespace:
                        self._data[entry["key"]] = entry["value"]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def get(self, key: str) -> Optional[dict]:
        return self._data.get(key)

    def put_many(self, items: Iterable) -> None:
        items = list(items)
        for key, value in items:
            self._data[key] = value
        if self.path and items:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "a") as f:
                for key, value in items:
                    f.write(json.dumps({"ns": self.namespace, "key": key, "value": value}, default=str) + "\n")
