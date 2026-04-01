from typing import Dict, Optional


class CounterfactualCache:
    def __init__(self) -> None:
        self._store: Dict[str, object] = {}
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[object]:
        if key in self._store:
            self._hits += 1
            return self._store[key]
        self._misses += 1
        return None

    def put(self, key: str, value: object) -> None:
        self._store[key] = value

    def stats(self) -> Dict[str, int]:
        return {"hits": self._hits, "misses": self._misses, "size": len(self._store)}
