from __future__ import annotations

from .experiment import ExperimentState


class ResultCache:
    """FIFO-bounded cache of computation results keyed by the full experiment state."""

    def __init__(self, max_entries: int = 256) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be a positive integer")
        self._cache: dict[tuple, object] = {}
        self._max_entries = max_entries

    def get(self, state: ExperimentState):
        return self._cache.get(state.cache_key())

    def put(self, state: ExperimentState, result) -> None:
        key = state.cache_key()
        if key not in self._cache and len(self._cache) >= self._max_entries:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = result

    def clear(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)
