from __future__ import annotations

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from .cache import ResultCache
from .computation import ComputationManager, ComputationResult
from .experiment import ExperimentState

DEBOUNCE_MS = 300


class AppController(QObject):
    """Owns the current experiment state and turns state changes into results.

    `set_state` with `defer=True` debounces rapid edits; only the newest
    request (tracked by a sequence number) may reach `result_ready`.
    """

    computation_started = pyqtSignal()
    result_ready = pyqtSignal(object)
    computation_failed = pyqtSignal(str)

    def __init__(
        self,
        computation_manager: ComputationManager | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._cache = ResultCache()
        self._manager = computation_manager or ComputationManager()
        self._manager.finished.connect(self._on_finished)
        self._manager.failed.connect(self._on_failed)

        self._state = ExperimentState()
        self._seq = 0

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(DEBOUNCE_MS)
        self._debounce.timeout.connect(self.compute_now)

    @property
    def state(self) -> ExperimentState:
        return self._state

    def set_state(self, state: ExperimentState, defer: bool = True) -> None:
        self._state = state
        if defer:
            self._debounce.start()
        else:
            self.compute_now()

    def compute_now(self) -> None:
        self._debounce.stop()
        self._seq += 1
        cached = self._cache.get(self._state)
        if cached is not None:
            self.result_ready.emit(cached)
            return
        self.computation_started.emit()
        self._manager.request_compute(self._seq, self._state)

    def shutdown(self) -> None:
        self._manager.shutdown()

    def _on_finished(self, result: ComputationResult) -> None:
        self._cache.put(result.state, result)
        if result.seq == self._seq:
            self.result_ready.emit(result)

    def _on_failed(self, seq: int, message: str) -> None:
        if seq == self._seq:
            self.computation_failed.emit(message)
