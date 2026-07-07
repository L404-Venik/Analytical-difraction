from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot

from analytical_diffraction import calculate_S

from .experiment import ExperimentState


@dataclass
class ComputationResult:
    """Scattering amplitudes for one experiment state.

    S_th / S_ph are 1-D arrays over `angles` (radians, full circle);
    `k` is the wavenumber for the state's single wavelength.
    """

    state: ExperimentState
    seq: int
    S_th: np.ndarray
    S_ph: np.ndarray
    angles: np.ndarray
    k: float
    elapsed_seconds: float


def compute_result(state: ExperimentState, seq: int) -> ComputationResult:
    body = state.to_body()
    observation = state.to_observation()
    start = time.perf_counter()
    S_th, S_ph = calculate_S(body, observation)
    return ComputationResult(
        state=state,
        seq=seq,
        S_th=S_th[0],
        S_ph=S_ph[0],
        angles=observation.angles,
        k=float(observation.k[0]),
        elapsed_seconds=time.perf_counter() - start,
    )


class Worker(QObject):
    """Runs computations on its own thread; only the most recent request survives."""

    finished = pyqtSignal(object)
    failed = pyqtSignal(int, str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._pending: tuple[int, ExperimentState] | None = None
        self._busy = False

    @pyqtSlot(int, object)
    def enqueue(self, seq: int, state: ExperimentState) -> None:
        self._pending = (seq, state)
        if self._busy:
            return
        self._process()

    def _process(self) -> None:
        self._busy = True
        try:
            while self._pending is not None:
                seq, state = self._pending
                self._pending = None
                try:
                    result = compute_result(state, seq)
                except Exception as exc:
                    self.failed.emit(seq, str(exc))
                    continue
                self.finished.emit(result)
        finally:
            self._busy = False


class ComputationManager(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(int, str)
    _compute_requested = pyqtSignal(int, object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread = QThread()
        self._worker = Worker()
        self._worker.moveToThread(self._thread)

        self._compute_requested.connect(self._worker.enqueue)
        self._worker.finished.connect(self.finished)
        self._worker.failed.connect(self.failed)

        self._thread.start()

    def request_compute(self, seq: int, state: ExperimentState) -> None:
        self._compute_requested.emit(seq, state)

    def shutdown(self) -> None:
        if self._thread.isRunning():
            self._thread.quit()
            self._thread.wait()
