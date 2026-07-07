import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QCoreApplication, QEventLoop, QObject, QTimer, pyqtSignal

from app.application.computation import ComputationResult, compute_result
from app.application.controller import AppController
from app.application.experiment import ExperimentState, LayerSpec


@pytest.fixture(scope="module", autouse=True)
def qt_app():
    app = QCoreApplication.instance() or QCoreApplication([])
    yield app


class FakeManager(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(int, str)

    def __init__(self):
        super().__init__()
        self.requests = []

    def request_compute(self, seq, state):
        self.requests.append((seq, state))

    def shutdown(self):
        pass


def make_state(**overrides):
    defaults = dict(layers=[LayerSpec(1.0)], wavelength=0.5, fidelity="Low")
    defaults.update(overrides)
    return ExperimentState(**defaults)


def make_result(state, seq):
    n = state.n_angles
    return ComputationResult(
        state=state,
        seq=seq,
        S_th=np.zeros(n, dtype=complex),
        S_ph=np.zeros(n, dtype=complex),
        angles=np.linspace(0, 2 * np.pi, n, endpoint=False),
        k=2 * np.pi / state.wavelength,
        elapsed_seconds=0.0,
    )


@pytest.fixture()
def controller():
    manager = FakeManager()
    controller = AppController(computation_manager=manager)
    received, failures, started = [], [], []
    controller.result_ready.connect(received.append)
    controller.computation_failed.connect(failures.append)
    controller.computation_started.connect(lambda: started.append(True))
    return controller, manager, received, failures, started


def test_immediate_set_state_requests_compute(controller):
    ctrl, manager, received, _, started = controller
    state = make_state()

    ctrl.set_state(state, defer=False)

    assert manager.requests == [(1, state)]
    assert started == [True]
    assert received == []


def test_finished_result_is_cached_and_reused(controller):
    ctrl, manager, received, _, _ = controller
    state = make_state()

    ctrl.set_state(state, defer=False)
    result = make_result(state, seq=1)
    manager.finished.emit(result)
    assert received == [result]

    ctrl.compute_now()
    assert received == [result, result]
    assert len(manager.requests) == 1


def test_stale_result_is_cached_but_not_displayed(controller):
    ctrl, manager, received, _, _ = controller
    old_state = make_state(wavelength=0.4)
    new_state = make_state(wavelength=0.6)

    ctrl.set_state(old_state, defer=False)
    ctrl.set_state(new_state, defer=False)

    manager.finished.emit(make_result(old_state, seq=1))
    assert received == []

    fresh = make_result(new_state, seq=2)
    manager.finished.emit(fresh)
    assert received == [fresh]

    ctrl.set_state(old_state, defer=False)
    assert len(manager.requests) == 2
    assert received[-1].state == old_state


def test_stale_failure_is_ignored(controller):
    ctrl, manager, _, failures, _ = controller

    ctrl.set_state(make_state(wavelength=0.4), defer=False)
    ctrl.set_state(make_state(wavelength=0.6), defer=False)

    manager.failed.emit(1, "stale boom")
    assert failures == []

    manager.failed.emit(2, "boom")
    assert failures == ["boom"]


def test_deferred_set_state_debounces(qt_app, controller):
    ctrl, manager, _, _, _ = controller

    for wavelength in (0.4, 0.5, 0.6):
        ctrl.set_state(make_state(wavelength=wavelength), defer=True)
    assert manager.requests == []

    loop = QEventLoop()
    QTimer.singleShot(600, loop.quit)
    loop.exec()

    assert len(manager.requests) == 1
    assert manager.requests[0][1].wavelength == 0.6


def test_compute_result_produces_full_circle_amplitudes():
    state = make_state()
    result = compute_result(state, seq=7)

    assert result.seq == 7
    assert result.S_th.shape == (state.n_angles,)
    assert result.S_ph.shape == (state.n_angles,)
    assert np.all(np.isfinite(result.S_th))
    assert result.k == pytest.approx(2 * np.pi / state.wavelength)
