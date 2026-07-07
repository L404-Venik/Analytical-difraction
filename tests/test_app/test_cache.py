import pytest

from app.application.cache import ResultCache
from app.application.experiment import ExperimentState, LayerSpec


def make_state(**overrides):
    defaults = dict(layers=[LayerSpec(1.0)], wavelength=0.5, fidelity="Low")
    defaults.update(overrides)
    return ExperimentState(**defaults)


def test_get_hits_on_equal_state_and_misses_on_different_fidelity():
    cache = ResultCache()
    state = make_state()
    assert cache.get(state) is None

    result = object()
    cache.put(state, result)

    assert cache.get(make_state()) is result
    assert cache.get(make_state(fidelity="High")) is None


def test_eviction_drops_oldest_entry():
    cache = ResultCache(max_entries=2)
    first = make_state(wavelength=0.1)
    second = make_state(wavelength=0.2)
    third = make_state(wavelength=0.3)

    cache.put(first, "a")
    cache.put(second, "b")
    cache.put(third, "c")

    assert len(cache) == 2
    assert cache.get(first) is None
    assert cache.get(second) == "b"
    assert cache.get(third) == "c"


def test_put_replaces_existing_entry_without_eviction():
    cache = ResultCache(max_entries=2)
    state = make_state()
    other = make_state(wavelength=0.9)

    cache.put(state, "old")
    cache.put(other, "kept")
    cache.put(state, "new")

    assert cache.get(state) == "new"
    assert cache.get(other) == "kept"


def test_clear_and_invalid_size():
    cache = ResultCache()
    cache.put(make_state(), "x")
    cache.clear()
    assert len(cache) == 0

    with pytest.raises(ValueError):
        ResultCache(max_entries=0)
