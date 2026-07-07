import numpy as np
import pytest

from app.application.experiment import (
    FIDELITY_ANGLES,
    ExperimentState,
    LayerSpec,
    load_preset,
    save_preset,
)


def make_state(**overrides):
    defaults = dict(
        layers=[LayerSpec(1.0), LayerSpec(0.5, 2.5, 0.1)],
        conducting_core=True,
        outer_eps_real=1.0,
        outer_eps_imag=0.0,
        wavelength=0.55,
        fidelity="Low",
    )
    defaults.update(overrides)
    return ExperimentState(**defaults)


def test_to_body_accumulates_radii_and_appends_outer_eps():
    state = make_state(outer_eps_real=1.2, outer_eps_imag=0.3)
    body = state.to_body()

    assert np.allclose(body.r, [1.0, 1.5])
    assert np.allclose(body.eps, [1.0 + 0j, 2.5 + 0.1j, 1.2 + 0.3j])
    assert body.conducting_core is True


def test_to_observation_uses_fidelity_grid_without_duplicate_endpoint():
    state = make_state(fidelity="Low")
    obs = state.to_observation()

    assert len(obs.angles) == FIDELITY_ANGLES["Low"]
    assert obs.angles[0] == 0.0
    assert obs.angles[-1] < 2.0 * np.pi
    assert np.allclose(obs.wavelengths, [0.55])


def test_validation_rejects_bad_input():
    with pytest.raises(ValueError):
        ExperimentState(layers=[])
    with pytest.raises(ValueError):
        make_state(wavelength=0.0)
    with pytest.raises(ValueError):
        make_state(fidelity="Ultra")


def test_cache_key_distinguishes_every_field():
    base = make_state()
    assert base.cache_key() == make_state().cache_key()

    variants = [
        make_state(layers=[LayerSpec(1.0), LayerSpec(0.6, 2.5, 0.1)]),
        make_state(conducting_core=False),
        make_state(outer_eps_real=1.5),
        make_state(wavelength=0.6),
        make_state(fidelity="High"),
    ]
    for variant in variants:
        assert variant.cache_key() != base.cache_key()


def test_dict_roundtrip_preserves_state():
    state = make_state(conducting_core=False, wavelength=0.3, fidelity="High")
    assert ExperimentState.from_dict(state.to_dict()) == state


def test_preset_roundtrip_through_file(tmp_path):
    state = make_state()
    path = tmp_path / "preset.json"
    save_preset(path, state)
    assert load_preset(path) == state
