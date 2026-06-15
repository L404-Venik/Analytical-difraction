# test_observation_parameters.py
import numpy as np
import re
import pytest
from sphere_diffraction.parameters import (ObservationParameters)
import scipy.constants as const
c = const.speed_of_light

_ANGLES = np.linspace(0, 2 * np.pi, 16, endpoint=False)


# ── wavelengths normalization ─────────────────────────────────────────────────

def test_scalar_wavelength_becomes_array():
    o = ObservationParameters(wavelengths=0.5, angles=_ANGLES)
    assert isinstance(o.wavelengths, np.ndarray)
    assert o.wavelengths.shape == (1,)
    assert o.wavelengths[0] == pytest.approx(0.5)

def test_array_wavelength_accepted():
    wl = np.array([0.3, 0.5, 0.7])
    o = ObservationParameters(wavelengths=wl, angles=_ANGLES)
    assert o.wavelengths.shape == (3,)

# ── derived properties ────────────────────────────────────────────────────────

def test_k_is_array_of_correct_length():
    wl = np.array([0.3, 0.5, 0.7])
    o = ObservationParameters(wavelengths=wl, angles=_ANGLES)
    assert o.k.shape == (3,)
    np.testing.assert_allclose(o.k, 2 * np.pi / wl)

def test_k_single_wavelength():
    o = ObservationParameters(wavelengths=0.5, angles=_ANGLES)
    assert o.k.shape == (1,)
    assert o.k[0] == pytest.approx(2 * np.pi / 0.5)

def test_frequency_consistency():
    o = ObservationParameters(wavelengths=0.03, angles=_ANGLES)
    assert o.frequency_hz[0]  == pytest.approx(c / 0.03)
    assert o.frequency_ghz[0] == pytest.approx(o.frequency_hz[0] / 1e9)

# ── validation: wavelengths ───────────────────────────────────────────────────

def test_zero_wavelength_raises():
    with pytest.raises(ValueError, match=re.escape("All wavelengths must be > 0")):
        ObservationParameters(wavelengths=0.0, angles=_ANGLES)

def test_negative_wavelength_raises():
    with pytest.raises(ValueError, match=re.escape("All wavelengths must be > 0")):
        ObservationParameters(wavelengths=-1.0, angles=_ANGLES)

def test_negative_in_wavelength_array_raises():
    with pytest.raises(ValueError, match=re.escape("All wavelengths must be > 0")):
        ObservationParameters(wavelengths=np.array([0.3, -0.5]), angles=_ANGLES)

# ── validation: angles ────────────────────────────────────────────────────────

def test_empty_angles_raises():
    with pytest.raises(ValueError, match="angles must be a non-empty 1-D array"):
        ObservationParameters(wavelengths=0.5, angles=[])

def test_2d_angles_raises():
    with pytest.raises(ValueError, match="angles must be a non-empty 1-D array"):
        ObservationParameters(wavelengths=0.5, angles=[[0.0, 1.0]])

def test_angles_converted_to_array():
    o = ObservationParameters(wavelengths=0.5, angles=[0.0, 1.0, 2.0])
    assert isinstance(o.angles, np.ndarray)
    assert o.angles.dtype == np.float64
