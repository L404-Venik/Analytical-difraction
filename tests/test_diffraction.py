# test_diffraction.py
import numpy as np
import pytest
from analytical_diffraction.parameters import (BodyParameters, ObservationParameters)
from analytical_diffraction.sphere import (calculate_S)

WL = 1.0

@pytest.fixture
def simple_body():
    return BodyParameters(eps=[2.0 + 0.5j, 1.0], r=[0.2])


def evenly_spaced(M):
    """Legacy-style evenly spaced angle grid over [0, 2π)."""
    return np.arange(M) * (2 * np.pi / M)


# ── Output shape ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("M", [360, 361, 3, 4, 1])
def test_output_shape_single_wavelength(simple_body, M):
    obs = ObservationParameters(wavelengths=WL, angles=evenly_spaced(M))
    S_th, S_ph = calculate_S(simple_body, obs)
    assert S_th.shape == (1, M)
    assert S_ph.shape == (1, M)

def test_output_shape_multi_wavelength(simple_body):
    angles = evenly_spaced(180)
    obs = ObservationParameters(wavelengths=np.array([0.5, 1.0, 1.5]), angles=angles)
    S_th, S_ph = calculate_S(simple_body, obs)
    assert S_th.shape == (3, 180)
    assert S_ph.shape == (3, 180)

def test_each_wavelength_row_matches_single_call(simple_body):
    """Row i of a multi-wavelength call equals an independent single-wavelength call."""
    angles = evenly_spaced(90)
    wls = [0.5, 1.0, 1.5]
    S_th_multi, S_ph_multi = calculate_S(
        simple_body, ObservationParameters(wavelengths=np.array(wls), angles=angles))
    for i, wl in enumerate(wls):
        S_th_one, S_ph_one = calculate_S(
            simple_body, ObservationParameters(wavelengths=wl, angles=angles))
        np.testing.assert_allclose(S_th_multi[i], S_th_one[0], rtol=1e-12)
        np.testing.assert_allclose(S_ph_multi[i], S_ph_one[0], rtol=1e-12)

# ── Forward scattering (theta = 0) ────────────────────────────────────────────

def test_forward_scattering_sph_is_negated(simple_body):
    """S_ph = -S_th at theta=0 (asymptotic formula)."""
    obs = ObservationParameters(wavelengths=WL, angles=np.array([0.0]))
    S_th, S_ph = calculate_S(simple_body, obs)
    assert S_ph[0, 0] == pytest.approx(-S_th[0, 0], rel=1e-12)

# ── Backscattering (theta = pi) ───────────────────────────────────────────────

def test_backscattering_components_equal(simple_body):
    """S_th == S_ph at exactly theta=pi (asymptotic formula)."""
    obs = ObservationParameters(wavelengths=WL, angles=np.array([np.pi]))
    S_th, S_ph = calculate_S(simple_body, obs)
    assert S_th[0, 0] == pytest.approx(S_ph[0, 0], rel=1e-12)

# ── Mirror symmetry (magnitude) ───────────────────────────────────────────────
# Physically |S(2π−θ)| == |S(θ)| (the angular functions are odd under the
# reflection, so the complex value flips sign but the magnitude is preserved —
# this is what the polar scattering diagrams rely on).

def test_mirror_symmetry_magnitude_S_th(simple_body):
    theta = np.linspace(0.1, np.pi - 0.1, 25)
    mirror = 2 * np.pi - theta
    obs = ObservationParameters(wavelengths=WL, angles=np.concatenate([theta, mirror]))
    S_th, _ = calculate_S(simple_body, obs)
    np.testing.assert_allclose(
        np.abs(S_th[0, :25]), np.abs(S_th[0, 25:]), rtol=1e-10,
        err_msg="S_th magnitude mirror symmetry failed")

def test_mirror_symmetry_magnitude_S_ph(simple_body):
    theta = np.linspace(0.1, np.pi - 0.1, 25)
    mirror = 2 * np.pi - theta
    obs = ObservationParameters(wavelengths=WL, angles=np.concatenate([theta, mirror]))
    _, S_ph = calculate_S(simple_body, obs)
    np.testing.assert_allclose(
        np.abs(S_ph[0, :25]), np.abs(S_ph[0, 25:]), rtol=1e-10,
        err_msg="S_ph magnitude mirror symmetry failed")

def test_mirror_symmetry_sign_flip_S_th(simple_body):
    """The complex value flips sign under θ → 2π−θ: S_th(2π−θ) = −S_th(θ)."""
    theta = np.linspace(0.1, np.pi - 0.1, 25)
    mirror = 2 * np.pi - theta
    obs = ObservationParameters(wavelengths=WL, angles=np.concatenate([theta, mirror]))
    S_th, _ = calculate_S(simple_body, obs)
    np.testing.assert_allclose(S_th[0, 25:], -S_th[0, :25], rtol=1e-10)

# ── Optical theorem ───────────────────────────────────────────────────────────

def test_optical_theorem(simple_body):
    """|Im(S_th)| == |Im(S_ph)| at theta=0 (both equal in magnitude)."""
    obs = ObservationParameters(wavelengths=WL, angles=np.array([0.0]))
    S_th, S_ph = calculate_S(simple_body, obs)
    assert abs(S_th[0, 0].imag) == pytest.approx(abs(S_ph[0, 0].imag), rel=1e-12)
