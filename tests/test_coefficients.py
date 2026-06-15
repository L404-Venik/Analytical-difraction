# test_coefficients.py
import numpy as np
import pytest
from sphere_diffraction.parameters import (BodyParameters)
from sphere_diffraction.sphere_diffraction import (calculate_coefficients)


def make_body_k(eps_inner, eps_outer, r=0.1, wl=1.0, **kwargs):
    body = BodyParameters(eps=[eps_inner, eps_outer], r=[r], **kwargs)
    k = 2.0 * np.pi / wl
    return body, k


def test_no_scattering_when_eps_matched():
    """If sphere and medium have the same permittivity, no scattering occurs."""
    body, k = make_body_k(eps_inner=1.5, eps_outer=1.5, r=0.2)
    D_e, D_m, N = calculate_coefficients(body, k)
    assert np.allclose(D_e, 0, atol=1e-12)
    assert np.allclose(D_m, 0, atol=1e-12)

def test_coefficients_are_finite():
    """Coefficients should never be NaN or Inf for well-posed inputs."""
    body, k = make_body_k(eps_inner=2.5 + 0.1j, eps_outer=1.0, r=0.3, wl=0.5)
    D_e, D_m, N = calculate_coefficients(body, k)
    assert np.all(np.isfinite(D_e))
    assert np.all(np.isfinite(D_m))

@pytest.mark.xfail(
    reason="scipy.special overflows for large complex z (eps=1e8+1e8j); "
           "stable Bessel recurrence needed to verify this physics limit"
)
def test_conducting_core_vs_eps_limit():
    """
    A conducting core should give the same result as eps → ∞ dielectric.
    This is a physics sanity check, not exact equality.
    """
    k = 2.0 * np.pi / 1.0
    conducting = BodyParameters(
        eps=[1.0, 1.0], r=[0.1], conducting_core=True)
    # Very high eps approximates a conductor
    dielectric = BodyParameters(
        eps=[1e8 + 1e8j, 1.0], r=[0.1], conducting_core=False)
    D_e_c, D_m_c, _ = calculate_coefficients(conducting, k)
    D_e_d, D_m_d, _ = calculate_coefficients(dielectric, k)

    assert np.allclose(D_e_c[:5], D_e_d[:5], rtol=1e-3)
    assert np.allclose(D_m_c[:5], D_m_d[:5], rtol=1e-3)
