
# test_special.py
import numpy as np
import pytest
from sphere_diffraction.sphere_diffraction import (xi, xi_derivative, psi, psi_derivative)

@pytest.mark.parametrize("n", [0, 1, 2, 5, 10])
@pytest.mark.parametrize("x", [0.1, 1.0, 5.0, 20.0])
def test_wronskian_identity(n, x):
    """ξ'ψ - ξψ' = i for all n, x."""
    wronskian = xi_derivative(n, x) * psi(n, x) - xi(n, x) * psi_derivative(n, x)
    assert wronskian == pytest.approx(1j, rel=1e-10)

@pytest.mark.parametrize("n", [1, 2, 5])
def test_psi_is_real_for_real_argument(n):
    """ψ_n(x) = x·j_n(x) must be real when x is real."""
    result = psi(n, 5.0)
    assert np.imag(result) == pytest.approx(0.0, abs=1e-14)