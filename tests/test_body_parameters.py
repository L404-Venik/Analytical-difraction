# test_body_parameters.py
import numpy as np
import re
import pytest
from sphere_diffraction.parameters import (BodyParameters)


# ── Constructor / normalization ───────────────────────────────────────────────

def test_valid_single_layer():
    b = BodyParameters(eps=[1.5, 1.0], r=[0.1])
    assert len(b.r) == 1
    assert b.conducting_core is False

def test_complex_eps_accepted():
    """Lossy materials have a complex permittivity — must not raise."""
    b = BodyParameters(eps=[2.5 + 0.3j, 1.0], r=[0.1])
    assert b.eps[0].imag == pytest.approx(0.3)

def test_inputs_are_converted_to_arrays():
    """Lists and tuples must be silently converted to ndarrays."""
    b = BodyParameters(eps=(2.0, 1.0), r=(0.1,))
    assert isinstance(b.eps, np.ndarray)
    assert isinstance(b.r,   np.ndarray)

def test_eps_dtype_is_complex128():
    b = BodyParameters(eps=[1.5, 1.0], r=[0.1])
    assert b.eps.dtype == np.complex128

def test_r_dtype_is_float64():
    b = BodyParameters(eps=[1.5, 1.0], r=[0.1])
    assert b.r.dtype == np.float64

# ── Validation: r ─────────────────────────────────────────────────────────────

def test_2d_r_raises():
    with pytest.raises(ValueError, match="r must be a 1D"):
        BodyParameters(eps=[1.5, 1.0, 1.0], r=[[0.1, 0.2]])

def test_empty_r_raises():
    """len(r) == 0 should be caught by the r[0] > 0 guard."""
    with pytest.raises(ValueError, match=re.escape("r[0] must be positive.")):
        BodyParameters(eps=[1.0], r=[])

def test_r0_zero_raises():
    with pytest.raises(ValueError, match=re.escape("r[0] must be positive.")):
        BodyParameters(eps=[1.5, 1.0], r=[0.0])

def test_r0_negative_raises():
    with pytest.raises(ValueError, match=re.escape("r[0] must be positive.")):
        BodyParameters(eps=[1.5, 1.0], r=[-0.1])

def test_inner_radius_negative_raises():
    """r[0] is valid but a later radius is negative."""
    with pytest.raises(ValueError, match="All radii must be non-negative."):
        BodyParameters(eps=[1.5, 1.0, 1.0], r=[0.1, -0.05])

def test_unsorted_radii_raises():
    with pytest.raises(ValueError, match=re.escape("Radii must be in non-decreasing order")):
        BodyParameters(eps=[1.5, 1.0, 1.0], r=[0.3, 0.1])

def test_equal_radii_accepted():
    """Duplicate radii represent a zero-thickness layer — allowed."""
    BodyParameters(eps=[2.0, 1.5, 1.0], r=[0.1, 0.1])

# ── Validation: eps ───────────────────────────────────────────────────────────

def test_2d_eps_raises():
    with pytest.raises(ValueError, match="eps must be a 1D"):
        BodyParameters(eps=[[1.5, 1.0]], r=[0.1])

def test_too_few_permittivities_raises():
    with pytest.raises(ValueError, match=re.escape("len(eps) must equal len(r) + 1")):
        BodyParameters(eps=[1.5], r=[0.1, 0.2])

def test_too_many_permittivities_raises():
    """Extra eps are not silently dropped — they signal a miscounted body."""
    with pytest.raises(ValueError, match=re.escape("len(eps) must equal len(r) + 1")):
        BodyParameters(eps=[2.0, 1.5, 1.0, 1.0], r=[0.1, 0.2])

# ── to_dict ───────────────────────────────────────────────────────────────────

def test_to_dict_round_trip():
    b = BodyParameters(
        eps=[2.0 + 0.1j, 1.0], r=[0.2],
        conducting_core=True, label="test"
    )
    d = b.to_dict()
    b2 = BodyParameters(**d)
    np.testing.assert_array_almost_equal(b2.eps, b.eps)
    np.testing.assert_array_almost_equal(b2.r,   b.r)
    assert b2.conducting_core == b.conducting_core
    assert b2.label           == b.label

def test_to_dict_contains_expected_keys():
    b = BodyParameters(eps=[1.5, 1.0], r=[0.1])
    assert set(b.to_dict().keys()) == {"eps", "r", "conducting_core", "label"}


# ── __repr__ ──────────────────────────────────────────────────────────────────

def test_repr_smoke():
    """repr must not raise and should mention key fields."""
    b = BodyParameters(eps=[1.5, 1.0], r=[0.1], label="x")
    r = repr(b)
    assert "x" in r
    assert "num_layers" in r
