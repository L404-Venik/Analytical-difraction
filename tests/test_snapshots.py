import os
import numpy as np
import pytest
from analytical_diffraction.parameters import ObservationParameters
from analytical_diffraction.sphere import calculate_S
from snapshot_configs import known_cases, SNAPSHOT_ANGLES, SNAPSHOT_M

SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "snapshots")

_cases = known_cases()

# The legacy snapshots stored M=3600 samples over [0, 2π), but only the
# [0, π] half (indices 0..M/2) was ever computed directly — the second half
# was an unnegated mirror copy (a plotting convenience, since |S(2π−θ)|=|S(θ)|).
# The refactored calculate_S computes every angle directly, so its second half
# is the true S(θ) — sign-flipped vs the old copy. We therefore regression-test
# against the directly-computed half only; it covers every independent value.
_HALF = SNAPSHOT_M // 2 + 1   # indices 0..1800 inclusive (θ from 0 to π)


@pytest.mark.parametrize("name,body,wavelength", _cases, ids=[c[0] for c in _cases])
def test_S_matches_snapshot(name, body, wavelength):
    snapshot_path = os.path.join(SNAPSHOT_DIR, f"{name}.npz")
    if not os.path.exists(snapshot_path):
        pytest.skip(f"snapshot missing — run scripts/generate_snapshots.py")

    ref = np.load(snapshot_path)
    observation = ObservationParameters(wavelengths=wavelength, angles=SNAPSHOT_ANGLES)
    S_th, S_ph = calculate_S(body, observation)

    np.testing.assert_allclose(S_th[0, :_HALF], ref["S_th"][:_HALF], rtol=1e-10)
    np.testing.assert_allclose(S_ph[0, :_HALF], ref["S_ph"][:_HALF], rtol=1e-10)
