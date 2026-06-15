"""
Shared example configurations used by both the snapshot generator script
and the snapshot test suite.
"""
import numpy as np
import scipy.constants as const
from analytical_diffraction.parameters import BodyParameters

c = const.speed_of_light

_lambdas = [c / (0.3e9), c / (1.0e9), c / (2.0e9)]
_freq_labels = ["0.3GHz", "1.0GHz", "2.0GHz"]

# Snapshots were captured over this evenly-spaced angle grid (the legacy
# calculate_S default of M=3600). Shared so the generator and the tests agree.
SNAPSHOT_M = 3600
SNAPSHOT_ANGLES = np.arange(SNAPSHOT_M) * (2.0 * np.pi / SNAPSHOT_M)


def known_cases():
    """Return list of (name, BodyParameters, wavelength) for all 16 configurations."""
    cases = []

    # Conducting sphere with lossless dielectric layer
    for lmbd, freq in zip(_lambdas, _freq_labels):
        cases.append((
            f"conducting_sphere_{freq}",
            BodyParameters(np.array([2.56, 1.0]), np.array([0.5]), True),
            lmbd,
        ))

    # Lossless dielectric sphere
    for lmbd, freq in zip(_lambdas, _freq_labels):
        cases.append((
            f"lossless_dielectric_{freq}",
            BodyParameters(np.array([2.56, 1.0]), np.array([0.5]), False),
            lmbd,
        ))

    # Lossy dielectric sphere
    for lmbd, freq in zip(_lambdas, _freq_labels):
        cases.append((
            f"lossy_dielectric_{freq}",
            BodyParameters(np.array([2.56 + 0.102j, 1.0]), np.array([0.5]), False),
            lmbd,
        ))

    # Lossless coated conducting sphere (two-layer)
    for lmbd, freq in zip(_lambdas, _freq_labels):
        cases.append((
            f"lossless_coated_{freq}",
            BodyParameters(np.array([1.0, 2.56, 1.0]), np.array([0.5, 0.55]), True),
            lmbd,
        ))

    # Lossy coated conducting sphere (two-layer)
    for lmbd, freq in zip(_lambdas, _freq_labels):
        cases.append((
            f"lossy_coated_{freq}",
            BodyParameters(np.array([1.0, 2.56 + 0.102j, 1.0]), np.array([0.5, 0.55]), True),
            lmbd,
        ))

    # Three-layer conducting core
    lmbd = 0.5
    cases.append((
        "three_layer_conducting_core",
        BodyParameters(
            np.array([1.0, 7.0, 5.0, 1.0]),
            np.array([0.25 * lmbd, 0.6 * lmbd, 1.0 * lmbd]),
            True,
        ),
        lmbd,
    ))

    return cases
