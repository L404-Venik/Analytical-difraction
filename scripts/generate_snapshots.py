#!/usr/bin/env python3
"""
Capture known-good S_th / S_ph snapshots for the known example configurations.

This overwrites existing snapshots unconditionally — only run when the current
output is known to be correct.
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "tests"))

import numpy as np
from sphere_diffraction.parameters import ObservationParameters
from sphere_diffraction.sphere_diffraction import calculate_S
from snapshot_configs import known_cases, SNAPSHOT_ANGLES

SNAPSHOT_DIR = os.path.join(PROJECT_ROOT, "tests", "snapshots")


def main():
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    cases = known_cases()
    for name, body, wavelength in cases:
        observation = ObservationParameters(wavelengths=wavelength, angles=SNAPSHOT_ANGLES)
        S_th, S_ph = calculate_S(body, observation)
        path = os.path.join(SNAPSHOT_DIR, f"{name}.npz")
        np.savez(path, S_th=S_th[0], S_ph=S_ph[0])
        print(f"  saved {name}.npz")
    print(f"\nDone — {len(cases)} snapshots written to {SNAPSHOT_DIR}")


if __name__ == "__main__":
    main()
