from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from analytical_diffraction import BodyParameters, ObservationParameters

FIDELITY_ANGLES = {"Low": 360, "Medium": 1200, "High": 3600}
DEFAULT_FIDELITY = "Medium"


@dataclass
class LayerSpec:
    """One region of the sphere: radius for the core (index 0), thickness for coatings."""

    thickness: float
    eps_real: float = 1.0
    eps_imag: float = 0.0

    @property
    def eps(self) -> complex:
        return complex(self.eps_real, self.eps_imag)

    def to_dict(self) -> dict:
        return {
            "thickness": self.thickness,
            "eps_real": self.eps_real,
            "eps_imag": self.eps_imag,
        }


@dataclass
class ExperimentState:
    """Everything the UI edits to define one experiment.

    Fields:
      layers: core plus coating layers, innermost first. `layers[0].thickness`
              is the core radius; every other `thickness` is a layer thickness.
      conducting_core: whether the core is a perfect conductor (its eps is then unused)
      outer_eps_real / outer_eps_imag: permittivity of the surrounding medium
      wavelength: excitation wavelength in meters
      fidelity: angular resolution preset, a key of FIDELITY_ANGLES
    """

    layers: list[LayerSpec] = field(default_factory=lambda: [LayerSpec(thickness=1.0)])
    conducting_core: bool = True
    outer_eps_real: float = 1.0
    outer_eps_imag: float = 0.0
    wavelength: float = 0.55
    fidelity: str = DEFAULT_FIDELITY

    def __post_init__(self):
        if not self.layers:
            raise ValueError("At least the core layer is required.")
        if self.wavelength <= 0:
            raise ValueError("wavelength must be > 0.")
        if self.fidelity not in FIDELITY_ANGLES:
            raise ValueError(f"fidelity must be one of {sorted(FIDELITY_ANGLES)}.")

    @property
    def n_angles(self) -> int:
        return FIDELITY_ANGLES[self.fidelity]

    def to_body(self) -> BodyParameters:
        r = np.cumsum([layer.thickness for layer in self.layers])
        eps = [layer.eps for layer in self.layers]
        eps.append(complex(self.outer_eps_real, self.outer_eps_imag))
        return BodyParameters(
            eps=np.asarray(eps, dtype=np.complex128),
            r=r,
            conducting_core=self.conducting_core,
        )

    def to_observation(self) -> ObservationParameters:
        angles = np.linspace(0.0, 2.0 * np.pi, self.n_angles, endpoint=False)
        return ObservationParameters(wavelengths=self.wavelength, angles=angles)

    def cache_key(self) -> tuple:
        return (
            tuple((l.thickness, l.eps_real, l.eps_imag) for l in self.layers),
            bool(self.conducting_core),
            (self.outer_eps_real, self.outer_eps_imag),
            float(self.wavelength),
            self.fidelity,
        )

    def to_dict(self) -> dict:
        return {
            "layers": [layer.to_dict() for layer in self.layers],
            "conducting_core": bool(self.conducting_core),
            "outer_eps_real": self.outer_eps_real,
            "outer_eps_imag": self.outer_eps_imag,
            "wavelength": self.wavelength,
            "fidelity": self.fidelity,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentState":
        layers = [
            LayerSpec(
                thickness=float(spec["thickness"]),
                eps_real=float(spec.get("eps_real", 1.0)),
                eps_imag=float(spec.get("eps_imag", 0.0)),
            )
            for spec in data["layers"]
        ]
        return cls(
            layers=layers,
            conducting_core=bool(data.get("conducting_core", False)),
            outer_eps_real=float(data.get("outer_eps_real", 1.0)),
            outer_eps_imag=float(data.get("outer_eps_imag", 0.0)),
            wavelength=float(data["wavelength"]),
            fidelity=str(data.get("fidelity", DEFAULT_FIDELITY)),
        )


def save_preset(path: str | Path, state: ExperimentState) -> None:
    Path(path).write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")


def load_preset(path: str | Path) -> ExperimentState:
    return ExperimentState.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
