from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import scipy.constants as const
c = const.speed_of_light

@dataclass
class BodyParameters:
    """
    Describes the physical layered sphere — invariant across wavelength.

    Fields:
      eps: array-like of relative permittivities (complex allowed). Length = num_layers + 1
      r: array-like of radii for layer boundaries. Length = num_layers
      conducting_core: whether core is conducting (bool)
      label: optional human-readable label
    """
    eps: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.complex128))
    r: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    conducting_core: bool = False
    label: Optional[str] = None

    def __post_init__(self):
        self.eps = np.asarray(self.eps, dtype=np.complex128)
        self.r = np.asarray(self.r, dtype=np.float64)

        if self.r.ndim != 1:
            raise ValueError("r must be a 1D sequence of radii.")
        if self.eps.ndim != 1:
            raise ValueError("eps must be a 1D sequence of permittivities (can be complex).")
        if len(self.eps) != len(self.r) + 1:
            raise ValueError("len(eps) must equal len(r) + 1")
        if len(self.r) == 0 or self.r[0] <= 0:
            raise ValueError("r[0] must be positive.")
        if np.any(self.r < 0):
            raise ValueError("All radii must be non-negative.")
        if np.any(np.diff(self.r) < 0):
            raise ValueError("Radii must be in non-decreasing order (innermost -> outermost).")

    def create_label(self, create_from: Optional[str] = None) -> str:
        if self.label:
            return self.label
        if create_from == 'r':
            return " ".join(f"r{i}={float(val):g}" for i, val in enumerate(self.r))
        if create_from == 'eps':
            # conducting core's eps[0] is an unused placeholder — don't label it
            start = 1 if self.conducting_core else 0
            eps_list = [f"ε{i}={self.eps[i]}" for i in range(start, len(self.eps)-1)]
            return " ".join(eps_list) if eps_list else "-"
        return "-"

    def to_dict(self) -> dict:
        return {
            "eps": self.eps.tolist(),
            "r": self.r.tolist(),
            "conducting_core": bool(self.conducting_core),
            "label": self.label
        }

    def __repr__(self) -> str:
        return (f"BodyParameters(label={self.label!r}, num_layers={len(self.r)}, "
                f"conducting_core={self.conducting_core})")


@dataclass
class ObservationParameters:
    """
    Describes the observation/excitation setup — independent of the body.

    Fields:
      wavelengths: one or more wavelengths in meters. A scalar is accepted and
                   stored as a 1-element array.
      angles: 1D array of angles in radians at which to evaluate S.

    Derived properties:
      k: wave number array (rad / m), shape == wavelengths.shape
      frequency_hz, frequency_ghz: wave frequencies (arrays)
    """
    wavelengths: np.ndarray = field(default_factory=lambda: np.array([1.0], dtype=np.float64))
    angles: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))

    def __post_init__(self):
        wl = np.asarray(self.wavelengths, dtype=np.float64)
        if wl.ndim == 0:
            wl = wl.reshape(1)
        if wl.ndim != 1 or len(wl) == 0:
            raise ValueError("wavelengths must be a scalar or a non-empty 1-D array.")
        if np.any(wl <= 0):
            raise ValueError("All wavelengths must be > 0.")
        self.wavelengths = wl

        angles = np.asarray(self.angles, dtype=np.float64)
        if angles.ndim != 1 or len(angles) == 0:
            raise ValueError("angles must be a non-empty 1-D array (radians).")
        self.angles = angles

    @property
    def k(self) -> np.ndarray:
        """Wave numbers in radians per meter, one per wavelength."""
        return 2.0 * np.pi / self.wavelengths

    @property
    def frequency_hz(self) -> np.ndarray:
        return c / self.wavelengths

    @property
    def frequency_ghz(self) -> np.ndarray:
        return self.frequency_hz / 1e9

    def __repr__(self) -> str:
        return (f"ObservationParameters(n_wavelengths={len(self.wavelengths)}, "
                f"n_angles={len(self.angles)})")


@dataclass
class PlotingParameters:
    title: str
    legend_title: str
    font_size: int
    show_ticklabels: bool

    experiment_description: str
    description_pos: tuple

    angle_limits: tuple

    experiments: BodyParameters
    polarization: str

    def __init__(self, Experiments,
                 Title = None,
                 LegendTitle = None,
                 ExpDescr = None,
                 DescriptionPos = [0.5, 0.1],
                 FontSize = 8,
                 ShowTicklabels = False,
                 Polarization = None,
                 AngleLimits = None):

        self.experiments = Experiments
        self.font_size = FontSize

        assert len(DescriptionPos) == 2, 'this should be x and y of the experiment_descriprion'
        self.description_pos = DescriptionPos

        self.title = Title
        self.legend_title = LegendTitle

        self.experiment_description = ExpDescr

        self.show_ticklabels = ShowTicklabels

        if Polarization is not None:
            assert Polarization == 'VV' or Polarization == 'HH', 'polarization could be only VV or HH'
        self.polarization = Polarization

        self.angle_limits = AngleLimits
