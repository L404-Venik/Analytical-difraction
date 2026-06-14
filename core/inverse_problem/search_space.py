# search_space.py — Parameter space definition for multilayer sphere synthesis.

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterator, Optional, Union
import numpy as np

from ..parameters import BodyParameters


# ---------------------------------------------------------------------------
# Thickness axis types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DiscreteRange:
    """Uniform grid of thickness values from min to max (inclusive), steps points."""
    min: float
    max: float
    steps: int

    def __post_init__(self):
        if self.min <= 0:
            raise ValueError(f"DiscreteRange.min must be > 0, got {self.min}")
        if self.max < self.min:
            raise ValueError(f"DiscreteRange.max ({self.max}) must be >= min ({self.min})")
        if self.steps < 1:
            raise ValueError(f"DiscreteRange.steps must be >= 1, got {self.steps}")

    def values(self) -> np.ndarray:
        return np.linspace(self.min, self.max, self.steps)

    def __len__(self) -> int:
        return self.steps


@dataclass(frozen=True)
class ContinuousRange:
    """Continuous thickness interval (min..max). Not iterable by the discrete search."""
    min: float
    max: float

    def __post_init__(self):
        if self.min <= 0:
            raise ValueError(f"ContinuousRange.min must be > 0, got {self.min}")
        if self.max < self.min:
            raise ValueError(f"ContinuousRange.max ({self.max}) must be >= min ({self.min})")


ThicknessSpec = Union[float, DiscreteRange, ContinuousRange]
MaterialSpec  = Union[str, list[str], None]   # None = all materials


# ---------------------------------------------------------------------------
# Layer specification
# ---------------------------------------------------------------------------

@dataclass
class Layer:
    """
    Specification for one coating layer.

    thickness: float            — fixed thickness
               DiscreteRange    — grid of values to search
               ContinuousRange  — continuous interval (not iterable in discrete search)

    material:  str              — fixed material name
               list[str]        — subset of materials to search over
               None             — search over all materials in the dict
    """
    thickness: ThicknessSpec
    material:  MaterialSpec = None

    def _thickness_values(self) -> list[float]:
        if isinstance(self.thickness, float):
            return [self.thickness]
        if isinstance(self.thickness, DiscreteRange):
            return list(self.thickness.values())
        if isinstance(self.thickness, ContinuousRange):
            raise TypeError(
                "ContinuousRange cannot be iterated in discrete search. "
                "Convert to DiscreteRange or use a continuous optimizer."
            )
        raise TypeError(f"Unknown thickness type: {type(self.thickness)}")

    def _material_names(self, materials: dict[str, complex]) -> list[str]:
        if isinstance(self.material, str):
            return [self.material]
        if isinstance(self.material, list):
            return list(self.material)
        if self.material is None:
            return list(materials.keys())
        raise TypeError(f"Unknown material spec type: {type(self.material)}")

    def n_thickness_options(self) -> int:
        if isinstance(self.thickness, float):
            return 1
        if isinstance(self.thickness, DiscreteRange):
            return self.thickness.steps
        return 1  # ContinuousRange: treated as 1 for size estimate

    def n_material_options(self, materials: dict[str, complex]) -> int:
        return len(self._material_names(materials))


# ---------------------------------------------------------------------------
# Main SearchSpace class
# ---------------------------------------------------------------------------

class SearchSpace:
    """
    Defines the discrete parameter space for multilayer sphere synthesis.

    Fixed (not searched):
      - core_radius, conducting_core, core_material
      - eps_outer (permittivity of surrounding medium)

    Searched:
      - per-layer thickness and material (as specified in each Layer)
      - number of layers if up_to=True

    Parameters
    ----------
    core_radius : float
        Radius of the sphere core.
    layers : list[Layer]
        Layer specifications, ordered innermost → outermost.
        If up_to=True, defines the *maximum* layers; shorter stacks are also searched.
    materials : dict[str, complex]
        Material library {name: permittivity}.
    conducting_core : bool
        Whether the core is a perfect conductor.
    core_material : complex
        Permittivity of the core (ignored if conducting_core=True).
    eps_outer : complex
        Permittivity of the surrounding medium (default: 1.0, vacuum/air).
    up_to : bool
        If True, search all stacks from 1 layer up to len(layers) layers.
        If False (default), search only exactly len(layers) layers.
    max_total_thickness : float or None
        If set, skip any candidate whose total coating thickness exceeds this value.
    """

    def __init__(
        self,
        core_radius: float,
        layers: list[Layer],
        materials: dict[str, complex],
        conducting_core: bool = False,
        core_material: complex = 1.0 + 0j,
        eps_outer: complex = 1.0 + 0j,
        up_to: bool = False,
        max_total_thickness: Optional[float] = None,
    ):
        if core_radius <= 0:
            raise ValueError(f"core_radius must be > 0, got {core_radius}")
        if len(layers) == 0:
            raise ValueError("layers must not be empty.")
        if max_total_thickness is not None and max_total_thickness <= 0:
            raise ValueError(f"max_total_thickness must be > 0, got {max_total_thickness}")

        self.core_radius       = core_radius
        self.layers            = layers
        self.materials         = dict(materials)
        self.conducting_core   = conducting_core
        self.core_material     = complex(core_material)
        self.eps_outer         = complex(eps_outer)
        self.up_to             = up_to
        self.max_total_thickness = max_total_thickness

        self._validate_materials()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_materials(self):
        """Check that all named materials in layer specs exist in the dictionary."""
        for i, layer in enumerate(self.layers):
            if isinstance(layer.material, str):
                if layer.material not in self.materials:
                    raise ValueError(
                        f"Layer {i}: material '{layer.material}' not found in materials dict. "
                        f"Available: {list(self.materials.keys())}"
                    )
            elif isinstance(layer.material, list):
                for name in layer.material:
                    if name not in self.materials:
                        raise ValueError(
                            f"Layer {i}: material '{name}' not found in materials dict. "
                            f"Available: {list(self.materials.keys())}"
                        )

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def _iter_stack(self, stack: list[Layer]) -> Iterator[BodyParameters]:
        """
        Yield all valid BodyParameters for a fixed-length stack of layers.

        Validity rules:
          - no two consecutive layers share the same material
          - total thickness does not exceed max_total_thickness (if set)
          - first layer material != core material (unless conducting core)
        """
        # Build per-layer option lists: list of (thickness, material_name) pairs
        per_layer_options: list[list[tuple[float, str]]] = []
        for layer in stack:
            thicknesses = layer._thickness_values()
            mat_names   = layer._material_names(self.materials)
            combos = [(t, m) for t in thicknesses for m in mat_names]
            per_layer_options.append(combos)

        for combination in itertools.product(*per_layer_options):
            # combination: tuple of (thickness, material_name), one per layer

            # --- Check consecutive same-material constraint (by eps value) ---
            # Two layers with identical eps create a phantom interface — no physical
            # meaning and the solver treats them as one layer anyway.
            materials_chosen = [c[1] for c in combination]
            eps_chosen = [self.materials[name] for name in materials_chosen]

            # First layer vs core (skip if conducting — core has no eps to compare)
            if not self.conducting_core:
                if eps_chosen[0] == self.core_material:
                    continue

            # Adjacent layers
            skip = False
            for j in range(len(eps_chosen) - 1):
                if eps_chosen[j] == eps_chosen[j + 1]:
                    skip = True
                    break
            if skip:
                continue

            # Last layer vs outer medium
            if eps_chosen[-1] == self.eps_outer:
                continue

            # --- Check total thickness constraint ---
            thicknesses_chosen = [c[0] for c in combination]
            if self.max_total_thickness is not None:
                if sum(thicknesses_chosen) > self.max_total_thickness:
                    continue

            # --- Build BodyParameters ---
            # r: [core_radius, core_radius + t1, core_radius + t1 + t2, ...]
            r = [self.core_radius]
            for t in thicknesses_chosen:
                r.append(r[-1] + t)

            # eps: [core_eps, layer1_eps, layer2_eps, ..., eps_outer]
            if self.conducting_core:
                # eps[0] is required by BodyParameters but unused for a conducting
                # core; 1.0 is a placeholder the solver ignores.
                eps = [1.0 + 0j]
            else:
                eps = [self.core_material]
            for name in materials_chosen:
                eps.append(self.materials[name])
            eps.append(self.eps_outer)

            yield BodyParameters(
                eps=np.array(eps, dtype=np.complex128),
                r=np.array(r, dtype=np.float64),
                conducting_core=self.conducting_core,
                label=None,
            )

    def iter_candidates(self) -> Iterator[BodyParameters]:
        """
        Iterate over all valid candidate bodies in the search space.
        """
        n_max = len(self.layers)
        n_min = 1 if self.up_to else n_max

        for n in range(n_min, n_max + 1):
            yield from self._iter_stack(self.layers[:n])

    def __iter__(self) -> Iterator[BodyParameters]:
        return self.iter_candidates()

    # ------------------------------------------------------------------
    # Size estimate
    # ------------------------------------------------------------------

    def _estimate_stack_size(self, n_layers: int) -> int:
        """
        Upper bound on combinations for a stack of n_layers.
        Does not account for the same-material filtering (those are hard to count
        analytically), so this is an overestimate.
        """
        total = 1
        for layer in self.layers[:n_layers]:
            total *= layer.n_thickness_options() * layer.n_material_options(self.materials)
        return total

    def size_estimate(self) -> int:
        """Upper bound on the number of candidates (before filtering)."""
        n_max = len(self.layers)
        n_min = 1 if self.up_to else n_max
        return sum(self._estimate_stack_size(n) for n in range(n_min, n_max + 1))

    def __repr__(self) -> str:
        mode = f"up to {len(self.layers)}" if self.up_to else f"exactly {len(self.layers)}"
        thickness_info = (
            f", max_total_thickness={self.max_total_thickness}" 
            if self.max_total_thickness else ""
        )
        return (
            f"SearchSpace("
            f"core_radius={self.core_radius}, "
            f"n_layers={mode}, "
            f"n_materials={len(self.materials)}, "
            f"~{self.size_estimate():,} candidates (before filtering)"
            f"{thickness_info})"
        )
