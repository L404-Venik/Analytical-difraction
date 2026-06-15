"""
Material library helpers.

A material is described by its real relative permittivity and loss tangent — the
form most dielectric tables publish. `lossy_eps` converts that pair to the complex
relative permittivity the solver uses; `load_materials` parses a CSV of such rows
into the ``{name: complex permittivity}`` dict that `SearchSpace` consumes.

Permittivity is taken as a single constant value, independent of wavelength — see
the README's "Scope & limitations". A ready-made example library lives at
`examples/materials.csv`.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Union

__all__ = ["lossy_eps", "load_materials"]

_REQUIRED_COLUMNS = ("name", "eps_r", "loss_tangent")


def lossy_eps(eps_r: float, loss_tangent: float = 0.0) -> complex:
    """
    Complex relative permittivity from a real permittivity and loss tangent.

    Returns ``eps_r * (1 + i*loss_tangent)`` — a positive imaginary part,
    matching the lossy sign convention used throughout the solver.
    """
    return complex(eps_r, eps_r * loss_tangent)


def load_materials(path: Union[str, Path]) -> dict[str, complex]:
    """
    Load a material library from a CSV file.

    Required columns: ``name``, ``eps_r``, ``loss_tangent``. Any further columns
    (e.g. ``source``, ``valid_band``) are ignored. Each row becomes one entry in
    the returned ``{name: complex relative permittivity}`` dict, ready to pass to
    ``SearchSpace(materials=...)``.

    Raises ``ValueError`` with the offending line number on a missing column,
    duplicate name, empty name, non-numeric value, ``eps_r < 1``, or negative
    loss tangent.
    """
    path = Path(path)
    materials: dict[str, complex] = {}

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: file is empty.")
        missing = [c for c in _REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"{path}: missing required column(s): {', '.join(missing)}. "
                f"Found: {', '.join(reader.fieldnames)}"
            )

        # Data rows start on line 2 (line 1 is the header).
        for lineno, row in enumerate(reader, start=2):
            name = (row["name"] or "").strip()
            if not name:
                raise ValueError(f"{path}:{lineno}: empty material name.")
            if name in materials:
                raise ValueError(f"{path}:{lineno}: duplicate material name '{name}'.")

            try:
                eps_r = float(row["eps_r"])
                loss_tangent = float(row["loss_tangent"])
            except (TypeError, ValueError):
                raise ValueError(
                    f"{path}:{lineno}: eps_r and loss_tangent must be numbers "
                    f"(got eps_r={row['eps_r']!r}, loss_tangent={row['loss_tangent']!r})."
                )

            if eps_r < 1.0:
                raise ValueError(
                    f"{path}:{lineno}: eps_r must be >= 1, got {eps_r}."
                )
            if loss_tangent < 0.0:
                raise ValueError(
                    f"{path}:{lineno}: loss_tangent must be >= 0, got {loss_tangent}."
                )

            materials[name] = lossy_eps(eps_r, loss_tangent)

    if not materials:
        raise ValueError(f"{path}: no material rows found.")
    return materials
