# Sphere Diffraction

A Python library for **Mie-theory electromagnetic scattering** by multilayer
spheres, and **inverse design** — searching for layer structures whose
scattering matches a target.

## What it does

- **Forward problem** — given a layered sphere (core + coatings) and an
  excitation, compute far-field scattering amplitudes, far-field and near-field
  E-fields, and Mie coefficients via a transfer-matrix method.
- **Inverse problem** — given a target scattering functional, search a discrete
  space of layer thicknesses and materials for the bodies that best match it.

## Install

```bash
git clone <repo-url>
cd Sphere-difraction
python -m venv .venv
.venv\Scripts\activate        # Windows;  source .venv/bin/activate on Unix
pip install numpy scipy matplotlib tqdm
```

## Quickstart — forward problem

```python
import numpy as np
from core.parameters import BodyParameters, ObservationParameters
from core.sphere_difraction import calculate_S

# A dielectric shell (eps = 2.1) on a conducting core, in air.
body = BodyParameters(
    eps=[1.0, 2.1, 1.0],          # [core, shell, surrounding medium]
    r=[0.01, 0.015],              # boundary radii in meters
    conducting_core=True,
)
obs = ObservationParameters(
    wavelengths=0.03,             # meters
    angles=np.linspace(0, np.pi, 361),
)
S_th, S_ph = calculate_S(body, obs)   # each shape (n_wavelengths, n_angles)
```

## Quickstart — inverse problem

```python
import numpy as np
from core.inverse_problem.search_space import SearchSpace, Layer, DiscreteRange
from core.inverse_problem.optimization import OptimizationTask, SolverConfig
from core.inverse_problem.brute_force_solver import BruteForceSolver

materials = {"glass": 2.25 + 0j, "foam": 1.2 + 0j}   # name -> relative permittivity

space = SearchSpace(
    core_radius=0.01,
    layers=[Layer(thickness=DiscreteRange(0.001, 0.005, 5), material=None)],  # None = any
    materials=materials,
    conducting_core=True,
)

# Minimize backscattering (theta = 0) at a single wavelength.
task = OptimizationTask(
    wavelengths=0.03,
    angles=np.linspace(0, 2 * np.pi, 360, endpoint=False),
    functional=lambda S_th, S_ph, angles: float(np.abs(S_th[0]) ** 2),
)

result = BruteForceSolver(SolverConfig(n_best=5, progress=False)).run(space, task)
best_F, best_body = result.best[0]
```

## Materials from a file

`core.materials.load_materials` reads a `{name: permittivity}` library from a CSV
of `name, eps_r, loss_tangent`, so materials can be edited without code:

```python
from core.materials import load_materials

materials = load_materials("examples/materials.csv")
space = SearchSpace(core_radius=0.01, layers=[...], materials=materials)
```

A starter library ships at [examples/materials.csv](examples/materials.csv).

## Conventions

- `eps` is **relative permittivity** (complex allowed), one value per region,
  ordered `[core, layer_1, …, layer_N, surrounding medium]` — length
  `num_layers + 1`.
- Refractive index `m` relates to permittivity as `eps = m²`. Optical
  databases usually publish `m = n + ik`; **square it** before putting it in a
  materials dict.

## Scope & limitations

**Permittivity is treated as constant (non-dispersive).** A materials dictionary
maps each name to a *single* complex permittivity, and the solver uses that same
value at *every* wavelength. There is no `ε(λ)` support, and modelling
wavelength-dependent permittivity is out of scope.

What this means in practice:

- **Pick `ε` for your band.** Look the material up in a proper source
  (e.g. [refractiveindex.info](https://refractiveindex.info)) at — or near — the
  wavelength you care about, convert `m = n + ik` to `eps = m²`, and use that.
- **Keep the band narrow.** Results are physical only where the material's true
  permittivity varies negligibly across the wavelengths you use. A band wide
  enough that `ε` changes appreciably — especially one straddling a resonance —
  is not modelled correctly.
- **Broadband mode carries the same assumption.** It evaluates the *same* body
  (same `ε`) at each wavelength, so it is trustworthy only over a band where
  dispersion is negligible. It does **not** account for `ε` changing with
  wavelength.
- **The solver does not check this.** It holds no dispersion data, so it cannot
  know whether your chosen `ε` is valid across your band. That judgement is the
  caller's responsibility.

## Tests

```bash
pytest
```

## Documentation

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for modules, public signatures,
and data flow.

## Acknowledgments

- **Claude** (Anthropic) — pair-programming
