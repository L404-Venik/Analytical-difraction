# Architecture

## Overview

Physics simulation library for far-field and near-field electromagnetic scattering by multilayer spheres (Mie theory), plus an inverse problem framework for finding sphere structures that match target scattering profiles.

## Physical model assumptions

**Constant (non-dispersive) permittivity.** Every region's permittivity is a
single complex number (`BodyParameters.eps`, `SearchSpace.materials` values),
used unchanged at all wavelengths. The library carries no `ε(λ)` data and never
checks whether a chosen `ε` is valid across the requested band — that is the
caller's responsibility. Broadband tasks reuse one body across all wavelengths,
so they inherit this assumption: results are physical only over a band where the
materials' true permittivity varies negligibly. See the README's
"Scope & limitations" for guidance.

## Core (`core/`)

### `parameters.py`
Three dataclasses, with a deliberate split between the **physical body** (invariant
across wavelength) and the **observation setup** (what/where we probe):
- **`BodyParameters`** — the physical layered sphere. Holds `eps` (permittivities per layer, complex), `r` (radii of layer boundaries), `conducting_core`, optional `label`. Validates on construction. Carries no wavelength.
- **`ObservationParameters`** — the excitation/observation setup. Holds `wavelengths` (scalar accepted, stored as a 1-element array) and `angles` (1D, radians). Derived properties: `k` (wavenumber array), `frequency_hz/ghz` (arrays).
- **`PlotingParameters`** — display config for matplotlib plots.

There is no composite "experiment" type: an experiment is the *action* `calculate_S(body, observation)`, not a struct. This keeps the two lifetimes independent — e.g. the inverse solver sweeps many bodies against one fixed observation.

### `sphere_difraction.py`
All direct-problem physics. Key functions:
- `calculate_coefficients(body, k)` → `(D_e, D_m, N)` — Mie scattering coefficients via transfer-matrix method. Builds T-matrices layer by layer, applies inner boundary condition (conducting or dielectric core), stops when series converges. Takes an explicit scalar `k`.
- `calculate_S(body, observation)` → `(S_th, S_ph)`, each shape `(n_wavelengths, n_angles)` — far-field scattering amplitudes at the given wavelengths and arbitrary angles. The angle axis is fully vectorized (matmul over the series); the wavelength loop is unavoidable (k-dependent), but the angle-dependent Legendre terms are computed once and reused across wavelengths. θ=0 and θ=π are handled by asymptotic formulas via boolean masks (the general term is singular there) — automatic and invisible to the caller. No mirror-symmetry optimization: every angle is computed directly, so `S(2π−θ) = −S(θ)` holds as true physics rather than a copied half.
- `calculate_electric_field_far(body, observation)` — far-field E field, shape `(n_wavelengths, n_angles)`.
- `calculate_electric_field_close_vectorized(body, k, limits)` — near-field E on a 2D grid; uses disk cache (`.npz`) for expensive angular-function arrays.
- Helper functions: `psi`, `xi`, `psi_derivative`, `xi_derivative` (Riccati–Bessel functions); `assoc_legendre_derivative` and its vectorized variant.

### `ploting_functions.py`
Matplotlib wrappers for visualizing scattering patterns.

### `materials.py`
Material-library helpers. `lossy_eps(eps_r, loss_tangent)` converts a real relative permittivity and loss tangent to the complex permittivity the solver uses, with a `+i` lossy convention (`eps = eps_r·(1 + i·tanδ)`). `load_materials(path)` parses a CSV of `name, eps_r, loss_tangent` rows (extra columns like `source`/`valid_band` ignored) into the `{name: complex}` dict that `SearchSpace` consumes, validating names and ranges with line-numbered errors. The parser is regime-agnostic; an example library ships at `examples/materials.csv`.

## Inverse Problem (`core/inverse_problem/`)

Framework for finding sphere structures whose scattering matches a target functional.
The search produces **bodies**; the wavelengths and angles to probe live in the task,
which internally builds an `ObservationParameters` to drive `calculate_S`.

### `optimization.py` — shared data types
- **`OptimizationTask`** — what to optimize: `wavelengths` (a scalar is accepted and stored as a 1-element array), angles array, and `functional(S_th, S_ph, angles) → float`. Builds an `ObservationParameters` for the solver. The functional is evaluated once per wavelength and the results are aggregated (`SolverConfig.aggregation`), single-wavelength tasks included.
- **`SolverConfig`** — controls `n_best` (how many top candidates to return), `aggregation` rule for broadband (`mean`/`max`/`sum`/custom callable), and `progress` flag.
- **`SolverResult`** — output: `best` (list of `(F, BodyParameters)` sorted ascending), `n_evaluated`, `n_skipped` (count of candidates whose objective was non-finite, NaN/inf, and excluded from ranking), `elapsed_seconds`.

### `solver_base.py`
Abstract `Solver` base class. One method to implement: `run(space, task) → SolverResult`.

### `search_space.py`
- **`DiscreteRange`** / **`ContinuousRange`** — thickness axis types for layer specs.
- **`Layer`** — one coating layer with `thickness` (fixed float / `DiscreteRange` / `ContinuousRange`) and `material` (fixed name / list of names / `None` = all materials). Validates and coerces both at construction: a fixed thickness is coerced to `float` and must be `> 0`; `material` must be `str`/`list[str]`/`None`, a list must be non-empty, and duplicates are dropped (order preserved) with a `UserWarning`.
- **`SearchSpace`** — full discrete parameter space. Defines core + list of layers + material library. `iter_candidates()` yields `BodyParameters` (no wavelength) using `itertools.product`; filters by **eps value** (not material name): drops candidates whose first layer matches the core, whose adjacent layers match each other, or whose last layer matches the outer medium, plus thickness budget violations. Supports `up_to=True` to search 1..N layer counts. `size_estimate()` gives an upper bound before filtering. `validate_discrete()` raises if any layer thickness is a `ContinuousRange`.

### `brute_force_solver.py`
**`BruteForceSolver`** — iterates every candidate body from `SearchSpace`, evaluates the functional against the task's observation, keeps top-N. Validates discreteness at `run()` entry (`space.validate_discrete()`), raises `ValueError` if the space produces no candidates, and probes the first candidate so a buggy functional surfaces as a `RuntimeError` rather than being swallowed; in the sweep only non-finite objectives are skipped (counted in `n_skipped`). Falls back to a plain percentage counter when tqdm is absent. Single-threaded.

## Tests (`tests/`)

pytest suite covering:
- `test_coefficients.py` — Mie coefficient correctness
- `test_special.py` — Riccati–Bessel helpers
- `test_difraction.py` — scattering amplitude functions (shape, forward/backward limits, magnitude mirror symmetry, sign-flip parity, optical theorem)
- `test_body_parameters.py` — `BodyParameters` validation
- `test_observation_parameters.py` — `ObservationParameters` validation and derived `k`/frequency arrays
- `test_snapshots.py` — end-to-end snapshot tests for 16 diploma example configurations. Compares only the directly-computed `[0, π]` half of the legacy snapshots (the old second half was an unnegated mirror copy; the refactor computes every angle directly, so its second half is sign-flipped — see `calculate_S`).
- `test_optimization.py` — `OptimizationTask`, `SolverConfig`, `SolverResult`, `BruteForceSolver`
- `test_search_space.py` — `SearchSpace` iteration, filtering, and size estimation
- `test_materials.py` — `lossy_eps`/`load_materials` parsing and validation, plus a guard pinning the `+i` lossy permittivity convention

## Examples (`examples/`)

- **`materials.csv`** — example material library (`name, eps_r, loss_tangent`, with `source`/`valid_band` provenance columns) for `load_materials`. A starting point users can copy and edit without writing code.

## Scripts

- **`scripts/generate_snapshots.py`** — re-captures `S_th`/`S_ph` snapshots into `tests/snapshots/*.npz`. Run when results have intentionally changed.
- **`snapshot_configs.py`** (project root) — shared config defining the 16 named `(name, BodyParameters, wavelength)` cases plus the snapshot angle grid (`SNAPSHOT_ANGLES`, the legacy M=3600 evenly-spaced grid), used by both the generator and the test suite.

## Dependencies

- `numpy`, `scipy` — numerics
- `matplotlib` — plotting
- `tqdm` (optional) — progress bars in brute-force search
