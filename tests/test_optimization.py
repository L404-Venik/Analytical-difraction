# test_optimization.py — Test suite for optimization.py and brute_force_solver.py

import numpy as np
import pytest

from core.parameters import BodyParameters, ObservationParameters
from core.inverse_problem.search_space import (ContinuousRange, DiscreteRange, Layer, SearchSpace)
from core.inverse_problem.optimization import (OptimizationTask, SolverConfig, SolverResult)
from core.inverse_problem.brute_force_solver import (BruteForceSolver)


# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------

MATS = {
    "glass":  2.25 + 0j,
    "teflon": 2.1  + 0j,
    "foam":   1.2  + 0j,
}

ANGLES_COARSE = np.linspace(0, 2 * np.pi, 36, endpoint=False)

BACKSCATTER = lambda S_th, S_ph, angles: float(np.abs(S_th[0]) ** 2)


def simple_space(n_thickness_steps: int = 3) -> SearchSpace:
    return SearchSpace(
        core_radius=0.01,
        layers=[Layer(thickness=DiscreteRange(0.001, 0.005, n_thickness_steps), material=None)],
        materials=MATS,
        conducting_core=True,
        eps_outer=1.0 + 0j,
    )


def single_wavelength_task(wl: float = 0.03) -> OptimizationTask:
    return OptimizationTask(
        wavelength=wl,
        angles=ANGLES_COARSE,
        functional=BACKSCATTER,
    )


def quiet_solver(**kwargs) -> BruteForceSolver:
    return BruteForceSolver(SolverConfig(progress=False, **kwargs))


# ===========================================================================
# OptimizationTask
# ===========================================================================

class TestOptimizationTask:

    def test_single_wavelength_scalar(self):
        task = OptimizationTask(wavelength=0.03, angles=ANGLES_COARSE, functional=BACKSCATTER)
        assert not task.is_broadband
        assert task.wavelength == 0.03

    def test_single_wavelength_array_length1(self):
        task = OptimizationTask(
            wavelength=np.array([0.03]),
            angles=ANGLES_COARSE,
            functional=BACKSCATTER,
        )
        assert not task.is_broadband
        assert task.wavelength == 0.03

    def test_broadband_array(self):
        wls = np.linspace(0.01, 0.05, 10)
        task = OptimizationTask(wavelength=wls, angles=ANGLES_COARSE, functional=BACKSCATTER)
        assert task.is_broadband
        assert len(task.wavelengths) == 10

    def test_broadband_raises_on_wavelength_property(self):
        task = OptimizationTask(
            wavelength=np.linspace(0.01, 0.05, 5),
            angles=ANGLES_COARSE,
            functional=BACKSCATTER,
        )
        with pytest.raises(AttributeError):
            _ = task.wavelength

    def test_wavelengths_always_array(self):
        task = OptimizationTask(wavelength=0.03, angles=ANGLES_COARSE, functional=BACKSCATTER)
        assert isinstance(task.wavelengths, np.ndarray)
        assert len(task.wavelengths) == 1

    def test_M_equals_len_angles(self):
        task = OptimizationTask(wavelength=0.03, angles=ANGLES_COARSE, functional=BACKSCATTER)
        assert task.M == len(ANGLES_COARSE)

    def test_invalid_wavelength_zero_raises(self):
        with pytest.raises(ValueError):
            OptimizationTask(wavelength=0.0, angles=ANGLES_COARSE, functional=BACKSCATTER)

    def test_invalid_wavelength_negative_raises(self):
        with pytest.raises(ValueError):
            OptimizationTask(wavelength=-0.01, angles=ANGLES_COARSE, functional=BACKSCATTER)

    def test_empty_angles_raises(self):
        with pytest.raises(ValueError):
            OptimizationTask(wavelength=0.03, angles=np.array([]), functional=BACKSCATTER)

    def test_repr_single(self):
        task = single_wavelength_task()
        r = repr(task)
        assert "0.03" in r
        assert str(len(ANGLES_COARSE)) in r

    def test_repr_broadband(self):
        task = OptimizationTask(
            wavelength=np.linspace(0.01, 0.05, 5),
            angles=ANGLES_COARSE,
            functional=BACKSCATTER,
        )
        assert "broadband" in repr(task)


# ===========================================================================
# SolverConfig
# ===========================================================================

class TestSolverConfig:

    def test_defaults(self):
        cfg = SolverConfig()
        assert cfg.n_best == 1
        assert cfg.aggregation == "mean"
        assert cfg.progress is True

    def test_n_best_zero_raises(self):
        with pytest.raises(ValueError):
            SolverConfig(n_best=0)

    def test_invalid_aggregation_string_raises(self):
        with pytest.raises(ValueError):
            SolverConfig(aggregation="geometric")

    def test_valid_aggregation_strings(self):
        for s in ("mean", "max", "sum"):
            assert SolverConfig(aggregation=s).aggregation == s

    def test_custom_aggregation_callable(self):
        cfg = SolverConfig(aggregation=lambda v: float(np.median(v)))
        assert cfg._aggregate(np.array([1.0, 3.0, 2.0])) == pytest.approx(2.0)

    def test_aggregate_mean(self):
        assert SolverConfig(aggregation="mean")._aggregate(np.array([1.0, 3.0])) == pytest.approx(2.0)

    def test_aggregate_max(self):
        assert SolverConfig(aggregation="max")._aggregate(np.array([1.0, 3.0])) == pytest.approx(3.0)

    def test_aggregate_sum(self):
        assert SolverConfig(aggregation="sum")._aggregate(np.array([1.0, 3.0])) == pytest.approx(4.0)


# ===========================================================================
# SolverResult
# ===========================================================================

class TestSolverResult:

    def _dummy(self) -> SolverResult:
        body = BodyParameters(
            eps=np.array([1.0, 2.25, 1.0], dtype=np.complex128),
            r=np.array([0.01, 0.015], dtype=np.float64),
            conducting_core=True,
        )
        return SolverResult(best=[(0.123, body)], n_evaluated=42, n_skipped=0, elapsed_seconds=1.5)

    def test_repr_contains_best_f(self):
        assert "0.123" in repr(self._dummy())

    def test_repr_contains_n_evaluated(self):
        assert "42" in repr(self._dummy())

    def test_repr_contains_elapsed(self):
        assert "1.50" in repr(self._dummy())

    def test_repr_no_results(self):
        result = SolverResult(best=[], n_evaluated=0, n_skipped=0, elapsed_seconds=0.0)
        assert "no results" in repr(result)


# ===========================================================================
# BruteForceSolver — construction
# ===========================================================================

class TestBruteForceSolverConstruction:

    def test_default_config(self):
        solver = BruteForceSolver()
        assert isinstance(solver.config, SolverConfig)
        assert solver.config.n_best == 1

    def test_custom_config_stored(self):
        cfg = SolverConfig(n_best=5, progress=False)
        solver = BruteForceSolver(cfg)
        assert solver.config is cfg

    def test_repr_contains_class_name(self):
        assert "BruteForceSolver" in repr(BruteForceSolver())


# ===========================================================================
# BruteForceSolver.run() — single wavelength
# ===========================================================================

class TestBruteForceSolverSingleWavelength:

    def test_returns_solver_result(self):
        result = quiet_solver().run(simple_space(2), single_wavelength_task())
        assert isinstance(result, SolverResult)

    def test_n_evaluated_matches_actual_count(self):
        space = simple_space(3)
        actual = sum(1 for _ in space)
        result = quiet_solver().run(space, single_wavelength_task())
        assert result.n_evaluated == actual

    def test_n_skipped_zero_on_clean_space(self):
        result = quiet_solver().run(simple_space(2), single_wavelength_task())
        assert result.n_skipped == 0

    def test_best_length_default(self):
        result = quiet_solver().run(simple_space(3), single_wavelength_task())
        assert len(result.best) == 1

    def test_best_sorted_ascending(self):
        result = quiet_solver(n_best=3).run(simple_space(5), single_wavelength_task())
        f_values = [f for f, _ in result.best]
        assert f_values == sorted(f_values)

    def test_best_params_type(self):
        result = quiet_solver().run(simple_space(2), single_wavelength_task())
        for _, body in result.best:
            assert isinstance(body, BodyParameters)

    def test_best_body_has_no_wavelength(self):
        """The result is a body; wavelength belongs to the task, not the body."""
        result = quiet_solver().run(simple_space(2), single_wavelength_task())
        for _, body in result.best:
            assert not hasattr(body, "wave_length")

    def test_best_body_reproduces_f(self):
        """The stored F can be recovered from the body via task.to_observation()."""
        from core.sphere_difraction import calculate_S
        task = single_wavelength_task()
        result = quiet_solver().run(simple_space(3), task)
        best_f, best_body = result.best[0]
        S_th, S_ph = calculate_S(best_body, task.to_observation())
        assert float(BACKSCATTER(S_th[0], S_ph[0], task.angles)) == pytest.approx(best_f, rel=1e-9)

    def test_f_value_finite(self):
        result = quiet_solver().run(simple_space(3), single_wavelength_task())
        for f, _ in result.best:
            assert np.isfinite(f)

    def test_elapsed_nonnegative(self):
        result = quiet_solver().run(simple_space(2), single_wavelength_task())
        assert result.elapsed_seconds >= 0.0

    def test_best_is_global_minimum(self):
        """Verify independently that the returned F is the true minimum."""
        from core.sphere_difraction import (calculate_S)
        space = simple_space(4)
        task = single_wavelength_task()
        obs = task.to_observation()
        result = quiet_solver().run(space, task)

        def eval_body(body):
            S_th, S_ph = calculate_S(body, obs)
            return float(BACKSCATTER(S_th[0], S_ph[0], task.angles))

        best_f_manual = min(eval_body(p) for p in space)
        assert result.best[0][0] == pytest.approx(best_f_manual, rel=1e-6)


# ===========================================================================
# BruteForceSolver.run() — n_best behaviour
# ===========================================================================

class TestBruteForceSolverNBest:

    def test_n_best_larger_than_candidates_returns_all(self):
        space = simple_space(2)
        actual = sum(1 for _ in space)
        result = quiet_solver(n_best=1000).run(space, single_wavelength_task())
        assert len(result.best) == actual

    def test_n_best_3_returns_3(self):
        result = quiet_solver(n_best=3).run(simple_space(5), single_wavelength_task())
        assert len(result.best) == 3

    def test_n_best_1_equals_global_min(self):
        space = simple_space(5)
        task = single_wavelength_task()
        r1 = quiet_solver(n_best=1).run(space, task)
        rn = quiet_solver(n_best=100).run(space, task)
        assert r1.best[0][0] == pytest.approx(rn.best[0][0], rel=1e-9)

    def test_n_best_results_unique(self):
        result = quiet_solver(n_best=3).run(simple_space(5), single_wavelength_task())
        r_outer = [p.r[-1] for _, p in result.best]
        assert len(set(r_outer)) == len(r_outer)


# ===========================================================================
# BruteForceSolver.run() — broadband
# ===========================================================================

class TestBruteForceSolverBroadband:

    def _bb_task(self, n=3):
        return OptimizationTask(
            wavelength=np.linspace(0.02, 0.04, n),
            angles=ANGLES_COARSE,
            functional=BACKSCATTER,
        )

    def test_broadband_returns_result(self):
        result = quiet_solver().run(simple_space(2), self._bb_task())
        assert isinstance(result, SolverResult)
        assert result.n_evaluated > 0

    def test_broadband_mean_aggregation(self):
        result = quiet_solver(aggregation="mean").run(simple_space(2), self._bb_task(4))
        for f, _ in result.best:
            assert np.isfinite(f)

    def test_broadband_max_aggregation(self):
        result = quiet_solver(aggregation="max").run(simple_space(2), self._bb_task(4))
        for f, _ in result.best:
            assert np.isfinite(f)

    def test_broadband_best_is_body(self):
        task = self._bb_task()
        result = quiet_solver().run(simple_space(2), task)
        for _, body in result.best:
            assert isinstance(body, BodyParameters)
            assert not hasattr(body, "wave_length")

    def test_broadband_vs_single_both_finite(self):
        space = simple_space(5)
        r_single = quiet_solver().run(space, single_wavelength_task())
        r_broad  = quiet_solver().run(space, self._bb_task(5))
        assert np.isfinite(r_single.best[0][0])
        assert np.isfinite(r_broad.best[0][0])


# ===========================================================================
# BruteForceSolver.run() — default config
# ===========================================================================

class TestBruteForceSolverDefaults:

    def test_default_config_runs(self):
        # BruteForceSolver() with no args should work (progress=True may print)
        result = BruteForceSolver(SolverConfig(progress=False)).run(
            simple_space(2), single_wavelength_task()
        )
        assert isinstance(result, SolverResult)


# ===========================================================================
# BruteForceSolver.run() — up_to mode
# ===========================================================================

class TestBruteForceSolverUpTo:

    def test_up_to_searches_shorter_stacks(self):
        space = SearchSpace(
            core_radius=0.01,
            layers=[Layer(0.002, "glass"), Layer(0.002, "foam")],
            materials=MATS,
            conducting_core=True,
            eps_outer=1.0 + 0j,
            up_to=True,
        )
        result = quiet_solver().run(space, single_wavelength_task())
        assert result.n_evaluated > 0


# ===========================================================================
# BruteForceSolver.run() — boundary validation
# ===========================================================================

class TestBruteForceSolverValidation:

    def test_continuous_range_raises_at_entry(self):
        space = SearchSpace(
            core_radius=0.01,
            layers=[Layer(thickness=ContinuousRange(0.001, 0.005), material=None)],
            materials=MATS,
            conducting_core=True,
            eps_outer=1.0 + 0j,
        )
        with pytest.raises(TypeError, match="Layer 0"):
            quiet_solver().run(space, single_wavelength_task())

    def test_empty_space_raises(self):
        mats = {"air": 1.0 + 0j}
        space = SearchSpace(
            core_radius=0.01,
            layers=[Layer(0.002, "air")],
            materials=mats,
            conducting_core=True,
            eps_outer=1.0 + 0j,
        )
        with pytest.raises(ValueError, match="no candidates"):
            quiet_solver().run(space, single_wavelength_task())

    def test_probe_failure_raises_runtime_error_with_cause(self):
        def boom(S_th, S_ph, angles):
            raise KeyError("functional bug")

        task = OptimizationTask(wavelength=0.03, angles=ANGLES_COARSE, functional=boom)
        with pytest.raises(RuntimeError) as exc:
            quiet_solver().run(simple_space(3), task)
        assert isinstance(exc.value.__cause__, KeyError)

    def test_nan_candidate_excluded_and_best_correct(self):
        from core.sphere_difraction import calculate_S

        space = simple_space(3)
        task = single_wavelength_task()
        obs = task.to_observation()
        clean = []
        for b in space:
            S_th, S_ph = calculate_S(b, obs)
            clean.append(BACKSCATTER(S_th[0], S_ph[0], task.angles))

        poison_idx = 1
        expected_best = min(f for i, f in enumerate(clean) if i != poison_idx)

        state = {"calls": 0}

        def nan_functional(S_th, S_ph, angles):
            idx = state["calls"]
            state["calls"] += 1
            if idx == poison_idx:
                return float("nan")
            return float(np.abs(S_th[0]) ** 2)

        task = OptimizationTask(wavelength=0.03, angles=ANGLES_COARSE, functional=nan_functional)
        result = quiet_solver().run(space, task)
        assert result.n_skipped == 1
        assert np.isfinite(result.best[0][0])
        assert result.best[0][0] == pytest.approx(expected_best, rel=1e-9)

    def test_inf_candidate_counted_in_skipped(self):
        state = {"calls": 0}

        def inf_functional(S_th, S_ph, angles):
            idx = state["calls"]
            state["calls"] += 1
            if idx == 1:
                return float("inf")
            return float(np.abs(S_th[0]) ** 2)

        task = OptimizationTask(wavelength=0.03, angles=ANGLES_COARSE, functional=inf_functional)
        result = quiet_solver().run(simple_space(2), task)
        assert result.n_skipped == 1