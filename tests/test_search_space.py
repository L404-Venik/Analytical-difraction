# test_search_space.py — Complete test suite for search_space.py
import pytest
import numpy as np
from analytical_diffraction.inverse_problem.search_space import (
    DiscreteRange,
    ContinuousRange,
    Layer,
    SearchSpace,
)
from analytical_diffraction.parameters import BodyParameters


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

MATS = {
    "a": 1.5 + 0j,
    "b": 2.5 + 0j,
    "c": 3.5 + 0j,
    "d": 4.5 + 0j,
}

# eps_outer default is 1.0 — make sure no material in MATS equals it
assert all(v != 1.0 + 0j for v in MATS.values())


def make_space(**kwargs) -> SearchSpace:
    """Convenience constructor with sensible defaults."""
    defaults = dict(
        core_radius=0.05,
        layers=[Layer(thickness=0.01, material="a")],
        materials=MATS,
        conducting_core=True,
    )
    defaults.update(kwargs)
    return SearchSpace(**defaults)


# ===========================================================================
# DiscreteRange
# ===========================================================================

class TestDiscreteRange:

    def test_basic_construction(self):
        dr = DiscreteRange(min=0.001, max=0.01, steps=5)
        assert dr.min == 0.001
        assert dr.max == 0.01
        assert dr.steps == 5

    def test_values_length(self):
        dr = DiscreteRange(0.001, 0.01, 10)
        vals = dr.values()
        assert len(vals) == 10

    def test_values_endpoints(self):
        dr = DiscreteRange(0.001, 0.01, 5)
        vals = dr.values()
        assert np.isclose(vals[0], 0.001)
        assert np.isclose(vals[-1], 0.01)

    def test_single_step(self):
        dr = DiscreteRange(0.005, 0.005, 1)
        vals = dr.values()
        assert len(vals) == 1
        assert np.isclose(vals[0], 0.005)

    def test_len(self):
        dr = DiscreteRange(0.001, 0.01, 7)
        assert len(dr) == 7

    def test_frozen(self):
        dr = DiscreteRange(0.001, 0.01, 5)
        with pytest.raises((AttributeError, TypeError)):
            dr.steps = 10  # type: ignore

    def test_min_zero_raises(self):
        with pytest.raises(ValueError, match="min must be > 0"):
            DiscreteRange(min=0.0, max=0.01, steps=5)

    def test_min_negative_raises(self):
        with pytest.raises(ValueError, match="min must be > 0"):
            DiscreteRange(min=-0.001, max=0.01, steps=5)

    def test_max_less_than_min_raises(self):
        with pytest.raises(ValueError, match="max"):
            DiscreteRange(min=0.01, max=0.001, steps=5)

    def test_steps_zero_raises(self):
        with pytest.raises(ValueError, match="steps must be >= 1"):
            DiscreteRange(min=0.001, max=0.01, steps=0)

    def test_steps_negative_raises(self):
        with pytest.raises(ValueError, match="steps must be >= 1"):
            DiscreteRange(min=0.001, max=0.01, steps=-3)

    def test_min_equals_max_valid(self):
        # min == max is allowed (single value, steps >= 1)
        dr = DiscreteRange(min=0.005, max=0.005, steps=3)
        vals = dr.values()
        assert np.all(np.isclose(vals, 0.005))

    def test_values_uniformly_spaced(self):
        dr = DiscreteRange(0.001, 0.010, 10)
        vals = dr.values()
        diffs = np.diff(vals)
        assert np.allclose(diffs, diffs[0])


# ===========================================================================
# ContinuousRange
# ===========================================================================

class TestContinuousRange:

    def test_basic_construction(self):
        cr = ContinuousRange(min=0.001, max=0.01)
        assert cr.min == 0.001
        assert cr.max == 0.01

    def test_frozen(self):
        cr = ContinuousRange(0.001, 0.01)
        with pytest.raises((AttributeError, TypeError)):
            cr.min = 0.002  # type: ignore

    def test_min_zero_raises(self):
        with pytest.raises(ValueError, match="min must be > 0"):
            ContinuousRange(min=0.0, max=0.01)

    def test_min_negative_raises(self):
        with pytest.raises(ValueError, match="min must be > 0"):
            ContinuousRange(min=-0.001, max=0.01)

    def test_max_less_than_min_raises(self):
        with pytest.raises(ValueError, match="max"):
            ContinuousRange(min=0.01, max=0.001)

    def test_min_equals_max_valid(self):
        cr = ContinuousRange(min=0.005, max=0.005)
        assert cr.min == cr.max


# ===========================================================================
# Layer
# ===========================================================================

class TestLayer:

    def test_fixed_thickness_values(self):
        layer = Layer(thickness=0.005, material="a")
        assert layer._thickness_values() == [0.005]

    def test_discrete_range_thickness_values(self):
        dr = DiscreteRange(0.001, 0.005, 3)
        layer = Layer(thickness=dr, material="a")
        vals = layer._thickness_values()
        assert len(vals) == 3
        assert np.isclose(vals[0], 0.001)
        assert np.isclose(vals[-1], 0.005)

    def test_continuous_range_thickness_raises(self):
        cr = ContinuousRange(0.001, 0.01)
        layer = Layer(thickness=cr, material="a")
        with pytest.raises(TypeError, match="ContinuousRange cannot be iterated"):
            layer._thickness_values()

    def test_material_str(self):
        layer = Layer(thickness=0.005, material="a")
        assert layer._material_names(MATS) == ["a"]

    def test_material_list(self):
        layer = Layer(thickness=0.005, material=["a", "b"])
        assert layer._material_names(MATS) == ["a", "b"]

    def test_material_none_returns_all(self):
        layer = Layer(thickness=0.005, material=None)
        names = layer._material_names(MATS)
        assert set(names) == set(MATS.keys())

    def test_n_thickness_options_fixed(self):
        layer = Layer(thickness=0.005, material="a")
        assert layer.n_thickness_options() == 1

    def test_n_thickness_options_discrete(self):
        layer = Layer(thickness=DiscreteRange(0.001, 0.01, 7), material="a")
        assert layer.n_thickness_options() == 7

    def test_n_thickness_options_continuous(self):
        layer = Layer(thickness=ContinuousRange(0.001, 0.01), material="a")
        assert layer.n_thickness_options() == 1  # treated as 1 for size estimate

    def test_n_material_options_str(self):
        layer = Layer(thickness=0.005, material="a")
        assert layer.n_material_options(MATS) == 1

    def test_n_material_options_list(self):
        layer = Layer(thickness=0.005, material=["a", "b", "c"])
        assert layer.n_material_options(MATS) == 3

    def test_n_material_options_none(self):
        layer = Layer(thickness=0.005, material=None)
        assert layer.n_material_options(MATS) == len(MATS)


# ===========================================================================
# Layer — construction validation
# ===========================================================================

class TestLayerValidation:

    def test_int_thickness_coerced_to_float(self):
        layer = Layer(thickness=5, material="a")
        assert isinstance(layer.thickness, float)
        assert layer.thickness == 5.0

    def test_int_thickness_space_runs(self):
        space = make_space(layers=[Layer(thickness=5, material="a")])
        assert len(list(space)) == 1

    def test_bool_thickness_raises(self):
        with pytest.raises(TypeError):
            Layer(thickness=True, material="a")

    def test_negative_thickness_raises(self):
        with pytest.raises(ValueError):
            Layer(thickness=-1.0, material="a")

    def test_zero_thickness_raises(self):
        with pytest.raises(ValueError):
            Layer(thickness=0.0, material="a")

    def test_empty_material_list_raises(self):
        with pytest.raises(ValueError):
            Layer(thickness=0.01, material=[])

    def test_non_str_material_element_raises(self):
        with pytest.raises(TypeError):
            Layer(thickness=0.01, material=["a", 1])

    def test_non_str_non_list_material_raises(self):
        with pytest.raises(TypeError):
            Layer(thickness=0.01, material=123)  # type: ignore

    def test_duplicate_materials_deduped_with_warning(self):
        with pytest.warns(UserWarning):
            layer = Layer(thickness=0.01, material=["a", "a", "b"])
        assert layer.material == ["a", "b"]


# ===========================================================================
# SearchSpace — validate_discrete
# ===========================================================================

class TestSearchSpaceValidateDiscrete:

    def test_continuous_range_first_layer_raises(self):
        space = make_space(
            layers=[Layer(thickness=ContinuousRange(0.001, 0.01), material="a")],
        )
        with pytest.raises(TypeError, match="Layer 0"):
            space.validate_discrete()

    def test_continuous_range_second_layer_names_index(self):
        space = make_space(
            layers=[
                Layer(0.01, "a"),
                Layer(thickness=ContinuousRange(0.001, 0.01), material="b"),
            ],
        )
        with pytest.raises(TypeError, match="Layer 1"):
            space.validate_discrete()

    def test_all_discrete_passes(self):
        make_space().validate_discrete()


# ===========================================================================
# SearchSpace — construction and validation
# ===========================================================================

class TestSearchSpaceConstruction:

    def test_basic_construction(self):
        space = make_space()
        assert space.core_radius == 0.05
        assert space.conducting_core is True

    def test_core_radius_zero_raises(self):
        with pytest.raises(ValueError, match="core_radius must be > 0"):
            make_space(core_radius=0.0)

    def test_core_radius_negative_raises(self):
        with pytest.raises(ValueError, match="core_radius must be > 0"):
            make_space(core_radius=-0.01)

    def test_empty_layers_raises(self):
        with pytest.raises(ValueError, match="layers must not be empty"):
            make_space(layers=[])

    def test_missing_materials_raises(self):
        with pytest.raises(TypeError):
            SearchSpace(core_radius=0.05, layers=[Layer(0.005, "a")])  # type: ignore

    def test_unknown_material_str_raises(self):
        with pytest.raises(ValueError, match="unobtainium"):
            make_space(layers=[Layer(thickness=0.005, material="unobtainium")])

    def test_unknown_material_in_list_raises(self):
        with pytest.raises(ValueError, match="unobtainium"):
            make_space(layers=[Layer(thickness=0.005, material=["a", "unobtainium"])])

    def test_max_total_thickness_zero_raises(self):
        with pytest.raises(ValueError, match="max_total_thickness must be > 0"):
            make_space(max_total_thickness=0.0)

    def test_max_total_thickness_negative_raises(self):
        with pytest.raises(ValueError, match="max_total_thickness must be > 0"):
            make_space(max_total_thickness=-0.01)

    def test_materials_dict_is_copied(self):
        mats = dict(MATS)
        space = make_space(materials=mats)
        mats["z"] = 99.0 + 0j
        assert "z" not in space.materials

    def test_dielectric_core_stored(self):
        space = make_space(conducting_core=False, core_material=2.0 + 0j)
        assert space.core_material == 2.0 + 0j

    def test_eps_outer_stored(self):
        space = make_space(eps_outer=1.5 + 0j)
        assert space.eps_outer == 1.5 + 0j


# ===========================================================================
# SearchSpace — BodyParameters structure
# ===========================================================================

class TestSearchSpaceCandidateStructure:

    def test_r_length(self):
        space = make_space(layers=[
            Layer(0.01, "a"),
            Layer(0.01, "b"),
        ])
        for p in space:
            # r has core + one boundary per layer = n_layers + 1... 
            # wait: r has core_radius + one entry per layer boundary
            assert len(p.r) == 3  # core, after layer1, after layer2

    def test_eps_length(self):
        space = make_space(layers=[
            Layer(0.01, "a"),
            Layer(0.01, "b"),
        ])
        for p in space:
            assert len(p.eps) == len(p.r) + 1

    def test_r_monotonically_increasing(self):
        space = make_space(layers=[
            Layer(DiscreteRange(0.001, 0.01, 4), material=None),
            Layer(DiscreteRange(0.001, 0.01, 4), material=None),
        ])
        for p in space:
            assert np.all(np.diff(p.r) > 0), f"r not increasing: {p.r}"

    def test_r_starts_at_core_radius(self):
        space = make_space(core_radius=0.07)
        for p in space:
            assert np.isclose(p.r[0], 0.07)

    def test_eps_ends_with_outer(self):
        space = make_space(eps_outer=1.3 + 0j)
        for p in space:
            assert p.eps[-1] == 1.3 + 0j

    def test_conducting_core_flag_propagated(self):
        space = make_space(conducting_core=True)
        for p in space:
            assert p.conducting_core is True

    def test_dielectric_core_eps(self):
        core_eps = 2.0 + 0j
        space = make_space(
            conducting_core=False,
            core_material=core_eps,
            layers=[Layer(0.01, "b")],  # "b" = 2.5, different from core
        )
        for p in space:
            assert p.eps[0] == core_eps

    def test_body_has_no_wavelength(self):
        """Bodies are wavelength-free — the observation lives in the task."""
        space = make_space()
        for p in space:
            assert not hasattr(p, "wave_length")

    def test_body_parameters_type(self):
        space = make_space()
        for p in space:
            assert isinstance(p, BodyParameters)


# ===========================================================================
# SearchSpace — filtering rules
# ===========================================================================

class TestSearchSpaceFiltering:

    def test_no_consecutive_same_material(self):
        space = make_space(
            layers=[
                Layer(0.01, material=None),
                Layer(0.01, material=None),
            ],
        )
        for p in space:
            for i in range(1, len(p.eps) - 1):   # interior eps only (skip core and outer)
                assert p.eps[i] != p.eps[i + 1], (
                    f"consecutive same eps at position {i}: {p.eps}"
                )

    def test_first_layer_differs_from_dielectric_core(self):
        core_eps = MATS["a"]   # 1.5+0j
        space = SearchSpace(
            core_radius=0.05,
            layers=[Layer(0.01, material=None)],
            materials=MATS,
            conducting_core=False,
            core_material=core_eps,
        )
        for p in space:
            assert p.eps[1] != core_eps, f"First layer same as core: {p.eps}"

    def test_last_layer_differs_from_outer(self):
        outer = 1.0 + 0j
        # Add a material that equals outer to MATS to make the test meaningful
        mats_with_air = dict(MATS, air=1.0 + 0j)
        space = SearchSpace(
            core_radius=0.05,
            layers=[Layer(0.01, material=None)],
            materials=mats_with_air,
            conducting_core=True,
            eps_outer=outer,
        )
        for p in space:
            assert p.eps[-2] != outer, f"Last layer same as outer: {p.eps}"

    def test_max_total_thickness_respected(self):
        limit = 0.005
        space = make_space(
            layers=[Layer(DiscreteRange(0.001, 0.010, 10), material="a")],
            max_total_thickness=limit,
        )
        for p in space:
            total = p.r[-1] - p.r[0]
            assert total <= limit + 1e-12, f"thickness {total} exceeds limit {limit}"

    def test_max_total_thickness_two_layers(self):
        limit = 0.008
        space = make_space(
            layers=[
                Layer(DiscreteRange(0.001, 0.005, 5), material="a"),
                Layer(DiscreteRange(0.001, 0.005, 5), material="b"),
            ],
            max_total_thickness=limit,
        )
        for p in space:
            total = p.r[-1] - p.r[0]
            assert total <= limit + 1e-12

    def test_conducting_core_no_core_eps_check(self):
        # With conducting_core=True, first layer may have any eps (no core check)
        # Just confirm we get candidates at all
        space = make_space(
            conducting_core=True,
            layers=[Layer(0.01, material="a")],
        )
        candidates = list(space)
        assert len(candidates) > 0

    def test_two_materials_same_eps_treated_as_same(self):
        # Two names with identical permittivity — should be filtered as "same material"
        mats = {"x": 2.5 + 0j, "y": 2.5 + 0j, "z": 3.5 + 0j}
        space = SearchSpace(
            core_radius=0.05,
            layers=[
                Layer(0.01, material=["x", "y", "z"]),
                Layer(0.01, material=["x", "y", "z"]),
            ],
            materials=mats,
            conducting_core=True,
        )
        for p in space:
            for i in range(1, len(p.eps) - 1):
                assert p.eps[i] != p.eps[i + 1]


# ===========================================================================
# SearchSpace — up_to mode
# ===========================================================================

class TestSearchSpaceUpTo:

    def test_up_to_false_yields_only_n_layers(self):
        space = make_space(
            layers=[
                Layer(0.01, "a"),
                Layer(0.01, "b"),
            ],
            up_to=False,
        )
        for p in space:
            assert len(p.r) == 3  # core + 2 layers

    def test_up_to_true_yields_shorter_stacks(self):
        space = make_space(
            layers=[
                Layer(0.01, "a"),
                Layer(0.01, "b"),
            ],
            up_to=True,
        )
        layer_counts = {len(p.r) for p in space}
        assert 2 in layer_counts  # 1-layer stack: r has 2 entries
        assert 3 in layer_counts  # 2-layer stack: r has 3 entries

    def test_up_to_true_single_layer_spec(self):
        # With only one layer spec, up_to=True is equivalent to up_to=False
        space_up = make_space(layers=[Layer(0.01, "a")], up_to=True)
        space_ex = make_space(layers=[Layer(0.01, "a")], up_to=False)
        assert list(p.r.tolist() for p in space_up) == list(p.r.tolist() for p in space_ex)

    def test_up_to_true_three_layers_all_counts_present(self):
        space = SearchSpace(
            core_radius=0.05,
            layers=[
                Layer(0.01, "a"),
                Layer(0.01, "b"),
                Layer(0.01, "c"),
            ],
            materials=MATS,
            conducting_core=True,
            up_to=True,
        )
        layer_counts = {len(p.r) for p in space}
        assert layer_counts == {2, 3, 4}  # 1-, 2-, 3-layer stacks


# ===========================================================================
# SearchSpace — size estimate
# ===========================================================================

class TestSearchSpaceSizeEstimate:

    def test_size_estimate_is_upper_bound(self):
        space = make_space(
            layers=[
                Layer(DiscreteRange(0.001, 0.01, 5), material=None),
                Layer(DiscreteRange(0.001, 0.01, 5), material=None),
            ],
        )
        actual = len(list(space))
        assert space.size_estimate() >= actual

    def test_size_estimate_fixed_layer(self):
        # One fixed layer, one material: exactly 1 combination before filtering
        space = make_space(layers=[Layer(0.01, "a")])
        assert space.size_estimate() == 1

    def test_size_estimate_discrete_single_material(self):
        space = make_space(layers=[Layer(DiscreteRange(0.001, 0.01, 7), "a")])
        assert space.size_estimate() == 7

    def test_size_estimate_up_to_adds_stacks(self):
        space_up = make_space(
            layers=[Layer(0.01, "a"), Layer(0.01, "b")],
            up_to=True,
        )
        space_ex = make_space(
            layers=[Layer(0.01, "a"), Layer(0.01, "b")],
            up_to=False,
        )
        # up_to includes 1-layer estimate too
        assert space_up.size_estimate() >= space_ex.size_estimate()


# ===========================================================================
# SearchSpace — repr
# ===========================================================================

class TestSearchSpaceRepr:

    def test_repr_contains_core_radius(self):
        space = make_space(core_radius=0.05)
        assert "0.05" in repr(space)

    def test_repr_contains_n_materials(self):
        space = make_space()
        assert str(len(MATS)) in repr(space)

    def test_repr_contains_up_to(self):
        space = make_space(up_to=True)
        assert "up to" in repr(space)

    def test_repr_contains_exactly(self):
        space = make_space(up_to=False)
        assert "exactly" in repr(space)

    def test_repr_contains_max_thickness(self):
        space = make_space(max_total_thickness=0.05)
        assert "max_total_thickness" in repr(space)

    def test_repr_no_max_thickness_when_none(self):
        space = make_space(max_total_thickness=None)
        assert "max_total_thickness" not in repr(space)


# ===========================================================================
# SearchSpace — iteration correctness (count-based checks)
# ===========================================================================

class TestSearchSpaceIterationCounts:

    def test_single_fixed_layer_conducting(self):
        # 1 thickness, 1 material, conducting core → 1 candidate
        # (only filtered by last-layer-vs-outer)
        mats = {"a": 2.5 + 0j}
        space = SearchSpace(
            core_radius=0.05,
            layers=[Layer(0.01, "a")],
            materials=mats,
            conducting_core=True,
            eps_outer=1.0 + 0j,
        )
        assert len(list(space)) == 1

    def test_single_fixed_layer_outer_match_filtered(self):
        # material eps == eps_outer → all filtered out
        mats = {"air": 1.0 + 0j}
        space = SearchSpace(
            core_radius=0.05,
            layers=[Layer(0.01, "air")],
            materials=mats,
            conducting_core=True,
            eps_outer=1.0 + 0j,
        )
        assert len(list(space)) == 0

    def test_two_fixed_layers_same_material_filtered(self):
        mats = {"a": 2.5 + 0j}
        space = SearchSpace(
            core_radius=0.05,
            layers=[Layer(0.01, "a"), Layer(0.01, "a")],
            materials=mats,
            conducting_core=True,
            eps_outer=1.0 + 0j,
        )
        assert len(list(space)) == 0

    def test_two_layers_two_materials_count(self):
        # Layer1 = {a, b}, Layer2 = {a, b}, fixed thickness
        # Valid combos: a-b and b-a = 2
        mats = {"a": 2.5 + 0j, "b": 3.5 + 0j}
        space = SearchSpace(
            core_radius=0.05,
            layers=[
                Layer(0.01, ["a", "b"]),
                Layer(0.01, ["a", "b"]),
            ],
            materials=mats,
            conducting_core=True,
            eps_outer=1.0 + 0j,
        )
        assert len(list(space)) == 2

    def test_discrete_range_all_candidates_unique(self):
        space = make_space(
            layers=[Layer(DiscreteRange(0.001, 0.01, 10), material="a")],
        )
        candidates = list(space)
        r_outer = [p.r[-1] for p in candidates]
        assert len(r_outer) == len(set(r_outer)), "Duplicate r values found"

    def test_iteration_reproducible(self):
        space = make_space(
            layers=[
                Layer(DiscreteRange(0.001, 0.005, 3), material=None),
                Layer(DiscreteRange(0.001, 0.005, 3), material=None),
            ],
        )
        run1 = [(p.r.tolist(), p.eps.tolist()) for p in space]
        run2 = [(p.r.tolist(), p.eps.tolist()) for p in space]
        assert run1 == run2

    def test_iter_equals_iter_candidates(self):
        space = make_space(
            layers=[Layer(DiscreteRange(0.001, 0.01, 4), material=None)],
        )
        via_iter = list(space)
        via_method = list(space.iter_candidates())
        assert len(via_iter) == len(via_method)
        for a, b in zip(via_iter, via_method):
            assert np.allclose(a.r, b.r)
            assert np.allclose(a.eps, b.eps)