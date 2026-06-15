# test_materials.py — Test suite for the material library helpers and example CSV.

from pathlib import Path

import numpy as np
import pytest

from sphere_diffraction.materials import lossy_eps, load_materials
from sphere_diffraction.inverse_problem.search_space import SearchSpace, Layer, DiscreteRange

EXAMPLE_CSV = Path(__file__).resolve().parent.parent / "examples" / "materials.csv"


# ===========================================================================
# Material convention guard
#
# Pins the +i lossy permittivity convention, eps = eps_r * (1 + i*tanδ), which
# snapshot_configs.py relies on (the lossy snapshot 2.56 + 0.102j). These tests
# must fail if that convention changes.
# ===========================================================================

class TestMaterialConvention:

    def test_lossless_has_zero_imaginary(self):
        assert lossy_eps(2.5) == 2.5 + 0j

    def test_real_part_is_eps_r(self):
        assert lossy_eps(4.0, 0.01).real == pytest.approx(4.0)

    def test_imaginary_part_is_eps_r_times_tan_delta(self):
        assert lossy_eps(4.0, 0.01).imag == pytest.approx(0.04)

    def test_imaginary_sign_is_positive(self):
        assert lossy_eps(2.56, 0.04).imag > 0

    def test_matches_snapshot_lossy_value(self):
        """lossy_eps must reproduce the +i lossy permittivity used in snapshots."""
        # snapshot_configs.py uses 2.56 + 0.102j  ->  tanδ = 0.102 / 2.56
        assert lossy_eps(2.56, 0.102 / 2.56) == pytest.approx(2.56 + 0.102j)

    def test_loader_applies_same_convention(self, tmp_path):
        csv = tmp_path / "m.csv"
        csv.write_text("name,eps_r,loss_tangent\nx,4.0,0.01\n", encoding="utf-8")
        assert load_materials(csv)["x"] == lossy_eps(4.0, 0.01)


# ===========================================================================
# load_materials — happy path on the shipped example
# ===========================================================================

class TestLoadExampleFile:

    def test_example_file_exists(self):
        assert EXAMPLE_CSV.is_file()

    def test_loads_more_than_ten_materials(self):
        assert len(load_materials(EXAMPLE_CSV)) > 10

    def test_all_values_complex(self):
        for v in load_materials(EXAMPLE_CSV).values():
            assert isinstance(v, complex)

    def test_all_real_parts_at_least_one(self):
        for v in load_materials(EXAMPLE_CSV).values():
            assert v.real >= 1.0

    def test_all_low_loss_nonnegative(self):
        for v in load_materials(EXAMPLE_CSV).values():
            assert 0.0 <= v.imag < v.real

    def test_usable_as_search_space_library(self):
        materials = load_materials(EXAMPLE_CSV)
        space = SearchSpace(
            core_radius=0.01,
            layers=[Layer(thickness=DiscreteRange(0.001, 0.005, 3), material=None)],
            materials=materials,
            conducting_core=True,
        )
        candidates = list(space)
        assert len(candidates) > 0
        for body in candidates:
            assert np.all(np.isfinite(body.eps))


# ===========================================================================
# load_materials — parsing and validation
# ===========================================================================

class TestLoadMaterialsParsing:

    def _write(self, tmp_path, text):
        p = tmp_path / "m.csv"
        p.write_text(text, encoding="utf-8")
        return p

    def test_ignores_extra_columns(self, tmp_path):
        p = self._write(
            tmp_path,
            "name,eps_r,loss_tangent,source,valid_band\n"
            "ptfe,2.10,0.0002,von Hippel,0.1-10 GHz\n",
        )
        mats = load_materials(p)
        assert mats == {"ptfe": lossy_eps(2.10, 0.0002)}

    def test_missing_required_column_raises(self, tmp_path):
        p = self._write(tmp_path, "name,eps_r\nptfe,2.1\n")
        with pytest.raises(ValueError, match="loss_tangent"):
            load_materials(p)

    def test_duplicate_name_raises(self, tmp_path):
        p = self._write(
            tmp_path,
            "name,eps_r,loss_tangent\nptfe,2.1,0.0\nptfe,2.2,0.0\n",
        )
        with pytest.raises(ValueError, match="duplicate"):
            load_materials(p)

    def test_empty_name_raises(self, tmp_path):
        p = self._write(tmp_path, "name,eps_r,loss_tangent\n,2.1,0.0\n")
        with pytest.raises(ValueError, match="empty"):
            load_materials(p)

    def test_non_numeric_value_raises(self, tmp_path):
        p = self._write(tmp_path, "name,eps_r,loss_tangent\nx,abc,0.0\n")
        with pytest.raises(ValueError, match="must be numbers"):
            load_materials(p)

    def test_eps_r_below_one_raises(self, tmp_path):
        p = self._write(tmp_path, "name,eps_r,loss_tangent\nx,0.5,0.0\n")
        with pytest.raises(ValueError, match="eps_r must be >= 1"):
            load_materials(p)

    def test_negative_loss_tangent_raises(self, tmp_path):
        p = self._write(tmp_path, "name,eps_r,loss_tangent\nx,2.0,-0.01\n")
        with pytest.raises(ValueError, match="loss_tangent must be >= 0"):
            load_materials(p)

    def test_no_data_rows_raises(self, tmp_path):
        p = self._write(tmp_path, "name,eps_r,loss_tangent\n")
        with pytest.raises(ValueError, match="no material rows"):
            load_materials(p)

    def test_error_reports_line_number(self, tmp_path):
        p = self._write(
            tmp_path,
            "name,eps_r,loss_tangent\nok,2.1,0.0\nbad,0.5,0.0\n",
        )
        with pytest.raises(ValueError, match=":3:"):
            load_materials(p)
