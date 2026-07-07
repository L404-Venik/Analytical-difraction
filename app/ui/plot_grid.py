from __future__ import annotations

from typing import List, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFrame, QHBoxLayout,
    QLabel, QPushButton, QSplitter, QVBoxLayout, QWidget,
)

from app.application.computation import ComputationResult

from .plots import (
    POLARIZATION_BOTH, POLARIZATIONS, SCALE_LINEAR, SCALES,
    PolarPatternCanvas, RcsAngleCanvas,
)
from .ui_config import UIConfig

PLOT_TYPES = {
    "Scattering pattern": PolarPatternCanvas,
    "RCS vs angle": RcsAngleCanvas,
}
MAX_CELLS = 6
Y_RANGE_MIN_GAP = 1.0


class PlotCell(QFrame):
    """One grid cell: a header with view options above a result canvas."""

    remove_requested = pyqtSignal(object)

    def __init__(
        self,
        config: Optional[UIConfig] = None,
        plot_type: str = "Scattering pattern",
        polarization: str = POLARIZATION_BOTH,
        scale: str = SCALE_LINEAR,
        y_min: float = -30.0,
        y_max: float = 30.0,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        if plot_type not in PLOT_TYPES:
            plot_type = next(iter(PLOT_TYPES))
        self._cfg = config or UIConfig()
        self._result: ComputationResult | None = None
        self._canvas = None

        self.setFrameShape(QFrame.Shape.StyledPanel)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(*([self._cfg.px(6)] * 4))
        layout.setSpacing(self._cfg.px(4))

        header = QHBoxLayout()
        header.setSpacing(self._cfg.px(6))

        self.type_combo = QComboBox()
        self.type_combo.addItems(list(PLOT_TYPES))
        self.type_combo.setCurrentText(plot_type)
        self.type_combo.currentTextChanged.connect(self._on_type_changed)
        header.addWidget(self.type_combo)

        self.polarization_combo = QComboBox()
        self.polarization_combo.addItems(POLARIZATIONS)
        self.polarization_combo.setCurrentText(
            polarization if polarization in POLARIZATIONS else POLARIZATION_BOTH
        )
        self.polarization_combo.currentTextChanged.connect(self._on_polarization_changed)
        header.addWidget(self.polarization_combo)

        self.scale_combo = QComboBox()
        self.scale_combo.addItems(SCALES)
        self.scale_combo.setCurrentText(scale if scale in SCALES else SCALE_LINEAR)
        self.scale_combo.currentTextChanged.connect(self._on_scale_changed)
        header.addWidget(self.scale_combo)

        self._y_range_label = QLabel("dB range:")
        header.addWidget(self._y_range_label)

        if y_max - y_min < Y_RANGE_MIN_GAP:
            y_min, y_max = -30.0, 30.0
        self.y_min_spin = self._make_range_spin(y_min)
        self.y_max_spin = self._make_range_spin(y_max)
        self._sync_y_range_bounds()
        header.addWidget(self.y_min_spin)
        header.addWidget(self.y_max_spin)

        header.addStretch()

        self.remove_button = QPushButton("✕")
        btn_size = self._cfg.px(24)
        self.remove_button.setFixedSize(btn_size, btn_size)
        self.remove_button.clicked.connect(lambda: self.remove_requested.emit(self))
        header.addWidget(self.remove_button)

        layout.addLayout(header)

        self._canvas_slot = QVBoxLayout()
        layout.addLayout(self._canvas_slot, stretch=1)

        self._restyle()
        self._rebuild_canvas()

    def _make_range_spin(self, value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(-200.0, 200.0)
        spin.setDecimals(0)
        spin.setLocale(self._cfg.locale())
        spin.setSingleStep(10.0)
        spin.setValue(value)
        spin.valueChanged.connect(self._on_y_range_changed)
        return spin

    # ------------------------------------------------------------------ #

    @property
    def plot_type(self) -> str:
        return self.type_combo.currentText()

    def spec(self) -> dict:
        return {
            "type": self.plot_type,
            "polarization": self.polarization_combo.currentText(),
            "scale": self.scale_combo.currentText(),
            "y_min": self.y_min_spin.value(),
            "y_max": self.y_max_spin.value(),
        }

    def show_result(self, result: ComputationResult) -> None:
        self._result = result
        self._canvas.show_result(result)

    def apply_config(self, config: UIConfig) -> None:
        self._cfg = config
        self._restyle()
        self.y_min_spin.setLocale(config.locale())
        self.y_max_spin.setLocale(config.locale())
        self._canvas.apply_config(config)

    # ------------------------------------------------------------------ #

    def _rebuild_canvas(self) -> None:
        if self._canvas is not None:
            self._canvas_slot.removeWidget(self._canvas)
            self._canvas.deleteLater()
        self._canvas = PLOT_TYPES[self.plot_type](self._cfg)
        self._canvas.set_polarization(self.polarization_combo.currentText())
        self._push_options()
        self._canvas_slot.addWidget(self._canvas)
        self._update_options_visibility()
        if self._result is not None:
            self._canvas.show_result(self._result)

    def _update_options_visibility(self) -> None:
        is_rcs = isinstance(self._canvas, RcsAngleCanvas)
        is_pattern = isinstance(self._canvas, PolarPatternCanvas)
        for widget in (self._y_range_label, self.y_min_spin, self.y_max_spin):
            widget.setVisible(is_rcs)
        self.scale_combo.setVisible(is_pattern)

    def _push_options(self) -> None:
        if isinstance(self._canvas, RcsAngleCanvas):
            self._canvas.set_y_range(self.y_min_spin.value(), self.y_max_spin.value())
        if isinstance(self._canvas, PolarPatternCanvas):
            self._canvas.set_scale(self.scale_combo.currentText())

    def _sync_y_range_bounds(self) -> None:
        self.y_min_spin.setMaximum(self.y_max_spin.value() - Y_RANGE_MIN_GAP)
        self.y_max_spin.setMinimum(self.y_min_spin.value() + Y_RANGE_MIN_GAP)

    def _on_type_changed(self, _text: str) -> None:
        self._rebuild_canvas()

    def _on_polarization_changed(self, text: str) -> None:
        self._canvas.set_polarization(text)

    def _on_scale_changed(self, _text: str) -> None:
        self._push_options()

    def _on_y_range_changed(self, _value: float) -> None:
        self._sync_y_range_bounds()
        self._push_options()

    def _restyle(self) -> None:
        c = self._cfg.theme
        cfg = self._cfg
        self.setStyleSheet(
            f"""
            PlotCell {{
                background-color: {c.card_bg};
                border: 1px solid {c.border};
                border-radius: {cfg.px(4)}px;
            }}
            """
        )
        self.setFont(cfg.font)
        combo_style = cfg._combo_style(c)
        self.type_combo.setStyleSheet(combo_style)
        self.polarization_combo.setStyleSheet(combo_style)
        self.scale_combo.setStyleSheet(combo_style)
        spin_style = cfg._spinbox_style(c)
        self.y_min_spin.setStyleSheet(spin_style)
        self.y_max_spin.setStyleSheet(spin_style)
        self._y_range_label.setFont(cfg.font)
        self._y_range_label.setStyleSheet(
            f"color: {c.text_primary}; background: transparent; padding: 0 {cfg.px(2)}px;"
        )
        self.remove_button.setStyleSheet(cfg._delete_btn_style(c))


DEFAULT_LAYOUT = [
    {"type": "Scattering pattern", "polarization": POLARIZATION_BOTH},
    {"type": "RCS vs angle", "polarization": POLARIZATION_BOTH},
]


class PlotGrid(QWidget):
    """Resizable grid of plot cells; every cell renders the same result.

    Cells live in a vertical splitter of rows; a row with two cells is itself
    a horizontal splitter, so every boundary between plots can be dragged.
    """

    def __init__(self, config: Optional[UIConfig] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._cfg = config or UIConfig()
        self._result: ComputationResult | None = None
        self.cells: List[PlotCell] = []
        self._splitter: QSplitter | None = None

        root = self._root = QVBoxLayout(self)
        root.setContentsMargins(*([self._cfg.px(6)] * 4))
        root.setSpacing(self._cfg.px(6))

        toolbar = QHBoxLayout()
        toolbar.addStretch()
        self.add_button = QPushButton("+ Add Plot")
        self.add_button.setFixedHeight(self._cfg.px(28))
        self.add_button.clicked.connect(lambda: self.add_cell())
        toolbar.addWidget(self.add_button)
        root.addLayout(toolbar)

        self._restyle()
        self.restore_layout(DEFAULT_LAYOUT)

    # ------------------------------------------------------------------ #

    def add_cell(self, spec: Optional[dict] = None) -> None:
        if len(self.cells) >= MAX_CELLS:
            return
        spec = spec or {}
        cell = PlotCell(
            config=self._cfg,
            plot_type=spec.get("type", "Scattering pattern"),
            polarization=spec.get("polarization", POLARIZATION_BOTH),
            scale=spec.get("scale", SCALE_LINEAR),
            y_min=float(spec.get("y_min", -30.0)),
            y_max=float(spec.get("y_max", 30.0)),
        )
        cell.remove_requested.connect(self._remove_cell)
        self.cells.append(cell)
        if self._result is not None:
            cell.show_result(self._result)
        self._reflow()

    def restore_layout(self, specs: list[dict]) -> None:
        for cell in list(self.cells):
            self.cells.remove(cell)
            cell.deleteLater()
        if not specs:
            specs = DEFAULT_LAYOUT
        for spec in specs[:MAX_CELLS]:
            self.add_cell(spec)

    def layout_spec(self) -> list[dict]:
        return [cell.spec() for cell in self.cells]

    def show_result(self, result: ComputationResult) -> None:
        self._result = result
        for cell in self.cells:
            cell.show_result(result)

    def apply_config(self, config: UIConfig) -> None:
        self._cfg = config
        self._restyle()
        for cell in self.cells:
            cell.apply_config(config)

    # ------------------------------------------------------------------ #

    def _remove_cell(self, cell: PlotCell) -> None:
        if len(self.cells) <= 1:
            return
        self.cells.remove(cell)
        cell.deleteLater()
        self._reflow()

    def _reflow(self) -> None:
        old = self._splitter
        n_cols = 1 if len(self.cells) <= 2 else 2

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setHandleWidth(6)
        splitter.setChildrenCollapsible(False)
        for start in range(0, len(self.cells), n_cols):
            row_cells = self.cells[start : start + n_cols]
            if len(row_cells) == 1:
                splitter.addWidget(row_cells[0])
            else:
                row = QSplitter(Qt.Orientation.Horizontal)
                row.setHandleWidth(6)
                row.setChildrenCollapsible(False)
                for cell in row_cells:
                    row.addWidget(cell)
                splitter.addWidget(row)

        self._root.addWidget(splitter, stretch=1)
        self._splitter = splitter
        if old is not None:
            old.setParent(None)
            old.deleteLater()

        self.add_button.setEnabled(len(self.cells) < MAX_CELLS)

    def _restyle(self) -> None:
        c = self._cfg.theme
        self.setStyleSheet(
            f"""
            PlotGrid {{ background-color: {c.window_bg}; }}
            QSplitter::handle {{ background-color: {c.border}; }}
            QSplitter::handle:hover {{ background-color: {c.neutral_border_hover}; }}
            """
        )
        self.add_button.setStyleSheet(self._cfg._add_btn_style(c))
