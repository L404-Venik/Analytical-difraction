from __future__ import annotations

from typing import List, Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QFormLayout,
    QDoubleSpinBox, QPushButton, QCheckBox, QFrame, QHBoxLayout,
    QLabel, QScrollArea, QComboBox,
)
from PyQt6.QtCore import Qt, pyqtSignal

from app.application.experiment import ExperimentState, FIDELITY_ANGLES, LayerSpec

from .ui_config import UIConfig


class LayerCard(QFrame):
    """Card editing one LayerSpec (radius/thickness plus complex permittivity)."""

    deleted = pyqtSignal(int)
    data_changed = pyqtSignal(int, LayerSpec)

    def __init__(
        self,
        index: int,
        data: LayerSpec,
        is_core: bool = False,
        conducting: bool = False,
        config: Optional[UIConfig] = None,
    ):
        super().__init__()
        self.index = index
        self.is_core = is_core
        self.data = data
        self._conducting = conducting
        self._cfg = config or UIConfig()
        c = self._cfg.theme

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(self._cfg._card_style(c))
        self.setFont(self._cfg.font)

        self._setup_ui()
        self._update_epsilon_state()

    def _setup_ui(self):
        cfg = self._cfg
        c = cfg.theme

        layout = QVBoxLayout(self)
        layout.setSpacing(cfg.px(6))
        layout.setContentsMargins(*([cfg.px(10)] * 4))

        header = QHBoxLayout()

        title_text = "Core" if self.is_core else f"Layer {self.index}"
        title = QLabel(title_text)
        title.setStyleSheet(cfg.card_title_style(c))
        header.addWidget(title)
        header.addStretch()

        if not self.is_core:
            btn_size = cfg.px(24)
            delete_btn = QPushButton("✕")
            delete_btn.setFixedSize(btn_size, btn_size)
            delete_btn.setStyleSheet(cfg._delete_btn_style(c))
            delete_btn.clicked.connect(lambda: self.deleted.emit(self.index))
            header.addWidget(delete_btn)

        layout.addLayout(header)

        form = QFormLayout()
        form.setSpacing(cfg.px(8))
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        spinbox_style = cfg._spinbox_style(c)
        label_style = cfg.label_style(c)

        thickness_label = "radius (m):" if self.is_core else "thickness (m):"
        min_thickness = 0.0001 if self.is_core else 0.0

        self.thickness_spin = self._make_spinbox(
            range_=(min_thickness, 1000.0),
            value=self.data.thickness,
            step=0.1,
            style=spinbox_style,
            slot=self._on_thickness_changed,
        )
        lbl = QLabel(thickness_label)
        lbl.setStyleSheet(label_style)
        form.addRow(lbl, self.thickness_spin)

        self.eps_real_spin = self._make_spinbox(
            range_=(0.0, 100.0),
            value=self.data.eps_real,
            step=0.1,
            style=spinbox_style,
            slot=self._on_eps_real_changed,
        )
        lbl_real = QLabel("Re ε:")
        lbl_real.setStyleSheet(label_style)
        form.addRow(lbl_real, self.eps_real_spin)

        self.eps_imag_spin = self._make_spinbox(
            range_=(0.0, 100.0),
            value=self.data.eps_imag,
            step=0.01,
            style=spinbox_style,
            slot=self._on_eps_imag_changed,
        )
        lbl_imag = QLabel("Im ε:")
        lbl_imag.setStyleSheet(label_style)
        form.addRow(lbl_imag, self.eps_imag_spin)

        layout.addLayout(form)

    def _make_spinbox(
        self,
        range_: tuple,
        value: float,
        step: float,
        style: str,
        slot,
    ) -> QDoubleSpinBox:
        sb = QDoubleSpinBox()
        sb.setRange(*range_)
        self._cfg.setup_double_spin(sb)
        sb.setSingleStep(step)
        sb.setValue(value)
        sb.setStyleSheet(style)
        sb.valueChanged.connect(slot)
        return sb

    def _on_thickness_changed(self, value: float):
        self.data.thickness = value
        self.data_changed.emit(self.index, self.data)

    def _on_eps_real_changed(self, value: float):
        self.data.eps_real = value
        self.data_changed.emit(self.index, self.data)

    def _on_eps_imag_changed(self, value: float):
        self.data.eps_imag = value
        self.data_changed.emit(self.index, self.data)

    def _update_epsilon_state(self):
        cfg = self._cfg
        c = cfg.theme
        disabled = self.is_core and self._conducting
        self.eps_real_spin.setEnabled(not disabled)
        self.eps_imag_spin.setEnabled(not disabled)
        style = cfg._spinbox_disabled_style(c) if disabled else cfg._spinbox_style(c)
        self.eps_real_spin.setStyleSheet(style)
        self.eps_imag_spin.setStyleSheet(style)

    def set_conducting(self, conducting: bool):
        if self.is_core:
            self._conducting = conducting
            self._update_epsilon_state()


class ParameterPanel(QWidget):
    """Left-hand panel editing the full ExperimentState."""

    parameters_changed = pyqtSignal()
    calculate_requested = pyqtSignal()

    def __init__(self, config: Optional[UIConfig] = None):
        super().__init__()
        self._cfg = config or UIConfig()
        self.layers: List[LayerSpec] = []
        self.layer_cards: List[LayerCard] = []
        self._conducting_core: bool = True
        self._outer_eps_real: float = 1.0
        self._outer_eps_imag: float = 0.0

        c = self._cfg.theme
        self.setStyleSheet(f"background-color: {c.window_bg};")
        self.setFont(self._cfg.font)
        self._setup_ui()
        self.set_state(ExperimentState())

    # ------------------------------------------------------------------ #
    # UI construction                                                      #
    # ------------------------------------------------------------------ #

    def _setup_ui(self):
        cfg = self._cfg
        c = cfg.theme

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setContentsMargins(*([cfg.px(10)] * 4))
        layout.setSpacing(cfg.px(12))

        # --- Wavelength group ---
        wave_group = self._wave_group = QGroupBox("Wave parameters")
        wave_group.setStyleSheet(cfg._group_box_style(c))
        wave_form = QFormLayout(wave_group)
        wave_form.setSpacing(cfg.px(8))

        self.wavelength_spin = QDoubleSpinBox()
        self.wavelength_spin.setRange(0.0001, 100.0)
        cfg.setup_double_spin(self.wavelength_spin)
        self.wavelength_spin.setSingleStep(0.01)
        self.wavelength_spin.setValue(0.55)
        self.wavelength_spin.setStyleSheet(cfg._spinbox_style(c))
        self.wavelength_spin.valueChanged.connect(self._on_any_change)

        self._lbl_wave = QLabel("Wavelength λ (m):")
        self._lbl_wave.setStyleSheet(cfg.label_style(c))
        wave_form.addRow(self._lbl_wave, self.wavelength_spin)
        layout.addWidget(wave_group)

        # --- Layer controls header ---
        controls_frame = QFrame()
        controls_frame.setStyleSheet("background-color: transparent;")
        controls_layout = QHBoxLayout(controls_frame)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(cfg.px(8))

        btn_h = cfg.px(32)

        self.add_button = QPushButton("+ Add Layer")
        self.add_button.setFixedHeight(btn_h)
        self.add_button.setStyleSheet(cfg._add_btn_style(c))
        self.add_button.clicked.connect(self._add_layer)

        self.reset_button = QPushButton("Reset to Default")
        self.reset_button.setFixedHeight(btn_h)
        self.reset_button.setStyleSheet(cfg._reset_btn_style(c))
        self.reset_button.clicked.connect(self._reset_to_default)

        self.conductive_core_cb = QCheckBox("Conducting Core (PEC)")
        self.conductive_core_cb.setStyleSheet(cfg._checkbox_style(c))
        self.conductive_core_cb.stateChanged.connect(self._on_conducting_core_changed)

        controls_layout.addWidget(self.add_button)
        controls_layout.addWidget(self.reset_button)
        controls_layout.addStretch()
        controls_layout.addWidget(self.conductive_core_cb)
        layout.addWidget(controls_frame)

        # --- Scrollable layers area ---
        scroll_area = self._scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet(cfg._scroll_area_style(c))

        self.layers_container = QWidget()
        self.layers_container.setStyleSheet("background-color: transparent;")
        self.layers_layout = QVBoxLayout(self.layers_container)
        self.layers_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.layers_layout.setSpacing(cfg.px(8))
        self.layers_layout.setContentsMargins(0, 0, 0, 0)

        scroll_area.setWidget(self.layers_container)
        layout.addWidget(scroll_area)

        # --- Computation group ---
        compute_group = self._compute_group = QGroupBox("Computation")
        compute_group.setStyleSheet(cfg._group_box_style(c))
        compute_layout = QVBoxLayout(compute_group)
        compute_layout.setSpacing(cfg.px(8))

        fidelity_row = QHBoxLayout()
        self.fidelity_combo = QComboBox()
        self.fidelity_combo.addItems(list(FIDELITY_ANGLES))
        self.fidelity_combo.setStyleSheet(cfg._combo_style(c))
        self.fidelity_combo.currentTextChanged.connect(self._on_any_change)

        self._lbl_fidelity = QLabel("Fidelity:")
        self._lbl_fidelity.setStyleSheet(cfg.label_style(c))
        fidelity_row.addWidget(self._lbl_fidelity)
        fidelity_row.addWidget(self.fidelity_combo, stretch=1)
        compute_layout.addLayout(fidelity_row)

        self.auto_refresh_cb = QCheckBox("Auto-refresh")
        self.auto_refresh_cb.setChecked(True)
        self.auto_refresh_cb.setStyleSheet(cfg._checkbox_style(c))
        compute_layout.addWidget(self.auto_refresh_cb)

        self.calc_button = QPushButton("Calculate Now")
        self.calc_button.setStyleSheet(cfg._calc_btn_style(c))
        self.calc_button.clicked.connect(self.calculate_requested)
        compute_layout.addWidget(self.calc_button)

        layout.addWidget(compute_group)

    # ------------------------------------------------------------------ #
    # State I/O                                                            #
    # ------------------------------------------------------------------ #

    def get_state(self) -> ExperimentState:
        return ExperimentState(
            layers=[
                LayerSpec(l.thickness, l.eps_real, l.eps_imag) for l in self.layers
            ],
            conducting_core=self._conducting_core,
            outer_eps_real=self._outer_eps_real,
            outer_eps_imag=self._outer_eps_imag,
            wavelength=self.wavelength_spin.value(),
            fidelity=self.fidelity_combo.currentText(),
        )

    def set_state(self, state: ExperimentState) -> None:
        self.layers = [
            LayerSpec(l.thickness, l.eps_real, l.eps_imag) for l in state.layers
        ]
        self._conducting_core = state.conducting_core
        self._outer_eps_real = state.outer_eps_real
        self._outer_eps_imag = state.outer_eps_imag

        for widget in (self.wavelength_spin, self.fidelity_combo, self.conductive_core_cb):
            widget.blockSignals(True)
        self.wavelength_spin.setValue(state.wavelength)
        self.fidelity_combo.setCurrentText(state.fidelity)
        self.conductive_core_cb.setChecked(state.conducting_core)
        for widget in (self.wavelength_spin, self.fidelity_combo, self.conductive_core_cb):
            widget.blockSignals(False)

        self._rebuild_cards()
        self.parameters_changed.emit()

    def get_auto_refresh(self) -> bool:
        return self.auto_refresh_cb.isChecked()

    # ------------------------------------------------------------------ #
    # Layer management                                                     #
    # ------------------------------------------------------------------ #

    def _reset_to_default(self):
        self.set_state(ExperimentState())

    def _add_layer(self):
        self.layers.append(LayerSpec(thickness=0.1, eps_real=1.5, eps_imag=0.0))
        self._rebuild_cards()
        self.parameters_changed.emit()

    def _delete_layer(self, index: int):
        if index == 0:
            return
        if 0 <= index < len(self.layers):
            self.layers.pop(index)
            self._rebuild_cards()
            self.parameters_changed.emit()

    def _on_layer_data_changed(self, index: int, data: LayerSpec):
        if 0 <= index < len(self.layers):
            self.layers[index] = data
            self.parameters_changed.emit()

    def _on_conducting_core_changed(self, state):
        self._conducting_core = state == Qt.CheckState.Checked.value
        if self.layer_cards:
            self.layer_cards[0].set_conducting(self._conducting_core)
        self.parameters_changed.emit()

    def _on_any_change(self, *args):
        self.parameters_changed.emit()

    def _on_outer_eps_real_changed(self, value: float):
        self._outer_eps_real = value
        self.parameters_changed.emit()

    def _on_outer_eps_imag_changed(self, value: float):
        self._outer_eps_imag = value
        self.parameters_changed.emit()

    def _clear_all_cards(self):
        for card in self.layer_cards:
            card.deleteLater()
        self.layer_cards.clear()
        while self.layers_layout.count():
            item = self.layers_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _rebuild_cards(self):
        self._clear_all_cards()
        cfg = self._cfg
        c = cfg.theme

        for i, layer_data in enumerate(self.layers):
            card = LayerCard(
                i,
                layer_data,
                is_core=(i == 0),
                conducting=self._conducting_core,
                config=cfg,
            )
            card.deleted.connect(self._delete_layer)
            card.data_changed.connect(self._on_layer_data_changed)
            self.layer_cards.append(card)
            self.layers_layout.addWidget(card)

        # Fixed outer-space card
        outer_card = QFrame()
        outer_card.setFrameShape(QFrame.Shape.StyledPanel)
        outer_card.setStyleSheet(cfg._outer_card_style(c))

        outer_layout = QVBoxLayout(outer_card)
        outer_layout.setSpacing(cfg.px(4))
        outer_layout.setContentsMargins(*([cfg.px(10)] * 4))

        outer_title = QLabel("OUTER SPACE")
        outer_title.setStyleSheet(cfg.outer_title_style(c))
        outer_layout.addWidget(outer_title)

        outer_form = QFormLayout()
        outer_form.setSpacing(cfg.px(4))
        outer_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        outer_spin_style = cfg._spinbox_style(c)

        self.outer_eps_real_spin = QDoubleSpinBox()
        self.outer_eps_real_spin.setRange(0.0, 100.0)
        cfg.setup_double_spin(self.outer_eps_real_spin)
        self.outer_eps_real_spin.setSingleStep(0.1)
        self.outer_eps_real_spin.setValue(self._outer_eps_real)
        self.outer_eps_real_spin.setStyleSheet(outer_spin_style)
        self.outer_eps_real_spin.valueChanged.connect(self._on_outer_eps_real_changed)

        self.outer_eps_imag_spin = QDoubleSpinBox()
        self.outer_eps_imag_spin.setRange(0.0, 100.0)
        cfg.setup_double_spin(self.outer_eps_imag_spin)
        self.outer_eps_imag_spin.setSingleStep(0.01)
        self.outer_eps_imag_spin.setValue(self._outer_eps_imag)
        self.outer_eps_imag_spin.setStyleSheet(outer_spin_style)
        self.outer_eps_imag_spin.valueChanged.connect(self._on_outer_eps_imag_changed)

        lbl_real = QLabel("Re ε:")
        lbl_real.setStyleSheet(cfg.outer_info_style(c))
        lbl_imag = QLabel("Im ε:")
        lbl_imag.setStyleSheet(cfg.outer_info_style(c))
        outer_form.addRow(lbl_real, self.outer_eps_real_spin)
        outer_form.addRow(lbl_imag, self.outer_eps_imag_spin)
        outer_layout.addLayout(outer_form)

        self.layers_layout.addWidget(outer_card)
        self.layers_layout.addStretch()

    # ------------------------------------------------------------------ #
    # Theme / config hot-swap                                              #
    # ------------------------------------------------------------------ #

    def apply_config(self, config: UIConfig):
        self._cfg = config
        self._restyle_widgets()
        self._rebuild_cards()

    def _restyle_widgets(self):
        cfg = self._cfg
        c = cfg.theme

        self.setStyleSheet(f"background-color: {c.window_bg};")
        self.setFont(cfg.font)

        group_style = cfg._group_box_style(c)
        self._wave_group.setStyleSheet(group_style)
        self._compute_group.setStyleSheet(group_style)

        self._scroll_area.setStyleSheet(cfg._scroll_area_style(c))

        label_style = cfg.label_style(c)
        self._lbl_wave.setStyleSheet(label_style)
        self._lbl_fidelity.setStyleSheet(label_style)

        self.wavelength_spin.setStyleSheet(cfg._spinbox_style(c))
        cfg.setup_double_spin(self.wavelength_spin)

        self.add_button.setStyleSheet(cfg._add_btn_style(c))
        self.reset_button.setStyleSheet(cfg._reset_btn_style(c))
        self.calc_button.setStyleSheet(cfg._calc_btn_style(c))

        cb_style = cfg._checkbox_style(c)
        self.conductive_core_cb.setStyleSheet(cb_style)
        self.auto_refresh_cb.setStyleSheet(cb_style)

        self.fidelity_combo.setStyleSheet(cfg._combo_style(c))
