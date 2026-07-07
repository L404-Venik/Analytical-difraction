from __future__ import annotations

import json
from importlib import metadata

from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtWidgets import QFileDialog, QLabel, QMainWindow, QMessageBox, QSplitter

from app.application.computation import ComputationResult
from app.application.controller import AppController
from app.application.experiment import load_preset, save_preset

from .parameter_panel import ParameterPanel
from .plot_grid import PlotGrid
from .settings_dialog import PALETTE_REGISTRY, SettingsDialog
from .ui_config import LIGHT_THEME, UIConfig

ORGANIZATION = "sphere-diffraction"
APPLICATION = "gui"


class MainWindow(QMainWindow):
    def __init__(self, controller: AppController):
        super().__init__()
        self._controller = controller
        self._settings = QSettings(ORGANIZATION, APPLICATION)
        self._ui_cfg = self._load_ui_config()

        self.setWindowTitle("Sphere Diffraction")
        self.resize(1600, 900)
        self._build_menu()
        self._build_central()
        self._build_status_bar()
        self._restyle_window()
        self._wire()
        self._restore_session()

        self._controller.set_state(self.param_panel.get_state(), defer=True)

    # ------------------------------------------------------------------ #
    # Construction                                                         #
    # ------------------------------------------------------------------ #

    def _load_ui_config(self) -> UIConfig:
        palette_name = self._settings.value("palette", "Light", type=str)
        font_family = self._settings.value("font_family", "", type=str)
        base_font_pt = self._settings.value("base_font_pt", 10, type=int)
        return UIConfig.from_screen(
            theme=PALETTE_REGISTRY.get(palette_name, LIGHT_THEME),
            base_font_pt=base_font_pt,
            font_family=font_family,
        )

    def _build_menu(self):
        menu = self.menuBar()

        file_menu = menu.addMenu("File")
        file_menu.addAction("Save Preset…").triggered.connect(self._save_preset)
        file_menu.addAction("Load Preset…").triggered.connect(self._load_preset)
        file_menu.addSeparator()
        file_menu.addAction("Exit").triggered.connect(self.close)

        settings_menu = menu.addMenu("Settings")
        settings_menu.addAction("Preferences…").triggered.connect(self._open_settings)

        help_menu = menu.addMenu("Help")
        help_menu.addAction("About").triggered.connect(self._show_about)

    def _build_central(self):
        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.setHandleWidth(6)

        self.param_panel = ParameterPanel(config=self._ui_cfg)
        self.plot_grid = PlotGrid(config=self._ui_cfg)

        self._splitter.addWidget(self.param_panel)
        self._splitter.addWidget(self.plot_grid)
        self._splitter.setSizes([400, 1200])
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)

        self.setCentralWidget(self._splitter)

    def _build_status_bar(self):
        self._status_label = QLabel("Ready")
        self.statusBar().addWidget(self._status_label)

    def _wire(self):
        self.param_panel.parameters_changed.connect(self._on_parameters_changed)
        self.param_panel.calculate_requested.connect(self._compute_now)
        self.param_panel.auto_refresh_cb.toggled.connect(self._on_auto_refresh_toggled)

        self._controller.computation_started.connect(self._on_computation_started)
        self._controller.result_ready.connect(self._on_result_ready)
        self._controller.computation_failed.connect(self._on_computation_failed)

    def _restore_session(self):
        geometry = self._settings.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        raw_layout = self._settings.value("plot_layout", "", type=str)
        if raw_layout:
            try:
                specs = json.loads(raw_layout)
            except ValueError:
                specs = []
            if isinstance(specs, list) and specs:
                self.plot_grid.restore_layout(specs)

    # ------------------------------------------------------------------ #
    # Computation flow                                                     #
    # ------------------------------------------------------------------ #

    def _on_parameters_changed(self):
        if self.param_panel.get_auto_refresh():
            self._controller.set_state(self.param_panel.get_state(), defer=True)

    def _compute_now(self):
        self._controller.set_state(self.param_panel.get_state(), defer=False)

    def _on_auto_refresh_toggled(self, checked: bool):
        if checked:
            self._controller.set_state(self.param_panel.get_state(), defer=True)

    def _on_computation_started(self):
        self._set_status("Computing…", muted=True)

    def _on_result_ready(self, result: ComputationResult):
        self.plot_grid.show_result(result)
        state = result.state
        self._set_status(
            f"Computed in {result.elapsed_seconds:.2f} s · "
            f"{len(result.angles)} angles · λ = {state.wavelength:g} m · "
            f"{len(state.layers)} region(s)"
        )

    def _on_computation_failed(self, message: str):
        self._set_status(f"Error: {message}", error=True)

    def _set_status(self, text: str, muted: bool = False, error: bool = False):
        c = self._ui_cfg.theme
        color = c.delete_fg if error else (c.text_muted if muted else c.text_primary)
        self._status_label.setStyleSheet(f"color: {color};")
        self._status_label.setText(text)

    # ------------------------------------------------------------------ #
    # Menu actions                                                         #
    # ------------------------------------------------------------------ #

    def _save_preset(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Preset", "", "Preset files (*.json)"
        )
        if path:
            save_preset(path, self.param_panel.get_state())
            self._set_status(f"Preset saved to {path}")

    def _load_preset(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Preset", "", "Preset files (*.json)"
        )
        if not path:
            return
        try:
            state = load_preset(path)
        except Exception as exc:
            QMessageBox.warning(self, "Load Preset", f"Could not load preset:\n{exc}")
            return
        self.param_panel.set_state(state)
        self._controller.set_state(state, defer=False)

    def _open_settings(self):
        dlg = SettingsDialog(current_config=self._ui_cfg, parent=self)
        if dlg.exec():
            self._apply_config(dlg.result_config())

    def _apply_config(self, config: UIConfig):
        self._ui_cfg = config
        self.param_panel.apply_config(config)
        self.plot_grid.apply_config(config)
        self._restyle_window()

        self._settings.setValue("palette", self._palette_name(config))
        self._settings.setValue("font_family", config.font_family)
        self._settings.setValue("base_font_pt", config.base_font_pt)

    @staticmethod
    def _palette_name(config: UIConfig) -> str:
        for name, palette in PALETTE_REGISTRY.items():
            if palette == config.theme:
                return name
        return "Light"

    def _show_about(self):
        try:
            version = metadata.version("analytical-diffraction")
        except metadata.PackageNotFoundError:
            version = "dev"
        QMessageBox.about(
            self,
            "About Sphere Diffraction",
            "Sphere Diffraction GUI\n\n"
            "Mie-theory scattering patterns and RCS for multilayer spheres.\n"
            f"analytical-diffraction {version}",
        )

    # ------------------------------------------------------------------ #
    # Theming / persistence                                                #
    # ------------------------------------------------------------------ #

    def _restyle_window(self):
        c = self._ui_cfg.theme
        self.setStyleSheet(
            f"""
            QMainWindow {{ background-color: {c.window_bg}; }}
            QMenuBar {{ background-color: {c.neutral_bg}; color: {c.text_primary}; }}
            QMenuBar::item:selected {{ background-color: {c.neutral_bg_hover}; }}
            QMenu {{ background-color: {c.input_bg}; color: {c.text_primary}; }}
            QMenu::item:selected {{ background-color: {c.combo_select_bg}; }}
            QStatusBar {{ background-color: {c.neutral_bg}; }}
            QSplitter::handle {{ background-color: {c.border}; }}
            QSplitter::handle:hover {{ background-color: {c.neutral_border_hover}; }}
            """
        )
        self._set_status(self._status_label.text() or "Ready")

    def closeEvent(self, event):
        self._settings.setValue("geometry", self.saveGeometry())
        self._settings.setValue(
            "plot_layout", json.dumps(self.plot_grid.layout_spec())
        )
        super().closeEvent(event)
