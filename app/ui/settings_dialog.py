# settings_dialog.py
"""
Settings dialog — tabbed, theme-aware.

Tabs
----
App  — reserved for future application-level settings (blank for now).
UI   — colour palette, font family, font size.

Usage
-----
    dlg = SettingsDialog(current_config=self._ui_cfg, parent=self)
    if dlg.exec():
        self._apply_config(dlg.result_config())
"""

from __future__ import annotations

from typing import Dict, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFontDatabase
from PyQt6.QtWidgets import (
    QDialog, QDialogButtonBox, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QSpinBox, QTabWidget, QVBoxLayout,
    QWidget, QComboBox, QFontComboBox, QFrame, QSizePolicy,
)

# Adjust this import path to match your package layout.
from .ui_config import UIConfig, ColorPalette, LIGHT_THEME, DARK_THEME


# ---------------------------------------------------------------------------
# Registry of named palettes.  Add your own entries here.
# ---------------------------------------------------------------------------
PALETTE_REGISTRY: Dict[str, ColorPalette] = {
    "Light": LIGHT_THEME,
    "Dark":  DARK_THEME,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_separator() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    return line


# ---------------------------------------------------------------------------
# Individual tabs
# ---------------------------------------------------------------------------

class _AppTab(QWidget):
    """Placeholder — fill in application-level settings here later."""

    def __init__(self, config: UIConfig, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        note = QLabel("Application settings will appear here.")
        note.setStyleSheet(f"color: {config.theme.text_muted};")
        note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(note)
        layout.addStretch()

    # Nothing to read back yet.
    def apply_to(self, config: UIConfig) -> UIConfig:
        return config


class _UITab(QWidget):
    """Colour palette, font family and font size controls."""

    def __init__(self, config: UIConfig, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._cfg = config
        c = config.theme

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # ── Colour theme ────────────────────────────────────────────────
        theme_group = QGroupBox("Colour theme")
        theme_group.setStyleSheet(self._group_style(c))
        theme_form = QFormLayout(theme_group)
        theme_form.setSpacing(10)
        theme_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self._palette_combo = QComboBox()
        self._palette_combo.setStyleSheet(self._combo_style(c))
        for name in PALETTE_REGISTRY:
            self._palette_combo.addItem(name)

        # Pre-select the palette that matches the current theme object.
        current_name = self._detect_palette_name(config.theme)
        if current_name:
            self._palette_combo.setCurrentText(current_name)

        palette_lbl = QLabel("Palette:")
        palette_lbl.setStyleSheet(self._label_style(c))
        theme_form.addRow(palette_lbl, self._palette_combo)

        # Colour preview strip — redrawn when the combo changes.
        self._preview_strip = _ColourPreviewStrip(config.theme)
        theme_form.addRow(QLabel(""), self._preview_strip)

        self._palette_combo.currentTextChanged.connect(self._on_palette_changed)
        layout.addWidget(theme_group)

        # ── Typography ──────────────────────────────────────────────────
        font_group = QGroupBox("Typography")
        font_group.setStyleSheet(self._group_style(c))
        font_form = QFormLayout(font_group)
        font_form.setSpacing(10)
        font_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self._font_combo = QFontComboBox()
        self._font_combo.setStyleSheet(self._combo_style(c))
        self._font_combo.setEditable(False)
        # Pre-select current family (empty string → leave at system default).
        if config.font_family:
            self._font_combo.setCurrentFont(self._font_combo.currentFont())
            # Walk to the right family name.
            idx = self._font_combo.findText(
                config.font_family, Qt.MatchFlag.MatchFixedString
            )
            if idx >= 0:
                self._font_combo.setCurrentIndex(idx)

        font_lbl = QLabel("Font family:")
        font_lbl.setStyleSheet(self._label_style(c))
        font_form.addRow(font_lbl, self._font_combo)

        self._size_spin = QSpinBox()
        self._size_spin.setRange(6, 32)
        self._size_spin.setValue(config.base_font_pt)
        self._size_spin.setSuffix(" pt")
        self._size_spin.setStyleSheet(self._spinbox_style(c))

        size_lbl = QLabel("Base font size:")
        size_lbl.setStyleSheet(self._label_style(c))
        font_form.addRow(size_lbl, self._size_spin)

        layout.addWidget(font_group)
        layout.addStretch()

    # ── Palette change ───────────────────────────────────────────────────

    def _on_palette_changed(self, name: str):
        palette = PALETTE_REGISTRY.get(name, LIGHT_THEME)
        self._preview_strip.update_palette(palette)

    # ── Read back ────────────────────────────────────────────────────────

    def result_config(self) -> UIConfig:
        """Return a new UIConfig reflecting the current widget state."""
        palette_name = self._palette_combo.currentText()
        palette = PALETTE_REGISTRY.get(palette_name, LIGHT_THEME)

        font_family = self._font_combo.currentFont().family()
        font_size_pt = self._size_spin.value()

        # Keep the scale factor from the original config (DPI-derived).
        return UIConfig(
            theme=palette,
            base_font_pt=font_size_pt,
            scale=self._cfg.scale,
            font_family=font_family,
        )

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _detect_palette_name(palette: ColorPalette) -> Optional[str]:
        """Return the registry key whose value is identical to *palette*."""
        for name, p in PALETTE_REGISTRY.items():
            if p == palette:
                return name
        return None

    @staticmethod
    def _group_style(c: ColorPalette) -> str:
        return f"""
            QGroupBox {{
                color: {c.text_primary};
                font-weight: bold;
                border: 1px solid {c.border_light};
                border-radius: 4px;
                margin-top: 12px;
                padding-top: 8px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }}
        """

    @staticmethod
    def _label_style(c: ColorPalette) -> str:
        return f"color: {c.text_primary};"

    @staticmethod
    def _combo_style(c: ColorPalette) -> str:
        return f"""
            QComboBox {{
                background-color: {c.input_bg};
                color: {c.text_primary};
                border: 1px solid {c.border_light};
                border-radius: 3px;
                padding: 4px 8px;
                min-width: 160px;
            }}
            QComboBox:hover {{ border-color: {c.neutral_border_hover}; }}
            QComboBox::drop-down {{ border: none; }}
            QComboBox QAbstractItemView {{
                background-color: {c.input_bg};
                color: {c.text_primary};
                selection-background-color: {c.combo_select_bg};
            }}
        """

    @staticmethod
    def _spinbox_style(c: ColorPalette) -> str:
        return f"""
            QSpinBox {{
                background-color: {c.input_bg};
                color: {c.text_primary};
                border: 1px solid {c.border_light};
                border-radius: 3px;
                padding: 4px;
                min-width: 80px;
            }}
            QSpinBox:focus {{ border-color: {c.border_focus}; }}
        """


# ---------------------------------------------------------------------------
# Colour preview strip
# ---------------------------------------------------------------------------

class _ColourPreviewStrip(QWidget):
    """A row of coloured swatches showing the key tokens of a palette."""

    _TOKENS = [
        ("window_bg",   "Window"),
        ("card_bg",     "Card"),
        ("input_bg",    "Input"),
        ("accent_add",  "Add"),
        ("accent_calc", "Calc"),
        ("text_primary","Text"),
    ]

    def __init__(self, palette: ColorPalette, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setFixedHeight(36)
        self._palette = palette
        self._swatches: list[_Swatch] = []

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)

        for attr, label in self._TOKENS:
            sw = _Swatch(getattr(palette, attr), label)
            self._swatches.append((attr, sw))
            row.addWidget(sw)

    def update_palette(self, palette: ColorPalette):
        self._palette = palette
        for attr, sw in self._swatches:
            sw.set_colour(getattr(palette, attr))


class _Swatch(QFrame):
    """Single colour swatch with a tooltip label."""

    def __init__(self, colour: str, label: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setFixedSize(48, 28)
        self.setToolTip(f"{label}\n{colour}")
        self._apply(colour)

    def set_colour(self, colour: str):
        self.setToolTip(f"{self.toolTip().split()[0]}\n{colour}")
        self._apply(colour)

    def _apply(self, colour: str):
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {colour};
                border: 1px solid #888;
                border-radius: 3px;
            }}
        """)


# ---------------------------------------------------------------------------
# The dialog itself
# ---------------------------------------------------------------------------

class SettingsDialog(QDialog):
    """Modal settings dialog.

    Parameters
    ----------
    current_config:
        The ``UIConfig`` currently in use.  Its values pre-populate all
        controls.
    parent:
        Parent widget (used for positioning and modality).

    After ``exec()`` returns ``QDialog.Accepted``, call
    :meth:`result_config` to obtain the new ``UIConfig``.
    """

    def __init__(
        self,
        current_config: UIConfig,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._cfg = current_config
        c = current_config.theme

        self.setWindowTitle("Settings")
        self.setMinimumWidth(420)
        self.setModal(True)
        self._apply_dialog_style(c)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 12)
        root.setSpacing(0)

        # ── Tab widget ───────────────────────────────────────────────────
        self._tabs = QTabWidget()
        self._tabs.setStyleSheet(self._tab_style(c))

        self._app_tab = _AppTab(current_config)
        self._ui_tab  = _UITab(current_config)

        self._tabs.addTab(self._app_tab, "App")
        self._tabs.addTab(self._ui_tab,  "UI")
        root.addWidget(self._tabs)

        # ── Buttons ──────────────────────────────────────────────────────
        root.addWidget(_make_separator())

        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.setStyleSheet(self._button_box_style(c))
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(12, 8, 12, 0)
        btn_row.addWidget(btn_box)
        root.addLayout(btn_row)

    # ── Public API ───────────────────────────────────────────────────────

    def result_config(self) -> UIConfig:
        """Return the UIConfig built from the current dialog state."""
        return self._ui_tab.result_config()

    # ── Stylesheet helpers ───────────────────────────────────────────────

    def _apply_dialog_style(self, c: ColorPalette):
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {c.window_bg};
                color: {c.text_primary};
            }}
            QLabel {{
                color: {c.text_primary};
            }}
        """)

    @staticmethod
    def _tab_style(c: ColorPalette) -> str:
        return f"""
            QTabWidget::pane {{
                border: none;
                background-color: {c.window_bg};
            }}
            QTabBar::tab {{
                background-color: {c.neutral_bg};
                color: {c.text_secondary};
                border: 1px solid {c.border_light};
                border-bottom: none;
                padding: 6px 18px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }}
            QTabBar::tab:selected {{
                background-color: {c.window_bg};
                color: {c.text_primary};
                font-weight: bold;
            }}
            QTabBar::tab:hover:!selected {{
                background-color: {c.neutral_bg_hover};
            }}
        """

    @staticmethod
    def _button_box_style(c: ColorPalette) -> str:
        return f"""
            QPushButton {{
                background-color: {c.neutral_bg};
                color: {c.neutral_fg};
                border: 1px solid {c.neutral_border};
                border-radius: 4px;
                padding: 5px 18px;
                min-width: 72px;
            }}
            QPushButton:hover {{
                background-color: {c.neutral_bg_hover};
                border-color: {c.neutral_border_hover};
            }}
            QPushButton:pressed {{
                background-color: {c.neutral_bg_press};
            }}
            QPushButton[text="OK"] {{
                background-color: {c.accent_calc};
                color: {c.text_on_accent};
                border: none;
            }}
            QPushButton[text="OK"]:hover {{
                background-color: {c.accent_calc_hover};
            }}
        """