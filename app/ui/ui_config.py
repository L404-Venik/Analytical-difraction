from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from PyQt6.QtCore import QLocale
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont, QScreen


# ---------------------------------------------------------------------------
# Theme / colour palette
# ---------------------------------------------------------------------------

@dataclass
class ColorPalette:
    """All named colour tokens used by the panel.

    Override any token to create a custom theme without touching widget code.
    """
    # Backgrounds
    window_bg: str = "#fafafa"
    card_bg: str = "#f5f5f5"
    card_bg_outer: str = "#e8f4f8"   # outer-space card
    input_bg: str = "#ffffff"
    input_bg_disabled: str = "#f0f0f0"
    group_bg: str = "transparent"

    # Borders
    border: str = "#d0d0d0"
    border_light: str = "#cccccc"
    border_focus: str = "#2196f3"
    border_outer: str = "#b8d4e8"
    border_disabled: str = "#e0e0e0"

    # Text
    text_primary: str = "#000000"
    text_secondary: str = "#333333"
    text_muted: str = "#666666"
    text_disabled: str = "#999999"
    text_outer: str = "#0a4a6a"
    text_outer_info: str = "#333333"
    text_on_accent: str = "#ffffff"

    # Accent / action colours
    accent_add: str = "#4caf50"
    accent_add_hover: str = "#45a049"
    accent_add_press: str = "#3d8b40"
    accent_calc: str = "#2196f3"
    accent_calc_hover: str = "#1976d2"
    accent_calc_press: str = "#0d47a1"
    delete_fg: str = "#d32f2f"
    delete_bg_hover: str = "#ffebee"

    # Reset / neutral button
    neutral_bg: str = "#f5f5f5"
    neutral_bg_hover: str = "#e8e8e8"
    neutral_bg_press: str = "#dddddd"
    neutral_fg: str = "#333333"
    neutral_border: str = "#cccccc"
    neutral_border_hover: str = "#aaaaaa"

    # Scrollbar
    scroll_track: str = "#f0f0f0"
    scroll_handle: str = "#c0c0c0"
    scroll_handle_hover: str = "#a0a0a0"

    # Combo-box selection
    combo_select_bg: str = "#e0e0e0"


# Ready-made themes
LIGHT_THEME = ColorPalette()   # default values above are the light theme

DARK_THEME = ColorPalette(
    window_bg="#1e1e1e",
    card_bg="#2d2d2d",
    card_bg_outer="#1a2a35",
    input_bg="#3c3c3c",
    input_bg_disabled="#2a2a2a",
    group_bg="transparent",

    border="#555555",
    border_light="#555555",
    border_focus="#42a5f5",
    border_outer="#2a5060",
    border_disabled="#404040",

    text_primary="#e0e0e0",
    text_secondary="#cccccc",
    text_muted="#999999",
    text_disabled="#666666",
    text_outer="#7ec8e3",
    text_outer_info="#aaaaaa",
    text_on_accent="#ffffff",

    accent_add="#388e3c",
    accent_add_hover="#2e7d32",
    accent_add_press="#1b5e20",
    accent_calc="#1565c0",
    accent_calc_hover="#0d47a1",
    accent_calc_press="#0a2d6e",
    delete_fg="#ef5350",
    delete_bg_hover="#3d1010",

    neutral_bg="#3a3a3a",
    neutral_bg_hover="#454545",
    neutral_bg_press="#505050",
    neutral_fg="#cccccc",
    neutral_border="#555555",
    neutral_border_hover="#777777",

    scroll_track="#2a2a2a",
    scroll_handle="#555555",
    scroll_handle_hover="#777777",

    combo_select_bg="#3a3a3a",
)


# ---------------------------------------------------------------------------
# UI configuration  (scale + theme + font)
# ---------------------------------------------------------------------------

@dataclass
class UIConfig:
    """Central configuration for all visual parameters.

    Parameters
    ----------
    theme:
        A :class:`ColorPalette` instance.  Use ``LIGHT_THEME``,
        ``DARK_THEME``, or build your own.
    base_font_pt:
        Base font size in *points* (not pixels).  All other sizes are
        derived from this so that the panel looks right at any system
        font-size setting.
    scale:
        Extra multiplicative factor applied on top of the DPI scale.
        ``1.0`` means "use only DPI-derived scaling"; ``1.25`` makes
        everything 25 % larger than the DPI-derived size.
    font_family:
        Font family name.  ``""`` (default) means "use the system
        default", which is almost always the right choice.
    spin_decimals:
        Number of decimals shown by floating-point spin controls.
    dot_decimal:
        ``True`` → spin controls use ``.`` as the decimal separator
        (C locale); ``False`` → the system locale's separator.
    """

    theme: ColorPalette = field(default_factory=ColorPalette)
    base_font_pt: int = 10
    scale: float = 1.0
    font_family: str = ""
    spin_decimals: int = 6
    dot_decimal: bool = True

    # ------------------------------------------------------------------ #
    # Factory                                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_screen(
        cls,
        screen: Optional[QScreen] = None,
        theme: ColorPalette = LIGHT_THEME,
        base_font_pt: int = 10,
        extra_scale: float = 1.0,
        font_family: str = "",
        spin_decimals: int = 6,
        dot_decimal: bool = True,
    ) -> "UIConfig":
        """Create a config whose scale is derived from the screen's logical DPI.

        The baseline DPI is 96 (Windows default / Qt default).  A 192-DPI
        (HiDPI) screen gets scale=2.0, a 144-DPI screen gets scale=1.5, etc.

        Parameters
        ----------
        screen:
            Target screen.  ``None`` → primary screen.
        extra_scale:
            Multiply the DPI-derived scale by this factor (useful for
            "large text" accessibility settings on top of DPI scaling).
        """
        if screen is None:
            app = QApplication.instance()
            screen = app.primaryScreen() if app else None

        dpi_scale = 1.0
        if screen is not None:
            logical_dpi = screen.logicalDotsPerInch()
            dpi_scale = logical_dpi / 96.0   # 96 dpi = scale 1.0 baseline

        return cls(
            theme=theme,
            base_font_pt=base_font_pt,
            scale=dpi_scale * extra_scale,
            font_family=font_family,
            spin_decimals=spin_decimals,
            dot_decimal=dot_decimal,
        )

    # ------------------------------------------------------------------ #
    # Derived size helpers                                                 #
    # ------------------------------------------------------------------ #

    def px(self, logical_96dpi_pixels: float) -> int:
        """Convert a 96-DPI logical pixel value to the scaled pixel count."""
        return max(1, round(logical_96dpi_pixels * self.scale))

    def pt(self, base_points: float) -> int:
        """Scale a point size value."""
        return max(1, round(base_points * self.scale))

    def locale(self) -> QLocale:
        """Locale for numeric input widgets, honouring `dot_decimal`."""
        locale = QLocale.c() if self.dot_decimal else QLocale.system()
        locale.setNumberOptions(QLocale.NumberOption.OmitGroupSeparator)
        return locale

    def setup_double_spin(self, spin) -> None:
        """Apply decimals and locale to a QDoubleSpinBox."""
        spin.setDecimals(self.spin_decimals)
        spin.setLocale(self.locale())

    @property
    def font(self) -> QFont:
        """Configured QFont object."""
        f = QFont()
        if self.font_family:
            f.setFamily(self.font_family)
        f.setPointSize(self.pt(self.base_font_pt))
        return f

    @property
    def font_bold(self) -> QFont:
        f = self.font
        f.setBold(True)
        return f

    # ------------------------------------------------------------------ #
    # Stylesheet factory methods                                           #
    # ------------------------------------------------------------------ #

    def _spinbox_style(self, c: ColorPalette) -> str:
        p = self.px(4)
        r = self.px(3)
        bw = self.px(16)
        return f"""
            QDoubleSpinBox {{
                background-color: {c.input_bg};
                color: {c.text_primary};
                border: 1px solid {c.border_light};
                border-radius: {r}px;
                padding: {p}px;
                font-size: {self.pt(self.base_font_pt)}pt;
            }}
            QDoubleSpinBox:focus {{
                border: 1px solid {c.border_focus};
            }}
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
                width: {bw}px;
            }}
        """

    def _spinbox_disabled_style(self, c: ColorPalette) -> str:
        p = self.px(4)
        r = self.px(3)
        return f"""
            QDoubleSpinBox {{
                background-color: {c.input_bg_disabled};
                color: {c.text_disabled};
                border: 1px solid {c.border_disabled};
                border-radius: {r}px;
                padding: {p}px;
                font-size: {self.pt(self.base_font_pt)}pt;
            }}
        """

    def _group_box_style(self, c: ColorPalette) -> str:
        mt = self.px(12)
        pt = self.px(8)
        r = self.px(4)
        lpad = self.px(10)
        fs = self.pt(self.base_font_pt)
        return f"""
            QGroupBox {{
                color: {c.text_primary};
                font-weight: bold;
                font-size: {fs}pt;
                border: 1px solid {c.border_light};
                border-radius: {r}px;
                margin-top: {mt}px;
                padding-top: {pt}px;
                background-color: {c.group_bg};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: {lpad}px;
                padding: 0 {self.px(5)}px 0 {self.px(5)}px;
            }}
        """

    def _card_style(self, c: ColorPalette) -> str:
        r = self.px(4)
        m = self.px(2)
        return f"""
            LayerCard {{
                background-color: {c.card_bg};
                border: 1px solid {c.border};
                border-radius: {r}px;
                margin: {m}px;
            }}
        """

    def _outer_card_style(self, c: ColorPalette) -> str:
        r = self.px(4)
        m = self.px(2)
        return f"""
            QFrame {{
                background-color: {c.card_bg_outer};
                border: 1px solid {c.border_outer};
                border-radius: {r}px;
                margin: {m}px;
            }}
        """

    def _delete_btn_style(self, c: ColorPalette) -> str:
        r = self.px(3)
        return f"""
            QPushButton {{
                background-color: {c.input_bg};
                color: {c.delete_fg};
                border: 1px solid {c.border_light};
                border-radius: {r}px;
                font-weight: bold;
                font-size: {self.pt(self.base_font_pt)}pt;
            }}
            QPushButton:hover {{
                background-color: {c.delete_bg_hover};
                border-color: {c.delete_fg};
            }}
        """

    def _add_btn_style(self, c: ColorPalette) -> str:
        r = self.px(4)
        pv = self.px(6)
        ph = self.px(12)
        fs = self.pt(self.base_font_pt)
        return f"""
            QPushButton {{
                background-color: {c.accent_add};
                color: {c.text_on_accent};
                border: none;
                border-radius: {r}px;
                padding: {pv}px {ph}px;
                font-weight: bold;
                font-size: {fs}pt;
            }}
            QPushButton:hover {{ background-color: {c.accent_add_hover}; }}
            QPushButton:pressed {{ background-color: {c.accent_add_press}; }}
        """

    def _reset_btn_style(self, c: ColorPalette) -> str:
        r = self.px(4)
        pv = self.px(6)
        ph = self.px(12)
        fs = self.pt(self.base_font_pt)
        return f"""
            QPushButton {{
                background-color: {c.neutral_bg};
                color: {c.neutral_fg};
                border: 1px solid {c.neutral_border};
                border-radius: {r}px;
                padding: {pv}px {ph}px;
                font-size: {fs}pt;
            }}
            QPushButton:hover {{
                background-color: {c.neutral_bg_hover};
                border-color: {c.neutral_border_hover};
            }}
            QPushButton:pressed {{ background-color: {c.neutral_bg_press}; }}
        """

    def _calc_btn_style(self, c: ColorPalette) -> str:
        r = self.px(4)
        p = self.px(8)
        fs = self.pt(self.base_font_pt)
        return f"""
            QPushButton {{
                background-color: {c.accent_calc};
                color: {c.text_on_accent};
                border: none;
                border-radius: {r}px;
                padding: {p}px;
                font-weight: bold;
                font-size: {fs}pt;
            }}
            QPushButton:hover {{ background-color: {c.accent_calc_hover}; }}
            QPushButton:pressed {{ background-color: {c.accent_calc_press}; }}
        """

    def _checkbox_style(self, c: ColorPalette) -> str:
        ind = self.px(16)
        sp = self.px(8)
        fs = self.pt(self.base_font_pt)
        return f"""
            QCheckBox {{
                color: {c.text_primary};
                spacing: {sp}px;
                font-size: {fs}pt;
            }}
            QCheckBox::indicator {{
                width: {ind}px;
                height: {ind}px;
            }}
        """

    def _combo_style(self, c: ColorPalette) -> str:
        r = self.px(3)
        p = self.px(4)
        fs = self.pt(self.base_font_pt)
        return f"""
            QComboBox {{
                background-color: {c.input_bg};
                color: {c.text_primary};
                border: 1px solid {c.border_light};
                border-radius: {r}px;
                padding: {p}px;
                font-size: {fs}pt;
            }}
            QComboBox:hover {{ border-color: {c.neutral_border_hover}; }}
            QComboBox::drop-down {{ border: none; }}
            QComboBox QAbstractItemView {{
                background-color: {c.input_bg};
                color: {c.text_primary};
                selection-background-color: {c.combo_select_bg};
            }}
        """

    def _scroll_area_style(self, c: ColorPalette) -> str:
        w = self.px(12)
        r = self.px(6)
        mh = self.px(20)
        return f"""
            QScrollArea {{
                border: none;
                background-color: transparent;
            }}
            QScrollBar:vertical {{
                background-color: {c.scroll_track};
                width: {w}px;
                border-radius: {r}px;
            }}
            QScrollBar::handle:vertical {{
                background-color: {c.scroll_handle};
                border-radius: {r}px;
                min-height: {mh}px;
            }}
            QScrollBar::handle:vertical:hover {{
                background-color: {c.scroll_handle_hover};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
        """

    def label_style(self, c: ColorPalette, color: Optional[str] = None) -> str:
        clr = color or c.text_primary
        fs = self.pt(self.base_font_pt)
        return f"color: {clr}; font-size: {fs}pt;"

    def drag_handle_style(self, c: ColorPalette) -> str:
        fs = self.pt(self.base_font_pt + 4)
        return f"color: {c.text_muted}; font-size: {fs}pt; font-weight: bold;"

    def card_title_style(self, c: ColorPalette) -> str:
        fs = self.pt(self.base_font_pt)
        return f"color: {c.text_primary}; font-weight: bold; font-size: {fs}pt;"

    def outer_title_style(self, c: ColorPalette) -> str:
        fs = self.pt(self.base_font_pt)
        return f"color: {c.text_outer}; font-weight: bold; font-size: {fs}pt;"

    def outer_info_style(self, c: ColorPalette) -> str:
        fs = self.pt(self.base_font_pt - 1)
        return f"color: {c.text_outer_info}; font-size: {fs}pt;"

