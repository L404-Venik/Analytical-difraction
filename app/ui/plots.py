from __future__ import annotations

from typing import Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QWidget

from app.application.computation import ComputationResult

from .ui_config import UIConfig

POLARIZATION_THETA = "S_θ"
POLARIZATION_PHI = "S_φ"
POLARIZATION_BOTH = "Both"
POLARIZATIONS = [POLARIZATION_THETA, POLARIZATION_PHI, POLARIZATION_BOTH]

SCALE_LINEAR = "Linear"
SCALE_LOG = "Log"
SCALES = [SCALE_LINEAR, SCALE_LOG]

_LOG_FLOOR = 1e-300
RESIZE_DEBOUNCE_MS = 120


class ResultCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas that renders one view of a ComputationResult.

    Uses fixed margins instead of a layout engine, and debounces re-renders
    while the widget is being resized, so splitter drags stay responsive.
    """

    polar = False
    margins = dict(left=0.12, right=0.96, bottom=0.15, top=0.9)

    def __init__(self, config: Optional[UIConfig] = None):
        self._fig = Figure(figsize=(4.0, 3.5))
        super().__init__(self._fig)
        self._cfg = config or UIConfig()
        self._result: ComputationResult | None = None
        self._polarization = POLARIZATION_BOTH
        projection = "polar" if self.polar else None
        self._ax = self._fig.add_subplot(111, projection=projection)
        self._fig.subplots_adjust(**self.margins)

        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(RESIZE_DEBOUNCE_MS)
        self._resize_timer.timeout.connect(self._finish_resize)

        self._redraw()

    def resizeEvent(self, event):
        if not self.isVisible():
            super().resizeEvent(event)
            return
        QWidget.resizeEvent(self, event)
        self._resize_timer.start()

    def _finish_resize(self):
        ratio = self.device_pixel_ratio or 1
        dpi = self._fig.dpi
        self._fig.set_size_inches(
            max(1, self.width()) * ratio / dpi,
            max(1, self.height()) * ratio / dpi,
            forward=False,
        )
        self.draw_idle()

    # ------------------------------------------------------------------ #

    def show_result(self, result: ComputationResult) -> None:
        self._result = result
        self._redraw()

    def set_polarization(self, polarization: str) -> None:
        if polarization not in POLARIZATIONS:
            raise ValueError(f"polarization must be one of {POLARIZATIONS}")
        self._polarization = polarization
        self._redraw()

    def apply_config(self, config: UIConfig) -> None:
        self._cfg = config
        self._redraw()

    def export(self, path: str) -> None:
        """Save the current figure to *path*; format follows the extension."""
        self._fig.savefig(path, dpi=200, facecolor=self._fig.get_facecolor())

    # ------------------------------------------------------------------ #

    def _redraw(self) -> None:
        ax = self._ax
        ax.clear()
        c = self._cfg.theme
        self._fig.set_facecolor(c.card_bg)
        ax.set_facecolor(c.input_bg)
        if self._result is None:
            ax.text(
                0.5, 0.5, "No result yet",
                transform=ax.transAxes,
                ha="center", va="center",
                color=c.text_muted,
                fontsize=self._cfg.base_font_pt + 1,
            )
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            self._draw(ax, self._result)
        self._style_axes(ax)
        self.draw_idle()

    def _style_axes(self, ax) -> None:
        c = self._cfg.theme
        fs = self._cfg.base_font_pt
        ax.tick_params(colors=c.text_secondary, labelsize=fs - 1)
        for spine in ax.spines.values():
            spine.set_color(c.border_light)
        ax.xaxis.label.set_color(c.text_primary)
        ax.yaxis.label.set_color(c.text_primary)
        ax.title.set_color(c.text_primary)
        ax.title.set_fontsize(fs)
        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor(c.input_bg)
            legend.get_frame().set_edgecolor(c.border_light)
            for text in legend.get_texts():
                text.set_color(c.text_primary)
                text.set_fontsize(fs - 1)

    def _draw(self, ax, result: ComputationResult) -> None:
        raise NotImplementedError


class PolarPatternCanvas(ResultCanvas):
    """Polar scattering pattern |S(θ)|, in the style of plot_field_scaterring.

    "Both" shows the diploma-style split: |S_φ| on the [0, π] half of the
    circle, |S_θ| on the other, each arc in its own color.
    """

    polar = True
    margins = dict(left=0.05, right=0.95, bottom=0.06, top=0.86)

    def __init__(self, config: Optional[UIConfig] = None):
        self._scale = SCALE_LINEAR
        super().__init__(config)

    def set_scale(self, scale: str) -> None:
        if scale not in SCALES:
            raise ValueError(f"scale must be one of {SCALES}")
        self._scale = scale
        self._redraw()

    def _draw(self, ax, result: ComputationResult) -> None:
        c = self._cfg.theme
        theta_color = c.accent_calc
        phi_color = c.accent_add
        angles = result.angles

        arcs = []    # (angles, |S| values, color)
        labels = []  # (text, color, axes-x, alignment)
        if self._polarization == POLARIZATION_BOTH:
            split = int(np.searchsorted(angles, np.pi, side="right"))
            arcs.append((angles[:split], np.abs(result.S_ph[:split]), phi_color))
            th_angles = np.concatenate((angles[split - 1:], angles[:1] + 2.0 * np.pi))
            th_values = np.abs(np.concatenate((result.S_th[split - 1:], result.S_th[:1])))
            arcs.append((th_angles, th_values, theta_color))
            labels = [
                ("$|S_{\\phi}|$", phi_color, 0.0, "left"),
                ("$|S_{\\theta}|$", theta_color, 1.0, "right"),
            ]
        else:
            if self._polarization == POLARIZATION_THETA:
                S, color, label = result.S_th, theta_color, "$|S_{\\theta}|$"
            else:
                S, color, label = result.S_ph, phi_color, "$|S_{\\phi}|$"
            arcs.append((
                np.append(angles, angles[0] + 2.0 * np.pi),
                np.abs(np.append(S, S[:1])),
                color,
            ))
            labels = [(label, color, 1.0, "right")]

        for arc_angles, arc_values, color in arcs:
            ax.plot(arc_angles, arc_values, linestyle="-", linewidth=1.5, color=color)

        for text, color, x, ha in labels:
            ax.text(
                x, 1.04, text,
                transform=ax.transAxes,
                ha=ha, va="bottom",
                color=color,
                fontsize=self._cfg.base_font_pt,
            )

        v_max = max(float(arc_values.max()) for _, arc_values, _ in arcs)
        r_max = v_max * 1.1 if v_max > 0 else 1.0
        if self._scale == SCALE_LOG:
            ax.set_rscale("symlog", linthresh=r_max * 1e-4)
        ax.set_ylim(0, r_max)
        ax.set_theta_zero_location("W")
        ax.set_title("Scattering pattern")
        ax.grid(True, color=c.border_light)


class RcsAngleCanvas(ResultCanvas):
    """RCS(θ) in dBm², same convention as plot_radar_cross_section: 10·log10(4πk²|S|²).

    The x axis is the angle from the backscatter direction (0° = backscatter,
    180° = forward), matching the legacy plots.
    """

    polar = False

    def __init__(self, config: Optional[UIConfig] = None):
        self._y_min = -30.0
        self._y_max = 30.0
        super().__init__(config)

    def set_y_range(self, y_min: float, y_max: float) -> None:
        if y_min < y_max:
            self._y_min, self._y_max = y_min, y_max
            self._redraw()

    @staticmethod
    def _rcs_db(S: np.ndarray, k: float) -> np.ndarray:
        values = np.abs(S) ** 2 * 4.0 * np.pi * k**2
        return 10.0 * np.log10(np.maximum(values, _LOG_FLOOR))

    def _draw(self, ax, result: ComputationResult) -> None:
        c = self._cfg.theme
        half = result.angles <= np.pi + 1e-12
        x = 180.0 - np.degrees(result.angles[half])
        order = np.argsort(x)
        x = x[order]

        series = []
        if self._polarization in (POLARIZATION_THETA, POLARIZATION_BOTH):
            series.append(("$S_{\\theta}$", result.S_th, c.accent_calc))
        if self._polarization in (POLARIZATION_PHI, POLARIZATION_BOTH):
            series.append(("$S_{\\phi}$", result.S_ph, c.accent_add))

        for label, S, color in series:
            rcs = self._rcs_db(S[half], result.k)[order]
            ax.plot(x, rcs, linestyle="-", label=label, color=color)

        if self._polarization == POLARIZATION_BOTH:
            ax.legend()

        ax.set_xlim(0, 180)
        ax.set_xticks(np.linspace(0, 180, 7))
        ax.set_ylim(self._y_min, self._y_max)
        ax.set_xlabel("Theta (deg)")
        ax.set_ylabel("dBm²")
        ax.set_title("RCS")
        ax.grid(True, color=c.border_light)
