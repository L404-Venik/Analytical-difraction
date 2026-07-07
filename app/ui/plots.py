from __future__ import annotations

from typing import Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from app.application.computation import ComputationResult

from .ui_config import UIConfig

POLARIZATION_THETA = "S_θ"
POLARIZATION_PHI = "S_φ"
POLARIZATION_BOTH = "Both"
POLARIZATIONS = [POLARIZATION_THETA, POLARIZATION_PHI, POLARIZATION_BOTH]

_LOG_FLOOR = 1e-300


class ResultCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas that renders one view of a ComputationResult."""

    polar = False

    def __init__(self, config: Optional[UIConfig] = None):
        self._fig = Figure(figsize=(4.0, 3.5), layout="constrained")
        super().__init__(self._fig)
        self._cfg = config or UIConfig()
        self._result: ComputationResult | None = None
        self._polarization = POLARIZATION_BOTH
        projection = "polar" if self.polar else None
        self._ax = self._fig.add_subplot(111, projection=projection)
        self._redraw()

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


SCALE_LINEAR = "Linear"
SCALE_LOG = "Log"
SCALES = [SCALE_LINEAR, SCALE_LOG]


class PolarPatternCanvas(ResultCanvas):
    """Polar scattering pattern |S(θ)|, in the style of plot_field_scaterring."""

    polar = True

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
        theta = np.append(result.angles, result.angles[0] + 2.0 * np.pi)

        series = []
        if self._polarization in (POLARIZATION_THETA, POLARIZATION_BOTH):
            series.append(("$|S_{\\theta}|$", result.S_th, c.accent_calc))
        if self._polarization in (POLARIZATION_PHI, POLARIZATION_BOTH):
            series.append(("$|S_{\\phi}|$", result.S_ph, c.accent_add))

        r_max = 1.0
        for label, S, color in series:
            values = np.abs(S)
            values = np.append(values, values[0])
            r_max = max(r_max, float(values.max()) * 1.05)
            ax.plot(theta, values, linestyle="-", linewidth=1.5, label=label, color=color)

        if self._polarization == POLARIZATION_BOTH:
            ax.legend(loc="upper left", bbox_to_anchor=(-0.25, 1.12))

        if self._scale == SCALE_LOG:
            ax.set_rscale("symlog", linthresh=r_max * 1e-4)
            ax.set_ylim(0, r_max)
        else:
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
