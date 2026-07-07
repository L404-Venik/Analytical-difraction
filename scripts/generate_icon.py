"""Procedurally generate the app icon.

Draws a cross-section of a multilayer sphere with a real computed Mie
scattering pattern |S_θ(θ)| wrapped around it, renders it at every size in
SIZES into app/resources/icon_<size>.png, and packs them into
app/resources/icon.ico.

Run from the repo root:

    python scripts/generate_icon.py

Everything visual lives in the CONFIG block below — tweak and re-run.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analytical_diffraction import BodyParameters, ObservationParameters, calculate_S

OUT_DIR = ROOT / "app" / "resources"

# --------------------------------------------------------------------------- #
# CONFIG — tweak to taste                                                      #
# --------------------------------------------------------------------------- #

SIZES = [16, 24, 32, 48, 64, 128, 256]

# Background plate (rounded square). Set PLATE = False for full transparency.
PLATE = True
PLATE_COLOR = "#16212e"
PLATE_CORNER = 0.18        # corner radius, in [0..1] of half-width
PLATE_MARGIN = 0.02        # gap between plate edge and canvas edge

# The layered sphere: (radius, fill color) from outermost to innermost,
# in canvas units (canvas spans -1..1).
SPHERE_CENTER = (0.0, 0.0)
SPHERE_LAYERS = [
    (0.42, "#0d2f57"),
    (0.33, "#1565c0"),
    (0.24, "#42a5f5"),
    (0.13, "#bbdefb"),
]
SPHERE_EDGE_COLOR = "#7ec8e3"
SPHERE_EDGE_WIDTH = 1.2    # points, at 256 px; scaled down for small sizes

# The body/wavelength whose scattering pattern is drawn (meters).
PATTERN_BODY = BodyParameters(
    eps=[1.0, 2.5 + 0.1j, 1.0],
    r=[1.0, 1.1],
    conducting_core=True,
)
PATTERN_WAVELENGTH = 0.55
PATTERN_POINTS = 720

PATTERN_GAMMA = 0.45       # |S| compression: v ** GAMMA (lower = fatter lobes)
PATTERN_BASE_R = 0.48      # radius where |S| = 0 sits
PATTERN_AMP = 0.42         # radial span of the pattern curve
PATTERN_ROTATION_DEG = 205 # rotates the pattern; main lobe ends up opposite
PATTERN_COLOR = "#5ad16b"
PATTERN_WIDTH = 2.2        # points, at 256 px
PATTERN_FILL_ALPHA = 0.18
PATTERN_GLOW = [(7.0, 0.10), (4.0, 0.18)]   # extra (linewidth, alpha) passes

# --------------------------------------------------------------------------- #


def scattering_curve() -> tuple[np.ndarray, np.ndarray]:
    """Return (theta, radius) of the pattern curve in canvas coordinates."""
    angles = np.linspace(0.0, 2.0 * np.pi, PATTERN_POINTS, endpoint=False)
    observation = ObservationParameters(wavelengths=PATTERN_WAVELENGTH, angles=angles)
    S_th, _ = calculate_S(PATTERN_BODY, observation)
    values = np.abs(S_th[0])
    values = (values / values.max()) ** PATTERN_GAMMA
    theta = angles + np.deg2rad(PATTERN_ROTATION_DEG)
    radius = PATTERN_BASE_R + PATTERN_AMP * values
    # close the loop
    theta = np.append(theta, theta[0])
    radius = np.append(radius, radius[0])
    return theta, radius


def draw(size: int, theta: np.ndarray, radius: np.ndarray) -> Path:
    lw_scale = max(size / 256.0, 0.35)
    fig = plt.figure(figsize=(1.0, 1.0), dpi=size)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")

    if PLATE:
        from matplotlib.patches import FancyBboxPatch

        half = 1.0 - PLATE_MARGIN - PLATE_CORNER
        ax.add_patch(FancyBboxPatch(
            (-half, -half), 2 * half, 2 * half,
            boxstyle=f"round,pad={PLATE_CORNER}",
            facecolor=PLATE_COLOR, edgecolor="none",
        ))

    cx, cy = SPHERE_CENTER
    x = cx + radius * np.cos(theta)
    y = cy + radius * np.sin(theta)
    ax.fill(x, y, color=PATTERN_COLOR, alpha=PATTERN_FILL_ALPHA, zorder=2)
    for glow_width, glow_alpha in PATTERN_GLOW:
        ax.plot(x, y, color=PATTERN_COLOR, alpha=glow_alpha,
                linewidth=glow_width * lw_scale, zorder=3)
    ax.plot(x, y, color=PATTERN_COLOR, linewidth=PATTERN_WIDTH * lw_scale, zorder=4)

    for layer_radius, color in SPHERE_LAYERS:
        ax.add_patch(plt.Circle(
            (cx, cy), layer_radius,
            facecolor=color,
            edgecolor=SPHERE_EDGE_COLOR,
            linewidth=SPHERE_EDGE_WIDTH * lw_scale,
            zorder=5,
        ))

    path = OUT_DIR / f"icon_{size}.png"
    fig.savefig(path, transparent=True)
    plt.close(fig)
    return path


def pack_ico(png_paths: list[Path], out_path: Path) -> None:
    """Pack PNGs into a single .ico (PNG-compressed entries, Vista+)."""
    blobs = [(path, path.read_bytes()) for path in png_paths]
    header = struct.pack("<HHH", 0, 1, len(blobs))
    entries = b""
    offset = len(header) + 16 * len(blobs)
    for path, blob in blobs:
        size = int(path.stem.split("_")[1])
        entries += struct.pack(
            "<BBBBHHII",
            size % 256, size % 256, 0, 0, 1, 32, len(blob), offset,
        )
        offset += len(blob)
    out_path.write_bytes(header + entries + b"".join(blob for _, blob in blobs))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    theta, radius = scattering_curve()
    png_paths = [draw(size, theta, radius) for size in SIZES]
    pack_ico(png_paths, OUT_DIR / "icon.ico")
    for path in [*png_paths, OUT_DIR / "icon.ico"]:
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
