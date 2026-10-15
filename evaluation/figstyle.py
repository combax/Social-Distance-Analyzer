"""One look for every figure: hand-drawn lines on GitHub's dark background.

One PNG per figure, background baked in, so it renders the same in any markdown viewer; on
GitHub's dark theme it blends with the page, on the light theme it reads as a dark card.
"""

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import matplotlib
from matplotlib.axes import Axes
from matplotlib.figure import Figure

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)  # fallback fonts are fine

BLUE, ORANGE, AQUA, YELLOW, PINK = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
PALETTE = [BLUE, ORANGE, AQUA, YELLOW, PINK]
BACKGROUND = "#0d1117"  # GitHub dark
THEMES: dict[str, dict[str, str]] = {
    "dark": {"ink": "#e6edf3", "ink2": "#9da7b1", "muted": "#6e7781", "grid": "#30363d"},
}
HAND_FONTS = ["xkcd Script", "Humor Sans", "Comic Neue", "Comic Sans MS", "Segoe Print"]


@contextmanager
def theme(name: str) -> Iterator[dict[str, str]]:
    """Apply the hand-drawn style with the ``name`` theme's ink colours; yields those colours."""
    c = THEMES[name]
    with plt.xkcd(scale=0.6, length=120, randomness=1.5):
        plt.rcParams.update(
            {
                "font.family": HAND_FONTS + ["DejaVu Sans"],
                "font.size": 9,
                "text.color": c["ink"],
                "axes.labelcolor": c["ink2"],
                "axes.titlecolor": c["ink"],
                "axes.edgecolor": c["muted"],
                "axes.linewidth": 1.2,
                "xtick.color": c["ink2"],
                "ytick.color": c["ink2"],
                "xtick.labelsize": 8,
                "ytick.labelsize": 8,
                "grid.color": c["grid"],
                "grid.linewidth": 0.8,
                "legend.frameon": False,
                "figure.facecolor": BACKGROUND,
                "axes.facecolor": BACKGROUND,
                "savefig.facecolor": BACKGROUND,
                "path.effects": [],  # xkcd's white halo around lines would show on dark pages
            }
        )
        yield c


def axes(
    width: float, height: float, grid: Literal["x", "y", "both"] = "x"
) -> tuple[Figure, Axes]:
    """A figure and axes with only the left and bottom spines, and a faint grid behind the data."""
    fig, ax = plt.subplots(figsize=(width, height), dpi=150)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis=grid, zorder=0)
    return fig, ax


def render(draw: Callable[[dict[str, str]], Figure], name: str, fig_dir: Path) -> str:
    """Draw ``name`` into ``fig_dir`` and return the markdown that embeds it.

    Args:
        draw: Takes the theme's colours, returns a matplotlib Figure.
        name: File name, e.g. ``speed.png``.
        fig_dir: Output directory (``results/figures``).
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    with theme("dark") as colours:
        fig = draw(colours)
        fig.savefig(fig_dir / name, dpi=150, bbox_inches="tight", pad_inches=0.25)
        plt.close(fig)
    return picture(name)


def picture(name: str, alt: str = "") -> str:
    """Markdown image for ``results/figures/<name>``."""
    return f"![{alt or name}](results/figures/{name})"
