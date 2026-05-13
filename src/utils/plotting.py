"""Shared plotting utilities — style initialisation and figure persistence."""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def setup_style() -> None:
    """Apply a consistent visual theme to all figures in this session."""
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
    plt.rcParams.update({
        "figure.dpi":    120,
        "savefig.bbox":  "tight",
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })


def save_fig(fig: plt.Figure, output_dir: Path, filename: str) -> None:
    """
    Save *fig* to *output_dir/filename* then close it.
    Closing immediately after saving prevents memory leaks in long pipelines.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / filename
    fig.savefig(dest)
    plt.close(fig)
    print(f"  [plot] saved -> {dest.relative_to(dest.parents[3])}")
