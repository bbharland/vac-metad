import functools
from pathlib import Path


def savefig(default=None, **save_kwargs):
    """Add a `pngfile` kwarg to a plotting function that returns a Figure.
    pngfile=None skips saving.

    Usage:
    ===============================================
    from .savefig import savefig
    from ..paths import FIGS

    @savefig(FIGS / "vampnet-vacuum.png")
    def plot_training(sd):
        fig, ax = plt.subplots(figsize=(4, 3))
        ...
        return fig

    plot_training(sd)                      # saves to the default
    plot_training(sd, pngfile=None)        # no save
    plot_training(sd, pngfile=FIGS/"x.png")
    """
    save_kwargs.setdefault("bbox_inches", "tight")

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, pngfile=default, **kwargs):
            fig = func(*args, **kwargs)
            if pngfile is not None:
                # Ensure figure parent directory exists
                Path(pngfile).parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(pngfile, **save_kwargs)
            return fig
        return wrapper
    return decorator
