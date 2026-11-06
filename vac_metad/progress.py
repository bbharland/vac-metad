"""Optional tqdm progress bar, with a passthrough when it is unwanted or absent.  Also allow an explicit quiet mode with kwarg usetqdm=False.


usage:

    from .progress import progress

    def f(args, usetqdm=True):
        for n in progress(range(N), usetqdm, desc="Iterating"):
            ...
"""

try:
    from tqdm.auto import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):
        return iterable


def progress(iterable, usetqdm=True, **kwargs):
    """Wrap *iterable* in a tqdm bar when *usetqdm*, otherwise return it."""
    return tqdm(iterable, **kwargs) if usetqdm else iterable
