from joblib import Memory

from .paths import DATA

memory = Memory(DATA / "cache")

def nbcache(name, verbose=0):
    """Factory for memory.cache decorator used in notebooks.  By default, the @memory.cache decorator creates a cache file that includes a kernel session id, which is not stable across kernel restarts.  This fixes this problem.

    Usage:
        @nbcache("subdir") # cache => data/cache/notebooks/subdir/analysis/
        def analysis(args):
            return SimpleNamespace(...)

    Force a recompute:
        analysis.clear()       # wipe this function's cache
        analysis.call(args)    # recompute and overwrite, for these args only

    verbose=1 prints "[Memory] Calling ..." whenever it computes.  Cache-read is silent.
    """
    def deco(f):
        f.__module__ = f"notebooks.{name}"
        return memory.cache(f, verbose=verbose)
    return deco
