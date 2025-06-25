

# [Patch v5.5.6] Controlled manual garbage collection utility
import gc
import os
ENABLE_MANUAL_GC = os.getenv("ENABLE_MANUAL_GC", "0") in {"1", "True", "true"}


def maybe_collect():
    """Run :func:`gc.collect` only if ``ENABLE_MANUAL_GC`` env var is truthy."""
    if ENABLE_MANUAL_GC:
        gc.collect()