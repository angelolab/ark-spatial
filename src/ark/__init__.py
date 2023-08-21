from importlib.metadata import version

from ray.util.dask import enable_dask_on_ray

from . import pl, pp, tl
from .core import indexing as idx
from .core import io

__all__ = ["pl", "pp", "tl", "io", "idx"]

__version__ = version("ark")


# Set ray as the default scheduler for dask
enable_dask_on_ray()
