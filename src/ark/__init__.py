from importlib.metadata import version

from . import pl, pp, tl
from .core import indexing as idx
from .core import io

__all__ = ["pl", "pp", "tl", "io", "idx"]

__version__ = version("ark")
