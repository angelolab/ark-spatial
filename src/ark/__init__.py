from importlib.metadata import version

from .core import indexing as idx
from .core import io

__all__ = ["io", "idx"]

__version__ = version("ark")
