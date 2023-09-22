from importlib.metadata import version

from dask.distributed import Client
from ray.util.dask import enable_dask_on_ray

from .core import indexing as idx
from .core import io

__all__ = ["io", "idx", "client", "address"]

__version__ = version("ark")


# Set the local cluster
client = Client()
address = client.scheduler.address

# Set ray as the default scheduler for dask
enable_dask_on_ray()
