"""Signal extraction module.

Currently the agg methods in `sdata.aggregate` which get called in `xarray-spatial.zonal_stats` do not
support dask backed arrays (i.e.) we can't pass custom functions to the `agg` method. Unfortunate.
"""


import inspect
from collections.abc import Callable
from functools import partial

from spatial_image import SpatialImage
from spatialdata.models import X, Y


def signal_extraction_wrapper(func: Callable):
    """Wraps a signal extraction function to allow for partial keyword arguments.

    Parameters
    ----------
    func : Callable
        The original function to wrap.
    """

    def wrapper(**kwargs):
        func_kwargs = inspect.signature(func).bind_partial(**kwargs).kwargs
        f = partial(func, **func_kwargs)
        return f

    return wrapper


@signal_extraction_wrapper
def positive_pixels_extraction(fov: SpatialImage, threshold: float = 0) -> SpatialImage:
    """Only extracts pixels above a threshold.

    Parameters
    ----------
    fov : SpatialImage
        The fov to extract pixels from.
    threshold : float, optional
        The minimum threshold to filter out pixels, by default 0

    Returns
    -------
    SpatialImage
        The summed extracted pixels.
    """
    f: SpatialImage = fov.where(fov > threshold, 0).sum(dim=(Y, X))
    return f
