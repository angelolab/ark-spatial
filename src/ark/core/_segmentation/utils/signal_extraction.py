"""Signal extraction module

Currently the agg methods in `sdata.aggregate` which get called in `xarray-spatial.zonal_stats` do not
support dask backed arrays (i.e.) we can't pass custom functions to the `agg` method. Unfortunate
"""


import inspect
from functools import partial

from spatial_image import SpatialImage
from spatialdata.models import X, Y


def signal_extraction_wrapper(func):
    def wrapper(**kwargs):
        func_kwargs = inspect.signature(func).bind_partial(**kwargs).kwargs
        f = partial(func, **func_kwargs)
        return f

    return wrapper


@signal_extraction_wrapper
def positive_pixels_extraction(fov: SpatialImage, threshold: float = 0) -> SpatialImage:
    f = fov.load().where(fov > threshold, 0).sum(dim=(Y, X)).values()
    return f
