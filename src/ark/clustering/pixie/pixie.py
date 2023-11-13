import natsort as ns
import spatialdata as sd

from ark.core._accessors import (
    SpatialDataAccessor,
    register_spatial_data_accessor,
)

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict


@register_spatial_data_accessor("pixie")
class PixieAccessor(SpatialDataAccessor):
    """
    Accessor for Pixie clustering.
    """

    pass
