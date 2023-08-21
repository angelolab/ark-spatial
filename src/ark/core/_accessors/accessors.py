from collections import OrderedDict

import spatialdata as sd
from anndata import AnnData
from dask.dataframe.core import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from multiscale_spatial_image.multiscale_spatial_image import MultiscaleSpatialImage
from spatial_image import SpatialImage
from xarray.core.extensions import _register_accessor


def register_spatial_data_accessor(name: str) -> sd.SpatialData:
    """Hijacks xarray _register_accessor to register a SpatialData accessor.

    Source: github/scverse/spatialdata-plot/src/spatialdata_plot/_accessor.py

    Parameters
    ----------
    name : str
        Name under which the accessor should be registered. A warning is issued
        if this name conflicts with a preexisting attribute.

    See Also
    --------
    register_dataset_accessor
    """
    return _register_accessor(name, sd.SpatialData)


class SpatialDataAccessor:
    @property
    def sdata(self) -> sd.SpatialData:
        """The `SpatialData` object to provide preprocessing functions for."""
        return self._sdata

    @sdata.setter
    def sdata(self, sdata: sd.SpatialData) -> None:
        self._sdata = sdata

    def __init__(self, sdata: sd.SpatialData) -> None:
        self._sdata = sdata

    def _copy(
        self,
        images: None | dict[str, SpatialImage | MultiscaleSpatialImage] = None,
        labels: None | dict[str, SpatialImage | MultiscaleSpatialImage] = None,
        points: None | dict[str, DaskDataFrame] = None,
        shapes: None | dict[str, GeoDataFrame] = None,
        table: None | AnnData = None,
    ) -> sd.SpatialData:
        """Copy the references from the original to the new SpatialData object."""
        sdata = sd.SpatialData(
            images=self._sdata.images if images is None else images,
            labels=self._sdata.labels if labels is None else labels,
            points=self._sdata.points if points is None else points,
            shapes=self._sdata.shapes if shapes is None else shapes,
            table=self._sdata.table if table is None else table,
        )
        sdata.plotting_tree = (
            self._sdata.plotting_tree if hasattr(self._sdata, "plotting_tree") else OrderedDict()
        )

        return sdata