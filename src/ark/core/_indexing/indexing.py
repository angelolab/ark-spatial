from collections import OrderedDict
from typing import Any

import natsort as ns
import spatialdata as sd
from torchdata.dataloader2 import DataLoader2
from torchdata.datapipes.iter import IterableWrapper, IterDataPipe

from ark.core._accessors import (
    SpatialDataAccessor,
    register_spatial_data_accessor,
)

from .utils import (
    _get_coordinate_system_mapping,
    _get_region_key,
)

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict


@register_spatial_data_accessor("sel")
class IndexingAccessor(SpatialDataAccessor):
    """
    The indexing accessor for SpatialData objects.

    Parameters
    ----------
    sdata :
        A spatial data object.
    """

    def _verify_plotting_tree_exists(self) -> None:
        if not hasattr(self._sdata, "plotting_tree"):
            self._sdata.plotting_tree = OrderedDict()

    def __call__(self, *args: Any, **kwds: Any) -> sd.SpatialData:
        return self._get_elements(*args, **kwds)

    def _get_elements(self, elements: str | list[str]) -> sd.SpatialData:
        """
        Get a subset of the spatial data object by specifying elements to keep.

        Parameters
        ----------
        elements :
            A string or a list of strings specifying the elements to keep.
            Valid element types are:

            - 'coordinate_systems'
            - 'images'
            - 'labels'
            - 'shapes'

        Returns
        -------
        sd.SpatialData
            A new spatial data object containing only the specified elements.

        Raises
        ------
        TypeError
            If `elements` is not a string or a list of strings.
            If `elements` is a list of strings but one or more of the strings
            are not valid element types.

        ValueError
            If any of the specified elements is not present in the original
            spatialdata object.

        AssertionError
            If `label_keys` is not an empty list but the spatial data object
            does not have a table or the table does not have 'uns' or 'obs'
            attributes.

        Notes
        -----
        If the original spatialdata object has a table, and `elements`
        includes label keys, the returned spatialdata object will have a
        subset of the original table with only the rows corresponding to the
        specified label keys. The `region` attribute of the returned spatial
        data object's table will be set to the list of specified label keys.

        If the original spatial data object has no table, or if `elements` does
        not include label keys, the returned spatialdata object will have no
        table.
        """
        if not isinstance(elements, str | list):
            raise TypeError("Parameter 'elements' must be a string or a list of strings.")

        if not all(isinstance(e, str) for e in elements):
            raise TypeError("When parameter 'elements' is a list, all elements must be strings.")

        if isinstance(elements, str):
            elements = [elements]

        coord_keys = []
        image_keys = []
        label_keys = []
        shape_keys = []
        point_keys = []

        # prepare list of valid keys to sort elements on
        valid_coord_keys = (
            self._sdata.coordinate_systems if hasattr(self._sdata, "coordinate_systems") else None
        )
        valid_image_keys = (
            list(self._sdata.images.keys()) if hasattr(self._sdata, "images") else None
        )
        valid_label_keys = (
            list(self._sdata.labels.keys()) if hasattr(self._sdata, "labels") else None
        )
        valid_shape_keys = (
            list(self._sdata.shapes.keys()) if hasattr(self._sdata, "shapes") else None
        )
        valid_point_keys = (
            list(self._sdata.points.keys()) if hasattr(self._sdata, "points") else None
        )

        # first, extract coordinate system keys because they generate implicit keys
        mapping = _get_coordinate_system_mapping(self._sdata)
        implicit_keys = []
        for e in elements:
            for valid_coord_key in valid_coord_keys:
                if (valid_coord_keys is not None) and (e == valid_coord_key):
                    coord_keys.append(e)
                    implicit_keys += mapping[e]

        for e in elements + implicit_keys:
            found = False

            if valid_coord_keys is not None:
                for valid_coord_key in valid_coord_keys:
                    if e == valid_coord_key:
                        coord_keys.append(e)
                        found = True

            if valid_image_keys is not None:
                for valid_image_key in valid_image_keys:
                    if e == valid_image_key:
                        image_keys.append(e)
                        found = True

            if valid_label_keys is not None:
                for valid_label_key in valid_label_keys:
                    if e == valid_label_key:
                        label_keys.append(e)
                        found = True

            if valid_shape_keys is not None:
                for valid_shape_key in valid_shape_keys:
                    if e == valid_shape_key:
                        shape_keys.append(e)
                        found = True

            if valid_point_keys is not None:
                for valid_point_key in valid_point_keys:
                    if e == valid_point_key:
                        point_keys.append(e)
                        found = True

            if not found:
                msg = f"Element '{e}' not found. Valid choices are:"
                if valid_coord_keys is not None:
                    msg += "\n\ncoordinate_systems\n├ "
                    msg += "\n├ ".join(valid_coord_keys)
                if valid_image_keys is not None:
                    msg += "\n\nimages\n├ "
                    msg += "\n├ ".join(valid_image_keys)
                if valid_label_keys is not None:
                    msg += "\n\nlabels\n├ "
                    msg += "\n├ ".join(valid_label_keys)
                if valid_shape_keys is not None:
                    msg += "\n\nshapes\n├ "
                    msg += "\n├ ".join(valid_shape_keys)
                raise ValueError(msg)

        # copy that we hard-modify
        sdata = self._copy()

        if (valid_coord_keys is not None) and (len(coord_keys) > 0):
            sdata = sdata.filter_by_coordinate_system(coord_keys, filter_table=False)

        elif len(coord_keys) == 0:
            if valid_image_keys is not None:
                if len(image_keys) == 0:
                    for valid_image_key in valid_image_keys:
                        del sdata.images[valid_image_key]
                elif len(image_keys) > 0:
                    for valid_image_key in valid_image_keys:
                        if valid_image_key not in image_keys:
                            del sdata.images[valid_image_key]

            if valid_label_keys is not None:
                if len(label_keys) == 0:
                    for valid_label_key in valid_label_keys:
                        del sdata.labels[valid_label_key]
                elif len(label_keys) > 0:
                    for valid_label_key in valid_label_keys:
                        if valid_label_key not in label_keys:
                            del sdata.labels[valid_label_key]

            if valid_shape_keys is not None:
                if len(shape_keys) == 0:
                    for valid_shape_key in valid_shape_keys:
                        del sdata.shapes[valid_shape_key]
                elif len(shape_keys) > 0:
                    for valid_shape_key in valid_shape_keys:
                        if valid_shape_key not in shape_keys:
                            del sdata.shapes[valid_shape_key]

            if valid_point_keys is not None:
                if len(point_keys) == 0:
                    for valid_point_key in valid_point_keys:
                        del sdata.points[valid_point_key]
                elif len(point_keys) > 0:
                    for valid_point_key in valid_point_keys:
                        if valid_point_key not in point_keys:
                            del sdata.points[valid_point_key]

        # subset table if it is present and the region key is a valid column
        if sdata.table is not None and len(shape_keys + label_keys) > 0:
            assert hasattr(sdata, "table"), "SpatialData object does not have a table."
            assert hasattr(sdata.table, "uns"), "Table in SpatialData object does not have 'uns'."
            assert hasattr(sdata.table, "obs"), "Table in SpatialData object does not have 'obs'."

            # create mask of used keys
            mask = sdata.table.obs[_get_region_key(sdata)]
            mask = list(mask.str.contains("|".join(shape_keys + label_keys)))

            # create copy and delete original so we can reuse slot
            old_table = sdata.table.copy()
            new_table = old_table[mask, :].copy()
            new_table.uns["spatialdata_attrs"]["region"] = list(
                set(new_table.obs[_get_region_key(sdata)])
            )
            del sdata.table
            sdata.table = new_table

        else:
            del sdata.table

        return sdata


class IterableWrapperKwargs(TypedDict):
    deepcopy: bool


@register_spatial_data_accessor("iter_coords")
class IteratorAccessor(SpatialDataAccessor):
    """An accessor used for iterating over coordinates in a `SpatialData` object. Sorted by natsort.

    Excludes the "global" coordinate system.

    Parameters
    ----------
    SpatialDataAccessor : SpatialDataAccessor
        The base `SpatialData` accessor class.

    Yields
    ------
    sd.SpatialData
        The spatial data object for the filtered by the current coordinate.
    """

    @property
    def fovs(self) -> list[str]:
        all_coords = filter(lambda c: c != "global", self.sdata.coordinate_systems.copy())
        fovs: list[str] = ns.natsorted(all_coords)
        return fovs

    def __call__(
        self, dataloader: bool, **iterable_wrapper_kwargs
    ) -> IterDataPipe[tuple[str, sd.SpatialData]] | DataLoader2[tuple[str, sd.SpatialData]]:
        iw = IterableWrapper(
            [(fov, self.sdata.sel(fov)) for fov in self.fovs], **iterable_wrapper_kwargs
        )
        if dataloader:
            return DataLoader2(iw)
        else:
            return iw
