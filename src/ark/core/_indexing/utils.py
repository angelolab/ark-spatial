import spatialdata as sd
from spatialdata.models import TableModel
from spatialdata.transformations import get_transformation


def _get_region_key(sdata: sd.SpatialData) -> str:
    """Quick access to the data's region key.

    Parameters
    ----------
    sdata : sd.SpatialData
        The `SpatialData` object to get the region key from.

    Returns
    -------
    str
        The region key.
    """
    return str(sdata.table.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY_KEY])


def _get_instance_key(sdata: sd.SpatialData) -> str:
    """Quick access to the data's instance key.

    Parameters
    ----------
    sdata : sd.SpatialData
        The `SpaitalData` object to get the instance key from.

    Returns
    -------
    str
        The instance key.
    """
    return str(sdata.table.uns[TableModel.ATTRS_KEY][TableModel.INSTANCE_KEY])


def _get_coordinate_system_mapping(sdata: sd.SpatialData) -> dict[str, list[str]]:
    """Gets the coordinate systems and their associated data in a `SpatialData` object.

    Parameters
    ----------
    sdata : sd.SpatialData
        The `SpatialData` object to get the coordinate systems from.

    Returns
    -------
    dict[str, list[str]]
        A dictionary of coordinate systems and their associated data.

    Raises
    ------
    ValueError
        When the `SpatialData` object does not have at least one coordinate system.
    """
    coordsys_keys = sdata.coordinate_systems
    image_keys = [] if sdata.images is None else sdata.images.keys()
    label_keys = [] if sdata.labels is None else sdata.labels.keys()
    shape_keys = [] if sdata.shapes is None else sdata.shapes.keys()
    point_keys = [] if sdata.points is None else sdata.points.keys()

    mapping: dict[str, list[str]] = {}

    if len(coordsys_keys) < 1:
        raise ValueError(
            "SpatialData object must have at least one coordinate system to generate a mapping."
        )

    for key in coordsys_keys:
        mapping[key] = []

        for image_key in image_keys:
            transformations = get_transformation(sdata.images[image_key], get_all=True)

            if key in list(transformations.keys()):
                mapping[key].append(image_key)

        for label_key in label_keys:
            transformations = get_transformation(sdata.labels[label_key], get_all=True)

            if key in list(transformations.keys()):
                mapping[key].append(label_key)

        for shape_key in shape_keys:
            transformations = get_transformation(sdata.shapes[shape_key], get_all=True)

            if key in list(transformations.keys()):
                mapping[key].append(shape_key)

        for point_key in point_keys:
            transformations = get_transformation(sdata.points[point_key], get_all=True)

            if key in list(transformations.keys()):
                mapping[key].append(point_key)

    return mapping
