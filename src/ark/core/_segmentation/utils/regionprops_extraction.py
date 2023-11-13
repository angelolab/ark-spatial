"""
`regionprops_extraction.py`.

From https://github.com/jrussell25/dask-regionprops/blob/main/dask_regionprops/regionprops.py,
Modified to accept other properties.
"""

import inspect
import re
from collections.abc import Callable, Hashable, Mapping
from itertools import product
import dask.dataframe as dd
import numpy as np
import pandas as pd
import xarray as xr
from dask import delayed
from numpy.typing import ArrayLike
from skimage.measure import label, moments, regionprops_table
from spatial_image import SpatialImage
from spatialdata.models import X, Y

REGIONPROPS_BASE: list[str] = [
    "label",
    "area",
    "area_convex",
    "centroid",
    "eccentricity",
    "equivalent_diameter",
    "axis_minor_length",
    "axis_major_length",
    "perimeter",
]

REGIONPROPS_BASE_TEMPORARY: list[str] = [
    "image",
    "image_convex",
]

REGIONPROPS_SINGLE_COMP: list[str] = [
    "major_minor_axis_ratio",
    "perim_square_over_area",
    "major_axis_equiv_diam_ratio",
    "convex_hull_equiv_diam_ratio",
    "centroid_dif",
    "num_concavities",
]

REGIONPROPS_MULTI_COMP: list[str] = ["nc_ratio"]

DEFAULT_REGIONPROPS: list[str] = [
    *REGIONPROPS_BASE,
    *REGIONPROPS_SINGLE_COMP
    # *REGIONPROPS_MULTI_COMP,
]

REGIONPROPS_NAMES = [
    "instance_id",
    "area",
    "area_convex",
    f"{X}_centroid",
    f"{Y}_centroid",
    "eccentricity",
    "equivalent_diameter",
    "axis_minor_length",
    "axis_major_length",
    "perimeter",
    *REGIONPROPS_SINGLE_COMP,
]


def ufunc_regionprops(
    labels: SpatialImage, dim: list[str], derived_props: list[str] = REGIONPROPS_SINGLE_COMP
) -> dd.DataFrame:
    rp_df: dd.DataFrame = (
        xr.apply_ufunc(
            regionprops,
            labels,
            vectorize=True,
            input_core_dims=[dim],
            output_core_dims=[["label", "properties"]],
            kwargs={"intensity": None, "derived_properties": derived_props},
            dask="allowed",
        )
        .to_pandas()
        .reset_index(drop=True, inplace=False)
        .rename(
            columns={i: rp_name for i, rp_name in enumerate([*REGIONPROPS_NAMES])}, inplace=False
        )
        .astype({"instance_id": int, "num_concavities": int})
    )
    rp_df.index = rp_df.index.astype("str")
    return rp_df


@delayed
def compute_region_props_df(
    labels: ArrayLike,
    intensity: ArrayLike | None = None,
    properties: list[str, ...] = REGIONPROPS_BASE,
    derived_properties: Mapping[str, ArrayLike] = {},
    other_cols: Mapping[str, ArrayLike] = None,
) -> pd.DataFrame:
    """Computes the regionprops of each labelled region in a segmentation image.

    Lightly wrap skimage.measure.regionprops_table to return a DataFrame.
    Also allow for the addition of extra columns, used in regionprops to track
    non-core dimensions of input.

    Parameters
    ----------
    labels : ArrayLike
        Array like container of labelled regions. Background is assumed to have a value of 0.
    intensity : ArrayLike | None, optional
        Optional intensity field to compute weighted region properties from, by default None
    properties : tuple[str, ...], optional
        Properties to compute for each region. Computes all properties which return fixed sized outputs. If provided
        an intensity image, corresponding weighted properties will also be computed by default, by default REGIONPROPS_BASE
    derived_properties : Mapping[str, ArrayLike], optional
        A list of custom region properties which are calculated from the original set of region properties, by default {}
    other_cols : Mapping[str, ArrayLike], optional
        Other columns in the DataFrame, by default None

    Returns
    -------
    pd.DataFrame
        Returns the regionprops as a DataFrame
    """
    if other_cols is None:
        other_cols = {}
    rp_table: dict[str, ArrayLike] = regionprops_table(
        labels, intensity, properties=properties + REGIONPROPS_BASE_TEMPORARY
    )

    # Compute REGIONPROPS_SINGLE_COMP
    for prop in derived_properties:
        rp_table[prop] = np.array(REGIONPROPS_FUNCTIONS[prop](rp_table))

    # Drop "image_convex" and "image"
    for prop in REGIONPROPS_BASE_TEMPORARY:
        if prop in rp_table:
            rp_table.pop(prop)

    df = pd.DataFrame(rp_table)
    for k, v in other_cols.items():
        df[k] = v
    return df


def regionprops(
    labels: SpatialImage,
    intensity: SpatialImage | None = None,
    properties: list[str, ...] = REGIONPROPS_BASE,
    derived_properties: list[str, ...] = REGIONPROPS_SINGLE_COMP,
    core_dims: list[int | str, ...] | None = None,
) -> dd.DataFrame:
    """
    Creates the delayed DataFrame and prepares the Dask execution graph for lazy execution.

    Sets the Dask execution graph to loop over the Delayed DataFrames and compute the regionprops for
    each labelled image in each frame.

    Parameters
    ----------
    labels : SpatialImage
        The SpatialImage of labelled regions. Background is assumed to have a value of 0.
    intensity : SpatialImage | None, optional
        A SpatialImage of the intensity field to compute weighted region proeprties from, by default None
    properties : tuple[str, ...], optional
        Properties to compute for each region. Computes all properties which return fixed sized outputs. If provided
        an intensity image, corresponding weighted proeprteis will also be computed by default, by default REGIONPROPS_BASE
    derived_properties : Mapping[str, ArrayLike], optional
        A list of custom region properties which are calculated from the orginal set of region properties, by default {}
    core_dims : tuple[int] or tuple[str] default None
        Dimensions of input arrays that correspond to spatial (xy) dimensions of each
        image. If None, it is assumed that the final two dimensions are the
        spatial dimensions.

    Returns
    -------
    regionprops_df : dask.DataFrame
        Lazily constructed dataframe containing columns for each specified
        property.
    """

    loop_sizes = _get_loop_sizes(labels, core_dims)

    if intensity is not None:
        properties = DEFAULT_REGIONPROPS

    meta = _get_meta(loop_sizes, properties, derived_properties)

    all_props = []

    for dims in product(*(range(v) for v in loop_sizes.values())):
        other_cols = dict(zip(loop_sizes.keys(), dims, strict=True))

        if intensity is not None:
            frame_props = compute_region_props_df(
                labels[dims], intensity[dims], properties, derived_properties, other_cols
            )
        else:
            frame_props = compute_region_props_df(
                labels[dims], None, properties, derived_properties, other_cols
            )

        all_props.append(frame_props)

    divisions = range(int(np.prod(tuple(loop_sizes.values())) + 1))

    cell_props = dd.from_delayed(all_props, meta=meta, divisions=divisions)

    return cell_props


DEFAULT_META = pd.DataFrame(
    regionprops_table(
        label_image=np.ones((1, 1), dtype="uint32"),
        intensity_image=np.ones((1, 1)),
        properties=REGIONPROPS_BASE,
    )
)

DEFAULT_PROPS_TO_COLS = {}

for prop in DEFAULT_REGIONPROPS:
    col_list = []
    for c in DEFAULT_META.columns:
        stripped = re.sub("[-0-9]", "", c)
        if stripped == prop:
            col_list.append(c)
    DEFAULT_PROPS_TO_COLS[prop] = col_list


def _get_meta(
    loop_sizes: dict[Hashable, list[int]], properties: list[str], derived_properties: list[str]
) -> pd.DataFrame:
    """Generates the meta DataFrame for the Dask DataFrame.

    Parameters
    ----------
    loop_sizes : dict[Hashable, list[int]
        The number of regions in the image.
    properties : list[str]
        The properties to compute for each region.
    derived_properties : list[str]
        The derived properties to compute for each region.

    Returns
    -------
    pd.DataFrame
        The meta DataFrame for the Dask DataFrame.
    """
    meta = pd.DataFrame()
    for prop in properties:
        meta = meta.join(DEFAULT_META[DEFAULT_PROPS_TO_COLS[prop]], how="outer")
    if derived_properties:
        for prop in derived_properties:
            meta[prop] = [np.nan]

    other_cols = pd.DataFrame(columns=list(loop_sizes.keys()), dtype=int)

    return meta.join(other_cols)


def _get_loop_sizes(
    labels: SpatialImage, core_dims: tuple[int | str, ...] | None
) -> dict[Hashable, list[int | str]]:
    """Gets the loop sizes for the Dask DataFrame.

    Parameters
    ----------
    labels : SpatialImage
        The labels to compute the regionprops for.
    core_dims : tuple[int  |  str, ...] | None
        Dimensions of input arrays that correspond to spatial (xy) dimensions of each
        image. If None, it is assumed that the final two dimensions are the
        spatial dimensions.

    Returns
    -------
    dict[Hashable, list[int]]
        Returns the loop sizes for the Dask DataFrame for each dimension.
    """
    if isinstance(labels, xr.DataArray):
        if core_dims is None:
            loop_sizes = {
                labels.dims[i]: labels.sizes[labels.dims[i]] for i in range(labels.ndim - 2)
            }
        elif isinstance(core_dims[0], str):
            loop_dims = set(labels.dims) - set(core_dims)
            loop_sizes = {d: labels.sizes[d] for d in loop_dims}
        elif isinstance(core_dims[0], int):
            pos_core_dims = _get_pos_core_dims(core_dims, labels.ndim)
            loop_dims = set(range(labels.ndim)) - set(pos_core_dims)
            loop_sizes = {labels.dims[i]: labels.shape[i] for i in pos_core_dims}

    else:
        if core_dims is None:
            loop_shape = labels.shape[:-2]
            loop_sizes = {f"dim-{i}": v for i, v in enumerate(loop_shape)}

        else:
            pos_core_dims = _get_pos_core_dims(core_dims, labels.ndim)
            loop_dims = set(range(labels.ndim)) - set(pos_core_dims)
            loop_shape = (labels.shape[d] for d in loop_dims)

            loop_sizes = {f"dim-{i}": v for i, v in enumerate(loop_shape)}

    return loop_sizes


def _get_pos_core_dims(core_dims: list, ndim: int) -> tuple[int]:
    """Gets the positive core dimensions.

    Parameters
    ----------
    core_dims : list
        The core dimensions.
    ndim : int
        The number of dimensions.

    Returns
    -------
    tuple[int]
        The positive core dimensions.
    """
    pos_core_dims = []
    for d in core_dims:
        if d < 0:
            pos = ndim + d
            pos_core_dims.append(pos)
        else:
            pos_core_dims.append(d)
    return tuple(pos_core_dims)


def rp_table_wrapper(func: Callable) -> Callable:
    """Wraps a function to extract a set of regionprops necessary for the function.

    Parameters
    ----------
    func : Callable
        The function to wrap

    Returns
    -------
    Callable
        The wrapped function with the necessary regionprops extracted.
    """

    def wrapper(region_properties: dict[str, ArrayLike], **kwargs) -> Callable:
        props: list[str] = inspect.getargs(func.__code__).args
        kwargs.update({k: region_properties[k] for k in props if k in region_properties})

        return func(**kwargs)

    return wrapper


@rp_table_wrapper
def major_minor_axis_ratio(axis_minor_length: ArrayLike, axis_major_length: ArrayLike) -> ArrayLike:
    """Computes the major axis length divided by the minor axis length.

    Parameters
    ----------
    axis_minor_length : ArrayLike
        The container of the minor axis length of each region.
    axis_major_length : ArrayLike
        The container of the major axis length of each region.

    Returns
    -------
    ArrayLike
        The major axis length divided by the minor axis length.
    """
    mmar: ArrayLike = np.divide(
        axis_major_length,
        axis_minor_length,
        out=np.full_like(a=axis_minor_length, fill_value=np.nan, dtype=np.float64),
        where=axis_minor_length != 0,
    )
    return mmar


@rp_table_wrapper
def perim_square_over_area(perimeter: ArrayLike, area: ArrayLike) -> ArrayLike:
    """Calculates the perimeter squared divided by the area of the region.

    Parameters
    ----------
    perimeter : ArrayLike
        The perimeter of each region.
    area : ArrayLike
        The area of each region.

    Returns
    -------
    ArrayLike
        The perimeter squared divided by the area of the region.
    """
    psoa: ArrayLike = perimeter**2 / area
    return psoa


@rp_table_wrapper
def major_axis_equiv_diam_ratio(
    axis_major_length: ArrayLike, equivalent_diameter: ArrayLike
) -> ArrayLike:
    """Calculates the ratio between the major axis length and the equivalent diameter.

    Parameters
    ----------
    axis_major_length : ArrayLike
        The major axis length of each region.
    equivalent_diameter : ArrayLike
        The equivalent diameter of each region.

    Returns
    -------
    ArrayLike
        The ratio between the major axis length and the equivalent diameter for each region.
    """
    aml: ArrayLike = np.divide(
        axis_major_length,
        equivalent_diameter,
        out=np.full_like(a=axis_major_length, fill_value=np.nan, dtype=np.float64),
        where=equivalent_diameter != 0,
    )
    return aml


@rp_table_wrapper
def convex_hull_equiv_diam_ratio(
    area_convex: ArrayLike, equivalent_diameter: ArrayLike
) -> ArrayLike:
    """Calculates the ratio between the convex area and the equivalent diameter.

    Parameters
    ----------
    area_convex : ArrayLike
        The convex area of each region.
    equivalent_diameter : ArrayLike
        The equivalent diameter of each region.

    Returns
    -------
    ArrayLike
        The ratio between the convex area and the equivalent diameter for each region.
    """
    chedr: ArrayLike = np.divide(
        area_convex,
        equivalent_diameter,
        out=np.full_like(a=area_convex, fill_value=np.nan, dtype=np.float64),
        where=equivalent_diameter != 0,
    )
    return chedr


@rp_table_wrapper
def centroid_diff(
    image: list[ArrayLike], image_convex: list[ArrayLike], area: ArrayLike
) -> ArrayLike:
    """Computes the L2 distance between the centroid of the region and the centroid of the convex hull.

    Parameters
    ----------
    image : list[ArrayLike]
        The container of the image of each region.
    image_convex : list[ArrayLike]
        The container of the convex image of each region.
    area : ArrayLike
        The area of each region.

    Returns
    -------
    ArrayLike
        The L2 distance between the centroid of the region and the centroid of the convex hull.
    """
    centroid_dist = np.empty_like(area, dtype=np.float64)
    for i, (im, im_c, a) in enumerate(zip(image, image_convex, area, strict=True)):
        cell_M = moments(im)
        cell_centroid = np.array([cell_M[1, 0] / cell_M[0, 0], cell_M[0, 1] / cell_M[0, 0]])

        convex_M = moments(im_c)
        convex_centroid = np.array(
            [convex_M[1, 0] / convex_M[0, 0], convex_M[0, 1] / convex_M[0, 0]]
        )

        centroid_dist[i] = np.linalg.norm(cell_centroid - convex_centroid) / np.sqrt(a)
    return centroid_dist


@rp_table_wrapper
def num_concavities(image: list[ArrayLike], image_convex: list[ArrayLike], **kwargs) -> ArrayLike:
    """Computes the number of concavities in the region.

    Parameters
    ----------
    image : list[ArrayLike]
        The container of the image of each region.
    image_convex : list[ArrayLike]
        The container of the convex image of each region.
    **kwargs:
        Keyword arguments for `_diff_img_concavities`. The following keywords are supported:

            - `small_idx_area_cutoff`: int, optional
                    The area cutoff for small regions, by default 10

            - `max_compactness`: int, optional
                    The maximum compactness for small regions, by default 60

            - `large_idx_area_cutoff`: int, optional
                    The area cutoff for large regions, by default 100

    Returns
    -------
    ArrayLike
        The number of concavities per region, threshold.
    """
    n_concavities: ArrayLike = np.zeros_like(image, dtype=np.int64)
    for i, (im, im_c) in enumerate(zip(image, image_convex, strict=True)):
        diff_img: ArrayLike = im_c ^ im

        if np.sum(diff_img) < 0:
            n_concavities[i] = _diff_img_concavities(diff_img, **kwargs)
    return n_concavities


def _diff_img_concavities(diff_img: ArrayLike, **kwargs) -> np.int64:
    """A helper function which  calculates the number of concativies for a single region.

    Parameters
    ----------
    diff_img : ArrayLike
        The difference image between the convex hull and the region.
    **kwargs:
        Keyword arguments for `_diff_img_concavities`. The following keywords are supported:

            - `small_idx_area_cutoff`: int, optional
                    The area cutoff for small regions, by default 10

            - `max_compactness`: int, optional
                    The maximum compactness for small regions, by default 60

            - `large_idx_area_cutoff`: int, optional
                    The area cutoff for large regions, by default 100

    Returns
    -------
    np.int64
        The number of concavities for the region.
    """
    labeled_diff_image = label(diff_img, connectivity=1)

    hull_df: pd.DataFrame = compute_region_props_df(
        labeled_diff_image, properties=["area", "perimeter"]
    )
    hull_df["compactness"] = np.square(hull_df["perimeter"]) / hull_df["area"]

    small_idx_area_cutoff = kwargs.get("small_idx_area_cutoff", 10)
    compactness_cutoff = kwargs.get("max_compactness", 60)
    large_idx_area_cutoff = kwargs.get("large_idx_area_cutoff", 100)

    small_idx = (hull_df["area"] < small_idx_area_cutoff) & (
        hull_df["compactness"] < compactness_cutoff
    )
    large_idx = (hull_df["area"] > large_idx_area_cutoff) & (
        hull_df["compactness"] < compactness_cutoff
    )
    combined_idx = small_idx | large_idx

    return np.sum(combined_idx)


# TODO: Implement nuclear properties
def nc_ratio(**kwargs):
    pass


REGIONPROPS_FUNCTIONS: dict[str, Callable] = {
    "major_minor_axis_ratio": major_minor_axis_ratio,
    "perim_square_over_area": perim_square_over_area,
    "major_axis_equiv_diam_ratio": major_axis_equiv_diam_ratio,
    "convex_hull_equiv_diam_ratio": convex_hull_equiv_diam_ratio,
    "centroid_dif": centroid_diff,
    "num_concavities": num_concavities,
    "nc_ratio": nc_ratio,
}
