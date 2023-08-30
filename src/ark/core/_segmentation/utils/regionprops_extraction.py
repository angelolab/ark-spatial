"""
`regionprops_extraction.py`.

From https://github.com/jrussell25/dask-regionprops/blob/main/dask_regionprops/regionprops.py,
Modified to accept other properties.
"""


import inspect
import re
from collections.abc import Callable, Mapping
from itertools import product

import dask.dataframe as dd
import numpy as np
import pandas as pd
import xarray as xr
from dask import delayed
from numpy.typing import ArrayLike
from skimage.measure import label, moments, regionprops_table

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


def regionprops_df(
    labels: ArrayLike,
    intensity: ArrayLike | None = None,
    properties: tuple[str, ...] = REGIONPROPS_BASE,
    derived_properties: Mapping[str, ArrayLike] = {},
    other_cols=None,
) -> pd.DataFrame:
    """
    Computes the regionprops of each labelled region in a segmentaiton image.

    Lightly wrap skimage.measure.regionprops_table to return a DataFrame.
    Also allow for the addition of extra columns, used in reginprops to track
    non core dimensions of input.

    Parameters
    ----------
    labels : types.ArrayLike
        Array containing labelled regions. Background is assumed to have
        value 0 and will be ignored.
    intensity : Optional[types.ArrayLike] Default None
        Optional intensity field to compute weighted region properties from.
    properties : tuple[str]
        Properties to compute for each region. By default compute all
        properties that return fixed size outputs. If provided an intensity image,
        corresponding weighted properties will also be computed by defualt.

    Returns
    -------
    properties : pd.DataFrame
        Dataframe containing the desired properties as columns and each
        labelled region as a row.
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
    for prop in ["image_convex", "image"]:
        if prop in rp_table:
            rp_table.pop(prop)

    df = pd.DataFrame(rp_table)
    for k, v in other_cols.items():
        df[k] = v
    return df


def regionprops(
    labels: ArrayLike,
    intensity: ArrayLike | None = None,
    properties: tuple[str, ...] = REGIONPROPS_BASE,
    derived_properties: tuple[str, ...] = REGIONPROPS_SINGLE_COMP,
    core_dims: tuple[int | str, ...] | None = None,
) -> dd.DataFrame:
    """
    f.

    Loop over the frames of ds and compute the regionprops for
    each labelled image in each frame.

    Parameters
    ----------
    labels : types.ArrayLike
        Array containing labelled regions. Background is assumed to have
        value 0 and will be ignored.
    intensity : array-like or None, Default None
        Optional intensity field to compute weighted region properties from.
    properties : str, tuple[str] default "non-image"
        Properties to compute for each region. Can pass an explicit tuple
        directly to regionprops or use one of the followings shortcuts:
        "minimal", "non-image", "all". If provided an intensity image, basic
        weighted properties will also be computed by defualt.
    core_dims : tuple[int] or tuple[str] default None
        Dimensions of input arrays that correspond to spatial (xy) dimensions of each
        image. If None, it is assumed that the final two dimensions are the
        spatial dimensions.

    Returns
    -------
    regionprops_df : dask.DataFrame
        Lazily constructed dataframe containing columns for each specifified
        property.
    """
    d_regionprops = delayed(regionprops_df)

    loop_sizes = _get_loop_sizes(labels, core_dims)

    if intensity is not None:
        properties = DEFAULT_REGIONPROPS

    meta = _get_meta(loop_sizes, properties, derived_properties)

    labels_arr, intensity_arr = _get_arrays(labels, intensity)

    all_props = []

    for dims in product(*(range(v) for v in loop_sizes.values())):
        other_cols = dict(zip(loop_sizes.keys(), dims, strict=True))

        if intensity_arr is not None:
            frame_props = d_regionprops(
                labels_arr[dims], intensity_arr[dims], properties, derived_properties, other_cols
            )
        else:
            frame_props = d_regionprops(
                labels_arr[dims], None, properties, derived_properties, other_cols
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


def _get_meta(loop_sizes, properties, derived_properties):
    meta = pd.DataFrame()
    for prop in properties:
        meta = meta.join(DEFAULT_META[DEFAULT_PROPS_TO_COLS[prop]], how="outer")
    if derived_properties:
        for prop in derived_properties:
            meta[prop] = [np.nan]

    other_cols = pd.DataFrame(columns=list(loop_sizes.keys()), dtype=int)

    return meta.join(other_cols)


def _get_loop_sizes(labels, core_dims):
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


def _get_pos_core_dims(core_dims, ndim):
    pos_core_dims = []
    for d in core_dims:
        if d < 0:
            pos = ndim + d
            pos_core_dims.append(pos)
        else:
            pos_core_dims.append(d)
    return tuple(pos_core_dims)


def _get_arrays(labels, intensity):
    if intensity is None:
        intensity_arr = None
    else:
        if isinstance(intensity, xr.DataArray):
            intensity_arr = intensity.data
        else:
            intensity_arr = intensity

    if isinstance(labels, xr.DataArray):
        labels_arr = labels.data
    else:
        labels_arr = labels

    return labels_arr, intensity_arr


def rp_table_wrapper(func):
    def wrapper(region_properties: dict[str, ArrayLike], **kwargs):
        props: list[str] = inspect.getargs(func.__code__).args
        kwargs.update({k: region_properties[k] for k in props if k in region_properties})

        return func(**kwargs)

    return wrapper


@rp_table_wrapper
def major_minor_axis_ratio(axis_minor_length: ArrayLike, axis_major_length: ArrayLike) -> ArrayLike:
    """_summary_.

    Parameters
    ----------
    axis_minor_length : ArrayLike
        _description_
    axis_major_length : ArrayLike
        _description_

    Returns
    -------
    ArrayLike
        _description_
    """
    mmar: ArrayLike = np.empty_like(axis_major_length, dtype=np.float64)
    for i, (a, b) in enumerate(zip(axis_minor_length, axis_major_length, strict=True)):
        if b == 0:
            mmar[i] = np.nan
        else:
            mmar[i] = a / b
    return mmar


@rp_table_wrapper
def perim_square_over_area(perimeter: ArrayLike, area: ArrayLike) -> ArrayLike:
    """_summary_.

    Parameters
    ----------
    perimeter : ArrayLike
        _description_
    area : ArrayLike
        _description_

    Returns
    -------
    ArrayLike
        _description_
    """
    psoa: ArrayLike = np.empty_like(perimeter, dtype=np.float64)
    for i, (p, a) in enumerate(zip(perimeter, area, strict=True)):
        psoa[i] = p**2 / a
    return psoa


@rp_table_wrapper
def major_axis_equiv_diam_ratio(
    axis_major_length: ArrayLike, equivalent_diameter: ArrayLike
) -> ArrayLike:
    """_summary_.

    Parameters
    ----------
    axis_major_length : ArrayLike
        _description_
    equivalent_diameter : ArrayLike
        _description_

    Returns
    -------
    ArrayLike
        _description_
    """
    aml: ArrayLike = np.empty_like(axis_major_length, dtype=np.float64)
    for i, (a, e) in enumerate(zip(axis_major_length, equivalent_diameter, strict=True)):
        if e == 0:
            aml[i] = np.nan
        else:
            aml[i] = a / e
    return aml


@rp_table_wrapper
def convex_hull_equiv_diam_ratio(
    area_convex: ArrayLike, equivalent_diameter: ArrayLike
) -> ArrayLike:
    """_summary_.

    Parameters
    ----------
    area_convex : ArrayLike
        _description_
    equivalent_diameter : ArrayLike
        _description_

    Returns
    -------
    ArrayLike
        _description_
    """
    chedr: ArrayLike = np.empty_like(area_convex, dtype=np.float64)
    for i, (a, e) in enumerate(zip(area_convex, equivalent_diameter, strict=True)):
        if e == 0:
            chedr[i] = np.nan
        else:
            chedr[i] = a / e
    return chedr


@rp_table_wrapper
def centroid_diff(
    image: list[ArrayLike], image_convex: list[ArrayLike], area: ArrayLike
) -> ArrayLike:
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
    n_concavities: ArrayLike = np.zeros_like(image, dtype=np.int64)
    for i, (im, im_c) in enumerate(zip(image, image_convex, strict=True)):
        diff_img: ArrayLike = im_c ^ im

        if np.sum(diff_img) < 0:
            n_concavities[i] = _diff_img_concavities(diff_img, **kwargs)
    return n_concavities


def _diff_img_concavities(diff_img, **kwargs):
    labeled_diff_image = label(diff_img, connectivity=1)

    hull_df: pd.DataFrame = regionprops_df(labeled_diff_image, properties=["area", "perimeter"])
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
