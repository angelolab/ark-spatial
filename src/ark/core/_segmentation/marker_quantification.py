from collections.abc import Iterable

import pandas as pd
import spatialdata as sd
from anndata import AnnData
from dask import compute, delayed
from spatial_image import SpatialImage
from spatialdata.models import X, Y
from tqdm.dask import TqdmCallback

from ark.core._accessors import SpatialDataAccessor, register_spatial_data_accessor
from ark.core._segmentation.utils.regionprops_extraction import REGIONPROPS_SINGLE_COMP
from .utils import REGIONPROPS_BASE, ufunc_regionprops


@delayed
def _compute_fov_table(fov_sd: sd.SpatialData, rp_df: pd.DataFrame) -> sd.SpatialData:
    """Computes the regionprops and marker counts for a single fov (delayed).

    Parameters
    ----------
    fov_sd : sd.SpatialData
        The `SpatialData` object for a single fov.
    rp_df : pd.DataFrame
        The regionprops dataframe for a single fov / associated segmentation mask.

    Returns
    -------
    sd.SpatialData
        The `SpatialData` object for a single fov, with the `SpatialData.Table` property.
    """
    fov_sd.table.obs = rp_df
    fov_sd.table.obsm["spatial"] = fov_sd.table.obs[[f"{X}_centroid", f"{Y}_centroid"]]
    return fov_sd


@delayed
def _compute_regionprops(
    fov_id: str,
    fov_sd: str,
    derived_props: list[str],
    label_id: str = "whole_cell"
) -> pd.DataFrame:
    """Computes the regionprops for a single fov.

    Parameters
    ----------
    fov_id : str
        The fov id.

    Returns
    -------
    pd.DataFrame
        The delayed regionprops dataframe.
    """
    fov_seg_mask_name = f"{fov_id}_{label_id}"

    segmentation_mask: SpatialImage = fov_sd.labels[fov_seg_mask_name]
    rp_df = ufunc_regionprops(
        labels=segmentation_mask,
        dim=[Y, X],
        derived_props=derived_props,
    )

    rp_df["region"] = pd.Categorical([fov_seg_mask_name] * rp_df.shape[0])
    return rp_df

@delayed
def _compute_marker_counts(
    fov_id: str,
    fov_sd: sd.SpatialData,
    agg_func: str = "sum",
) -> sd.SpatialData:
    """Computes the marker counts for a single fov / label mask.

    Parameters
    ----------
    fov_id : str
        The fov id.
    agg_func : str, optional
        The aggregation function to use, by default "sum".

    Returns
    -------
    sd.SpatialData
        The `SpatialData` object for a single fov, with the `AnnData` `spatialdata.models.Table` property
        containing the marker counts aggregated by sum.
    """
    agg_val: sd.SpatialData = fov_sd.aggregate(
        values=fov_id,
        by=f"{fov_id}_whole_cell",
        agg_func=agg_func,
        region_key="region",
        instance_key="instance_id",
        target_coordinate_system=fov_id,
        deepcopy=False,
    )

    agg_val.table.var_names = agg_val.table.var_names.map(
        lambda vn: vn.replace(f"_{agg_func}", "").replace("channel_", "")
    )
    agg_val.table.obs_names = agg_val.table.obs_names.map(lambda on: f"{fov_id}_{on}")

    return agg_val

def _filter_fovs_by_label(args, label_id: str):
    """Filters the fovs by the label id.

    Parameters
    ----------
    fov_id : str
        The fov id.
    fov_sd : sd.SpatialData
        The `SpatialData` object for a single fov.

    Returns
    -------
    bool
        Whether the fov id is in the `SpatialData.labels` object.
    """
    fov_id, fov_sd = args
    try:
        if any(label_id in k for k in fov_sd.labels.keys()):
            return True
        else:
            return False
    except KeyError:
        return False

@register_spatial_data_accessor("marker_quantification")
class MarkerQuantificationAccessor(SpatialDataAccessor):
    _region_props: list[str] = REGIONPROPS_BASE.copy()
    _derived_props: list[str] = REGIONPROPS_SINGLE_COMP.copy()

    @property
    def region_props(self) -> list[str]:
        """Returns the default region properties to extract.

        Returns
        -------
        list[str]
            The default region properties to extract.
        """
        return self._region_props

    @region_props.setter
    def region_props(self, value: Iterable[str]) -> None:
        """Sets the default region properties to extract.

        Parameters
        ----------
        value : Iterable[str]
            Sets the default region properties to extract.
        """
        self._region_props = list(set(self._region_props) | set(value))


    def generate_cell_table(self, label: str, properties: Iterable[str] = ("label", "area")) -> None:
        """Generates the cell table for the `SpatialData` object, for all fovs.

        Parameters
        ----------
        nuclear_counts : bool, optional
            Whether to compute the nuclear regionprops statistics, by default False
        properties : Iterable[str], optional
            properties to compute for the regionprops function, by default ("label", "area")
        """

        self.region_props = properties

        from functools import partial

        fov_sd_ct = []
        for fov_id, fov_sd in self.sdata.iter_coords().filter(filter_fn=partial(_filter_fovs_by_label, label_id=label)):
            fov_sd = _compute_marker_counts(fov_id, fov_sd, agg_func="sum")
            fov_rp = _compute_regionprops(fov_id, fov_sd, derived_props=self._derived_props, label_id=label)

            fov_sd_ct.append(
                _compute_fov_table(
                    fov_sd=fov_sd,
                    rp_df=fov_rp,
                )
            )

        with TqdmCallback():
            computed_cts = compute(*fov_sd_ct)
        cell_table: AnnData = sd.concatenate(list(computed_cts)).table

        cell_table.obs_names_make_unique(join="_")
        self.sdata.table = cell_table
