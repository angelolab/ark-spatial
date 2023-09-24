from collections.abc import Iterable
from itertools import chain

import dask.dataframe as dd
import spatialdata as sd
from anndata import AnnData
from dask import compute, delayed
from spatial_image import SpatialImage
from spatialdata.models import X, Y

from ark.core._accessors import SpatialDataAccessor, register_spatial_data_accessor
from ark.core._segmentation.utils.regionprops_extraction import REGIONPROPS_SINGLE_COMP

from .utils import REGIONPROPS_BASE, regionprops


@register_spatial_data_accessor("marker_quantification")
class MarkerQuantificationAccessor(SpatialDataAccessor):
    _region_props: list[str] = REGIONPROPS_BASE.copy()
    _derived_props: list[str] = REGIONPROPS_SINGLE_COMP.copy()

    @property
    def default_region_props(self) -> list[str]:
        """Returns the default region properties to extract.

        Returns
        -------
        list[str]
            The default region properties to extract.
        """
        return self._region_props

    @default_region_props.setter
    def region_props(self, value: Iterable[str]) -> None:
        """Sets the default region properties to extract.

        Parameters
        ----------
        value : Iterable[str]
            Sets the default region properties to extract.
        """
        self._region_props = list(set(self._region_props) | set(value))

    @delayed
    def _compute_fov_table(self, fov_sd: sd.SpatialData, rp_df: dd.DataFrame) -> sd.SpatialData:
        """Computes the regionprops and marker counts for a single fov (delayed).

        Parameters
        ----------
        fov_sd : sd.SpatialData
            The `SpatialData` object for a single fov.
        rp_df : dd.DataFrame
            The regionprops dataframe for a single fov / associated segmentation mask.

        Returns
        -------
        sd.SpatialData
            The `SpatialData` object for a single fov, with the `SpatialData.Table` property.
        """
        fov_sd.table.obs = rp_df.compute()
        fov_sd.table.obsm["spatial"] = fov_sd.table.obs[
            [f"{X}_centroid", f"{Y}_centroid"]
        ].to_numpy()

        return fov_sd

    @delayed
    def _compute_regionprops(
        self,
        fov_id: str,
        nuclear_counts: bool = False,
    ) -> dd.DataFrame:
        """Computes the regionprops for a single fov.

        Parameters
        ----------
        fov_id : str
            The fov id.
        nuclear_counts : bool, optional
            Whether to compute the nuclear regionprops statistics, by default False

        Returns
        -------
        dd.DataFrame
            The delayed regionprops dataframe.
        """
        segmentation_mask: SpatialImage = self.sdata.labels[f"{fov_id}_whole_cell"]
        rp_df: dd.DataFrame = regionprops(
            labels=segmentation_mask,
            properties=self.region_props,
            derived_properties=self._derived_props,
        ).rename(
            columns={
                "label": "instance_id",
                "centroid-0": f"{X}_centroid",
                "centroid-1": f"{Y}_centroid",
            }
        )
        rp_df["region"] = f"{fov_id}_whole_cell"
        rp_df["region"] = rp_df["region"].astype("category")
        return rp_df

    @delayed
    def _compute_marker_counts(
        self,
        fov_id: str,
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
        agg_val: sd.SpatialData = self.sdata.aggregate(
            values=fov_id,
            by=f"{fov_id}_whole_cell",
            agg_func=agg_func,
            region_key="region",
            instance_key="instance_id",
            target_coordinate_system=fov_id,
            deepcopy=False,
        )
        agg_val.table.obs = agg_val.table.obs.rename(columns={"label": "instance_id"})
        return agg_val

    def generate_cell_table(
        self, nuclear_counts: bool = False, properties: Iterable[str] = ("label", "area")
    ) -> None:
        """Generates the cell table for the `SpatialData` object, for all fovs.

        Parameters
        ----------
        nuclear_counts : bool, optional
            Whether to compute the nuclear regionprops statistics, by default False
        properties : Iterable[str], optional
            properties to compute for the regionprops function, by default ("label", "area")
        """
        self.region_props = properties
        fov_ids, _ = chain(zip(*self.sdata.iter_coords, strict=True))

        fov_sd_ct = []
        for fov_id in fov_ids:
            rp_df: dd.DataFrame = self._compute_regionprops(fov_id)
            fov_sd: sd.SpatialData = self._compute_marker_counts(fov_id)

            fov_sd_ct.append(self._compute_fov_table(fov_sd, rp_df))

        (q,) = compute(fov_sd_ct, scheduler="processes")
        cell_table: AnnData = sd.concatenate([*q]).table
        # Make the obs index contain all unique values
        cell_table.obs_names_make_unique()
        self.sdata.table = cell_table
