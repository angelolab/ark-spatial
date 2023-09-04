from collections.abc import Iterable
from itertools import chain

import dask.dataframe as dd
import spatialdata as sd
from anndata import AnnData
from dask import compute, delayed
from spatial_image import SpatialImage
from spatialdata.models import X, Y

import ark
from ark.core._accessors import SpatialDataAccessor, register_spatial_data_accessor
from ark.core._segmentation.utils.regionprops_extraction import REGIONPROPS_SINGLE_COMP

from .utils import REGIONPROPS_BASE, regionprops


@register_spatial_data_accessor("marker_quantification")
class MarkerQuantificationAccessor(SpatialDataAccessor):
    _region_props: list[str] = REGIONPROPS_BASE.copy()
    _derived_props: list[str] = REGIONPROPS_SINGLE_COMP.copy()

    @property
    def default_region_props(self) -> list[str]:
        return self._region_props

    @default_region_props.setter
    def region_props(self, value: Iterable[str]) -> None:
        self._region_props = list(set(self._region_props) | set(value))

    def generate_cell_table(
        self, nuclear_counts: bool = False, properties: Iterable[str] = ("label", "area")
    ):
        self.region_props = properties
        # c = Client(address=ark.address)
        fov_ids, _ = chain(zip(*self.sdata.iter_coords, strict=True))

        fov_sd_ct = []
        for fov_id in fov_ids:
            rp_df: dd.DataFrame = self._get_single_compartment_props(fov_id)
            fov_sd: sd.SpatialData = self._compute_marker_counts(fov_id)

            fov_sd_ct.append(self._generate_fov_table(fov_sd, rp_df))

        (q,) = compute(fov_sd_ct, address=ark.address)
        cell_table: AnnData = sd.concatenate([*q]).table
        # Make the obs index contain all unique values
        cell_table.obs_names_make_unique()
        self.sdata.table = cell_table

    @delayed
    def _generate_fov_table(self, fov_sd: sd.SpatialData, rp_df: dd.DataFrame) -> sd.SpatialData:
        fov_sd.table.obs = rp_df.compute()
        fov_sd.table.obsm["spatial"] = fov_sd.table.obs[
            [f"{X}_centroid", f"{Y}_centroid"]
        ].to_numpy()

        return fov_sd

    @delayed
    def _get_single_compartment_props(
        self,
        fov_id: str,
        nuclear_counts: bool = False,
    ) -> dd.DataFrame:
        segmentation_mask: SpatialImage = self.sdata.labels[f"{fov_id}_whole_cell"]
        rp_df = regionprops(
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
    ) -> sd.SpatialData:
        agg_val = self.sdata.aggregate(
            values=fov_id,
            by=f"{fov_id}_whole_cell",
            agg_func="sum",
            region_key="region",
            instance_key="instance_id",
            target_coordinate_system=fov_id,
            deepcopy=False,
        )
        agg_val.table.obs = agg_val.table.obs.rename(columns={"label": "instance_id"})
        return agg_val
