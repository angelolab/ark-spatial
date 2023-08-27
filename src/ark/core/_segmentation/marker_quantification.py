from collections.abc import Iterable

import dask.dataframe as dd
import spatialdata as sd
from anndata import AnnData, concat
from dask.distributed import get_client
from spatial_image import SpatialImage
from tqdm.auto import tqdm

import ark
from ark.core._accessors import SpatialDataAccessor, register_spatial_data_accessor
from ark.core._segmentation.utils.regionprops_extraction import REGIONPROPS_SINGLE_COMP

from .utils import REGIONPROPS_BASE, regionprops


@register_spatial_data_accessor("marker_quantification")
class MarkerQuantificationAccessor(SpatialDataAccessor):
    _region_props: list[str] = REGIONPROPS_BASE.copy()
    _derived_props: list[str] = REGIONPROPS_SINGLE_COMP

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

        fov_regionprops_tables = []
        fov_markers_tables = []

        for fov_id, fov_sd in tqdm(
            self.sdata.iter_cohort(labels="whole_cell"), total=len(self.sdata.iter_cohort)
        ):
            f: dd.DataFrame = self._get_single_compartment_props(
                fov_id=fov_id,
                fov_sd=fov_sd,
                nuclear_counts=nuclear_counts,
            )
            g: sd.SpatialData = ark.client.submit(self._compute_marker_counts, fov_id, fov_sd)

            fov_regionprops_tables.append(f)
            fov_markers_tables.append(g)

        obs: dd.DataFrame = dd.concat(dfs=fov_regionprops_tables).compute()
        d = concat(adatas=ark.client.gather(fov_markers_tables))
        tab1 = AnnData(X=d.X, obs=obs, uns=d.uns, var=d.var)

        return obs, tab1
        # print(spatial_data_attrs)

        # self.sdata.table = AnnData(obs=cell_table, uns={"spatialdata_attrs": spatial_data_attrs})

    def generate_fov_table(self, fov_id, fov_sd):
        local_c = get_client()

        scp = local_c.submit(self._get_single_compartment_props, fov_id, fov_sd)
        cmc = local_c.submit(self._compute_marker_counts, fov_id, fov_sd)

        return (scp, cmc)

    def _get_single_compartment_props(
        self,
        fov_id: str,
        fov_sd: sd.SpatialData,
        nuclear_counts: bool = False,
    ) -> dd.DataFrame:
        segmentation_mask: SpatialImage = fov_sd.labels[f"{fov_id}_whole_cell"]
        f = regionprops(
            labels=segmentation_mask,
            properties=self.region_props,
            derived_properties=self._derived_props,
        ).rename(columns={"label": "cell_id"})
        f["fov_id"] = f"{fov_id}_whole_cell"
        f["fov_id"] = f["fov_id"].astype("category")
        return f

    def _compute_marker_counts(
        self,
        fov_id: str,
        fov_sd: sd.SpatialData,
    ) -> AnnData:
        fov_markers: AnnData = fov_sd.aggregate(
            values=fov_id,
            by=f"{fov_id}_whole_cell",
            agg_func="sum",
            instance_key="cell_id",
            region_key="fov_id",
        ).table
        return fov_markers
