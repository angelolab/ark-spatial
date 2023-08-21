import spatialdata as sd
from dask_regionprops import regionprops

from ark.core._accessors import SpatialDataAccessor, register_spatial_data_accessor


@register_spatial_data_accessor("marker_quantification")
class MarkerQuantificationAccessor(SpatialDataAccessor):
    def generate_cell_table(self, nuclear_counts: bool = False):
        # self.sdata.table(AnnData())

        # for fov in self.sdata.iter_fovs():
        #     ...
        for fov_sd in self.sdata.iter_fovs():
            f = self._get_single_compartment_props(fov_sd, nuclear_counts=True)
            print(f)

    def _get_single_compartment_props(self, fov_sd: sd.SpatialData, nuclear_counts: bool = False):
        f = regionprops(labels=fov_sd.labels[f"{list(fov_sd.images.keys())[0]}_nuclear"])
        return f.compute()
