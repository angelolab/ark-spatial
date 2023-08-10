import natsort as ns
import ray

from ark.core._accessors import SpatialDataAccessor, register_spatial_data_accessor

from .utils import _create_deepcell_input


@register_spatial_data_accessor("segmentation")
class SegmentationAccessor(SpatialDataAccessor):
    def create_deepcell_input(
        self,
    ):
        futures = []
        for _, fov_data in ns.natsorted(self.sdata.images.items(), key=lambda x: x[0]):
            futures.append(_create_deepcell_input.remote(fov_data))
        print(ray.get(futures))
