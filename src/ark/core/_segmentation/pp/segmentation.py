from concurrent.futures import as_completed

import httpx
import natsort as ns
from anyio import start_blocking_portal
from spatial_image import SpatialImage
from spatialdata.models import C

from ark.core._accessors import SpatialDataAccessor, register_spatial_data_accessor

from .utils import _create_deepcell_input

DEEPCELL_URL = "https://deepcell.org"


@register_spatial_data_accessor("segmentation")
class SegmentationAccessor(SpatialDataAccessor):
    spatial_images: list[SpatialImage]

    def run_deepcell(
        self,
        nucs: list[str],
        mems: list[str],
    ):
        with start_blocking_portal() as portal:
            futures = [
                portal.start_task_soon(self.create_deepcell_input, nucs, mems, fov)
                for fov in ns.natsorted(self.sdata.images.values(), key=lambda x: x.name)
            ]
            for future in as_completed(futures):
                print(future.result())

    async def create_deepcell_input(
        self,
        nucs: list[str],
        mems: list[str],
        fov: SpatialImage,
    ):
        f: SpatialImage = fov.sel({C: [*nucs, *mems]}).sum(C)

        async with httpx.AsyncClient(base_url=DEEPCELL_URL) as dc_session:
            segmentation_images = await _create_deepcell_input(f, dc_session)

        return segmentation_images
