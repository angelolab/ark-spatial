from concurrent.futures import as_completed

import httpx
from anyio import start_blocking_portal
from spatial_image import SpatialImage
from spatialdata.models import C

from ark.core._accessors import SpatialDataAccessor, register_spatial_data_accessor

from .utils.deepcell import SegmentationImageContainer, _create_deepcell_input

DEEPCELL_URL = "https://deepcell.org"


@register_spatial_data_accessor("segmentation")
class SegmentationAccessor(SpatialDataAccessor):
    def run_deepcell(
        self,
        nucs: list[str],
        mems: list[str],
    ) -> None:
        with start_blocking_portal() as portal:
            futures = [
                portal.start_task_soon(self.create_deepcell_input, nucs, mems, fov)
                for fov in self.sdata.iter_images()
            ]
            for future in as_completed(futures):
                sic: SegmentationImageContainer = future.result()
                for seg_type, seg_label in sic.segmentation_label_masks.items():
                    try:
                        self.sdata.add_labels(name=f"{sic.fov_name}_{seg_type}", labels=seg_label)
                    except KeyError:
                        continue

    async def create_deepcell_input(
        self,
        nucs: list[str] | None,
        mems: list[str] | None,
        fov: SpatialImage,
    ):
        nuc_mem_img = []
        if nucs:
            nuc_mem_img.append(fov.sel({C: nucs}).sum(C))
        if mems:
            nuc_mem_img.append(fov.sel({C: mems}).sum(C))

        f = SpatialImage(nuc_mem_img, name=fov.name)

        async with httpx.AsyncClient(base_url=DEEPCELL_URL) as dc_session:
            segmentation_images: SegmentationImageContainer = await _create_deepcell_input(
                f, dc_session
            )

        return segmentation_images
