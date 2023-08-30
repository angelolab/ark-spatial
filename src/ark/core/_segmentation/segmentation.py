from concurrent.futures import as_completed

import httpx
import xbatcher as xb
from anyio import start_blocking_portal
from spatial_image import SpatialImage
from spatialdata.models import C, X, Y

from ark.core._accessors import SpatialDataAccessor, register_spatial_data_accessor

from .utils.deepcell import SegmentationImageContainer, _create_deepcell_input

DEEPCELL_URL = "https://deepcell.org"

DEEPCELL_MAX_X_PIXELS = 2048
DEEPCELL_MAX_Y_PIXELS = 2048


@register_spatial_data_accessor("segmentation")
class SegmentationAccessor(SpatialDataAccessor):
    @property
    def nuclear_markers(self) -> list[str]:
        return self._nuclear_markers

    @property
    def membrane_markers(self) -> list[str]:
        return self._membrane_markers

    @nuclear_markers.setter
    def nuclear_markers(self, value: list[str]) -> None:
        self._nuclear_markers = value

    @membrane_markers.setter
    def membrane_markers(self, value: list[str]) -> None:
        self._membrane_markers = value

    def run_deepcell(
        self,
        nucs: list[str] | None,
        mems: list[str] | None,
    ) -> None:
        if (nucs is None and mems is None) or (len(nucs) == 0 and len(mems) == 0):
            raise ValueError("Must specify at least one nuclear marker or one membrane marker.")

        self.nuclear_markers = nucs
        self.membrane_markers = mems

        # with WorkerPool(n_jobs=None) as pool:
        #     for fov_name, fov_sd in self.sdata.iter_coords:
        #         pool.imap(
        #             func=self._run_per_fov_deepcell,
        #             iterable_of_args=
        #         )

        with start_blocking_portal() as portal:
            futures = [
                portal.start_task_soon(self._create_deepcell_input, fov_name, fov_sd)
                for fov_name, fov_sd in self.sdata.iter_coords
            ]
            for future in as_completed(futures):
                sic: SegmentationImageContainer = future.result()
                for seg_type, seg_label in sic.segmentation_label_masks.items():
                    self.sdata.add_labels(name=f"{sic.fov_name}_{seg_type}", labels=seg_label)

    def _run_per_fov_deepcell(self, fov_sd: SpatialImage):
        fov_name: str = fov_sd.name
        nuc_mem_img = []

        nuc_mem_img.append(fov_sd.images[fov_name].sel({C: self.nuclear_markers}).sum(C))
        nuc_mem_img.append(fov_sd.images[fov_name].sel({C: self.membrane_markers}).sum(C))
        f = SpatialImage(nuc_mem_img, name=fov_name)

        with start_blocking_portal() as portal:
            fov_parts: list[SpatialImage] = self._partition_spatialimage(seg_fov=f)

            futures = [
                portal.start_task_soon(self._create_deepcell_input, fov_name, fov_part)
                for fov_part in fov_parts
            ]

            for _future in as_completed(futures):
                pass

            sic: SegmentationImageContainer = portal.start_task_soon(
                self._create_deepcell_input, fov_name, fov_parts
            )
            for seg_type, seg_label in sic.segmentation_label_masks.items():
                self.sdata.add_labels(name=f"{sic.fov_name}_{seg_type}", labels=seg_label)

    def _partition_spatialimage(self, seg_fov: SpatialImage) -> list[SpatialImage]:
        min_max_y = DEEPCELL_MAX_Y_PIXELS
        min_max_x = DEEPCELL_MAX_X_PIXELS

        y_size: int = seg_fov.sizes[Y]
        x_size: int = seg_fov.sizes[X]

        # Try to reduce the batch size from the max size until it is divisible by the fov size
        # Go from 2048 -> 1024 -> 512
        for fov_size in [2048, 1024, 512]:
            if y_size % fov_size == 0:
                min_max_y: int = fov_size
            if x_size % fov_size == 0:
                min_max_x: int = fov_size

        return list(
            xb.BatchGenerator(
                ds=seg_fov,
                input_dims={X: min_max_x, Y: min_max_y},
            )
        )

    async def _create_deepcell_input(
        self,
        fov_si: SpatialImage,
    ) -> SegmentationImageContainer:
        async with httpx.AsyncClient(base_url=DEEPCELL_URL) as dc_session:
            segmentation_images: SegmentationImageContainer = await _create_deepcell_input(
                fov_si, dc_session
            )

        return segmentation_images
