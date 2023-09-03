from concurrent.futures import as_completed

import httpx
import spatialdata as sd
import xarray as xr
from anyio import start_blocking_portal
from spatial_image import SpatialImage
from spatialdata.models import C

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
        nucs: list[str] | None = None,
        mems: list[str] | None = None,
        overwrite_masks: bool = False,
    ) -> None:
        match (nucs, mems):
            case (None, None):
                raise ValueError("Must specify at least one nuclear marker or one membrane marker.")
            case (list(), list()) if (len(nucs) == 0 and len(mems) == 0):
                raise ValueError("Must specify at least one nuclear marker or one membrane marker.")

        self.nuclear_markers = nucs
        self.membrane_markers = mems

        with start_blocking_portal() as portal:
            futures = [
                portal.start_task_soon(self._run_per_fov_deepcell, fov_name, fov_sd)
                for fov_name, fov_sd in self.sdata.iter_coords
            ]
            for future in as_completed(futures):
                sic: SegmentationImageContainer = future.result()
                for seg_type, seg_label in sic.segmentation_label_masks.items():
                    try:
                        self.sdata.add_labels(name=f"{sic.fov_name}_{seg_type}", labels=seg_label)
                    except KeyError:
                        if overwrite_masks:
                            _ = self.sdata.labels.pop(f"{sic.fov_name}_{seg_type}", None)
                            self.sdata.add_labels(
                                name=f"{sic.fov_name}_{seg_type}", labels=seg_label
                            )

    def _sum_markers(self, fov_name: str, fov_sd: sd.SpatialData) -> SpatialImage:
        fov_si: SpatialImage = fov_sd.images[fov_name]

        labels = xr.where(fov_si.c.isin(self.nuclear_markers), "nucs", "_")
        labels[fov_si.c.isin(self.membrane_markers)] = "mems"
        return fov_si.groupby(labels, squeeze=False).sum().drop_sel({C: "_"})

    async def _run_per_fov_deepcell(
        self,
        fov_name: str,
        fov_sd: sd.SpatialData,
    ) -> SegmentationImageContainer:
        fov_si: SpatialImage = self._sum_markers(fov_name, fov_sd)
        async with httpx.AsyncClient(base_url=DEEPCELL_URL) as dc_session:
            segmentation_images: SegmentationImageContainer = await _create_deepcell_input(
                fov_si, dc_session
            )

        return segmentation_images
