from concurrent.futures import as_completed

import httpx
import spatial_image as si
import spatialdata as sd
import xarray as xr
from anyio import start_blocking_portal
from spatialdata.models import C, Labels2DModel, X, Y
from spatialdata.transformations import Identity
from tqdm.auto import tqdm

from ark.core._accessors import SpatialDataAccessor, register_spatial_data_accessor

from .utils.deepcell import SegmentationImageContainer, _create_deepcell_input

DEEPCELL_URL = "https://deepcell.org"

DEEPCELL_MAX_X_PIXELS = 2048
DEEPCELL_MAX_Y_PIXELS = 2048


@register_spatial_data_accessor("segmentation")
class SegmentationAccessor(SpatialDataAccessor):
    def __init__(self, sdata: sd.SpatialData):
        super().__init__(sdata)
        self._membrane_markers = None
        self._nuclear_markers = None

    @property
    def nuclear_markers(self) -> list[str]:
        """The nuclear markers used for segmentation.

        Returns
        -------
        list[str]
            A list of nuclear markers used for segmentation.
        """
        return self._nuclear_markers

    @property
    def membrane_markers(self) -> list[str]:
        """The membrane markers used for segmentation.

        Returns
        -------
        list[str]
            A list of membrane markers used for segmentation.
        """
        return self._membrane_markers

    @nuclear_markers.setter
    def nuclear_markers(self, value: list[str]) -> None:
        """Sets the nuclear markers used for segmentation.

        These should be channel names in the `si.SpatialImage` object.

        Parameters
        ----------
        value : list[str]
            A list of nuclear markers used for segmentation.
        """
        self._nuclear_markers = value

    @membrane_markers.setter
    def membrane_markers(self, value: list[str]) -> None:
        """Sets the membrane markers used for segmentation.

        These should be channel names in the `si.SpatialImage` object.

        Parameters
        ----------
        value : list[str]
            A list of membrane markers used for segmentation.
        """
        self._membrane_markers = value

    def _set_markers(self, nucs: list[str] | None, mems: list[str] | None) -> None:
        """Sets the nuclear and membrane markers.

        Parameters
        ----------
        nucs : list[str] | None
            A list of nuclear markers used for segmentation.
        mems : list[str] | None
            A list of membrane markers used for segmentation.

        Raises
        ------
        ValueError
            Raised when `None` is passed for both nuclear and membrane markers are.
        ValueError
            Raised when no nuclear or membrane markers are specified.
        """
        match (nucs, mems):
            case (None, None):
                raise ValueError("Must specify at least one nuclear marker or one membrane marker.")
            case (list(), list()) if (len(nucs) == 0 and len(mems) == 0):
                raise ValueError("Must specify at least one nuclear marker or one membrane marker.")

        self.nuclear_markers = nucs
        self.membrane_markers = mems

    def run_deepcell(
        self,
        nucs: list[str] | None = None,
        mems: list[str] | None = None,
        overwrite_masks: bool = False,
    ) -> None:
        """Runs Deepcell segmentation on the `SpatialData` object.

        Parameters
        ----------
        nucs : list[str] | None, optional
            The nuclear markers, by default None
        mems : list[str] | None, optional
            The membrane markers, by default None
        overwrite_masks : bool, optional
            Overwrite masks which already exist in the `SpatialData` object, by default False
        """
        self._set_markers(nucs, mems)

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

    def _sum_markers(self, fov_name: str, fov_sd: sd.SpatialData) -> si.SpatialImage:
        """Performs groupby-reduce on the `si.SpatialImage` object to sum the nuclear and membrane markers.

        Parameters
        ----------
        fov_name : str
            The name of the field of view to sum channels for.
        fov_sd : sd.SpatialData
            The `SpatialData` object containing the field of view.

        Returns
        -------
        si.SpatialImage
            A `si.SpatialImage` object containing the summed nuclear and / or membrane markers.
        """
        fov_si: si.SpatialImage = fov_sd.images[fov_name]

        labels: si.SpatialImage = xr.where(fov_si.c.isin(self.nuclear_markers), "nucs", "_")
        labels[fov_si.c.isin(self.membrane_markers)] = "mems"
        return (
            fov_si.groupby(group=labels, squeeze=False)
            .sum(
                method="map-reduce",
            )
            .drop_sel({C: "_"})
        )

    async def _run_per_fov_deepcell(
        self,
        fov_name: str,
        fov_sd: sd.SpatialData,
    ) -> SegmentationImageContainer:
        """Runs Deepcell segmentation on a single field of view.

        Parameters
        ----------
        fov_name : str
            The name of the field of view to run Deepcell segmentation on.
        fov_sd : sd.SpatialData
            The `SpatialData` object containing the field of view.

        Returns
        -------
        SegmentationImageContainer
            A container for the segmentation label masks.
        """
        fov_si: si.SpatialImage = self._sum_markers(fov_name, fov_sd)
        async with httpx.AsyncClient(base_url=DEEPCELL_URL) as dc_session:
            segmentation_images: SegmentationImageContainer = await _create_deepcell_input(
                fov_si, dc_session
            )

        return segmentation_images

    def run_cellpose2(
        self,
        nucs: list[str] | None = None,
        mems: list[str] | None = None,
        model_type: str = "cyto2",
        batch_size: int = 8,
    ) -> None:
        """Runs Cellpose2 segmentation on the `SpatialData` object.

        Parameters
        ----------
        nucs : list[str] | None, optional
            The nuclear markers, by default None
        mems : list[str] | None, optional
            the membrane markers, by default None
        model_type : str, optional
            The model to use, by default "cyto2"
        batch_size : int, optional
            The batch size for the model, by default 8
        """
        from cellpose.models import Cellpose

        cyto2_model = Cellpose(model_type=model_type)

        self._set_markers(nucs, mems)

        for fov_name, fov_sd in tqdm(self.sdata.iter_coords):
            fov_si: si.SpatialImage = self._sum_markers(fov_name, fov_sd)
            mask, _ = cyto2_model.eval(
                x=fov_si.data,
                channel_axis=0,
                batch_size=batch_size,
            )
            mask_labels = Labels2DModel.parse(
                data=mask,
                dims=(Y, X),
                transformations={fov_name: Identity()},
            )
            self.sdata.add_labels(name=f"{fov_name}_cytoplasm", labels=mask_labels)
