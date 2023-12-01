import os
from dataclasses import dataclass
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Literal

import dask.array as da
import natsort as ns
import spatialdata as sd
import xarray as xr
from dask.distributed import as_completed, get_client
from dask_image.imread import imread
from spatial_image import SpatialImage
from spatialdata.models import (
    C,
    Image2DModel,
    X,
    Y,
)
from spatialdata.transformations import Identity
from tifffile import imwrite
from tqdm.auto import tqdm

from ark.core.typing import NDArrayA


@dataclass
class Fov:
    name: str
    image: SpatialImage


def load_cohort(
    cohort_dir: Path,
    save_dir: Path,
    array_type: Literal["numpy", "cupy"] = "numpy",
) -> sd.SpatialData:
    """Load a cohort of images into a SpatialData object.

    Parameters
    ----------
    cohort_dir : Path
        Path to the directory containing the cohort.
    array_type : str, optional
        Array type to use for the image data, by default "numpy".
        Options are "numpy" or "cupy" if CuPy is installed.

    Returns
    -------
    sd.SpatialData
        sd.SpatialData object containing the cohort.
    """
    client = get_client()

    fovs: list[Path] = list(cohort_dir.glob("[!.]*/"))

    sdata = sd.SpatialData()

    sdata.write(save_dir)

    futures = client.map(lambda fov: partial(convert_fov, array_type=array_type)(fov), fovs)

    for _, result in tqdm(as_completed(futures, with_results=True), total=len(fovs)):
        sdata.add_image(name=result.name, image=result.image)

    return sdata


def convert_fov(fov: Path, array_type: str) -> Fov:
    """
    Convert a single FOV into a SpatialImage.

    Parameters
    ----------
    array_type : str
        The array type to use for the image data. Options are "numpy" or "cupy" if cupy is installed.
    fov : Path
        The path to the FOV

    Returns
    -------
    Fov
        A dataclass containing the FOV name and SpatialImage.
    """
    data: da.Array = imread(fname=f"{fov.as_posix()}/*.tiff", arraytype=array_type)
    channels: list[str] = ns.natsorted([f.stem for f in fov.glob("*.tiff")])
    fov_si: SpatialImage = Image2DModel().parse(
        data=data,
        dims=(C, Y, X),
        c_coords=channels,
        transformations={
            fov.stem: Identity(),  # Per FOV coordinate system
        },
    )

    return Fov(fov.stem, fov_si)


class SDataComponents(str, Enum):
    """
    Components of a SpatialData object.

    asdf
    """

    IMAGES = "images"
    LABELS = "labels"


def sdata_to_tiffs(
    sdata: sd.SpatialData,
    component: tuple[SDataComponents, ...],
    save_dir: os.PathLike,
):
    """
    Convert the Spatial Images in a SpatialData object to TIFFs or OME-TIFFs.

    sdf
    """
    if component not in [SDataComponents.IMAGES, SDataComponents.LABELS]:
        raise ValueError(
            f"Invalid components {component}. Must be one of [{SDataComponents.images}, {SDataComponents.labels}]"
        )

    if not Path(save_dir).exists():
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    tiff_out = partial(_to_tiff_ufunc, save_dir=save_dir, component=component)

    for fov_id, fov_sd in tqdm(sdata.iter_coords(dataloader=True)):
        tiff_out(fov_id, fov_sd)


def _image_to_tiff(
    image: NDArrayA,
    fov_dir: Path,
    fov_id: str,
    channels: list[str],
):
    """
    Convert a SpatialImage to a TIFF
    """

    for channel, channel_data in zip(channels, image, strict=True):
        imwrite(
            fov_dir / f"{channel}.tiff",
            data=channel_data,
            compression="zlib",
            compressionargs={"level": 6},
        )
    return [fov_id]


def _label_to_tiff(
    label: NDArrayA,
    label_dir: Path,
    label_id: str,
):
    """
    Convert a SpatialImage to a TIFF
    """
    imwrite(
        label_dir / f"{label_id}.tiff",
        data=label,
        compression="zlib",
        compressionargs={"level": 6},
    )
    return [label_id]


def _images_to_tiff_ufunc(si: SpatialImage, fov_id: str, save_dir: Path):
    image_dir = save_dir / "images"
    fov_dir = image_dir / fov_id
    if not fov_dir.exists():
        fov_dir.mkdir(parents=True, exist_ok=True)

    xr.apply_ufunc(
        _image_to_tiff,
        si,
        vectorize=False,
        input_core_dims=[[C, Y, X]],
        output_core_dims=[["fov"]],
        kwargs={
            "fov_dir": fov_dir,
            "fov_id": fov_id,
            "channels": list(si.c.values),
        },
        dask="allowed",
    )


def _labels_to_tiff_ufunc(si: SpatialImage, label_id: str, save_dir: Path):
    label_dir = save_dir / "labels"
    if not label_dir.exists():
        label_dir.mkdir(parents=True, exist_ok=True)

    xr.apply_ufunc(
        _label_to_tiff,
        si,
        vectorize=False,
        input_core_dims=[[Y, X]],
        output_core_dims=[["fov"]],
        kwargs={
            "label_dir": label_dir,
            "label_id": label_id,
        },
        dask="allowed",
    )


def _to_tiff_ufunc(fov_id: str, fov_sd: sd.SpatialData, component: str, save_dir: os.PathLike):
    """
    Convert a SpatialImage to a TIFF
    """
    if hasattr(fov_sd, component):
        component_keys = getattr(fov_sd, component).keys()
        if component_keys:
            for component_id in component_keys:
                match component:
                    case SDataComponents.IMAGES:
                        si = fov_sd.images[component_id]
                        _images_to_tiff_ufunc(si, fov_id, save_dir)
                    case SDataComponents.LABELS:
                        si = fov_sd.labels[component_id]
                        _labels_to_tiff_ufunc(si, component_id, save_dir)
