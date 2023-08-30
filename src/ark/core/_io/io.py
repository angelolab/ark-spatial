from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import NewType

import dask.array as da
import natsort as ns
import spatialdata as sd
from dask_image.imread import imread
from mpire import WorkerPool
from spatial_image import SpatialImage
from spatialdata.models import (
    C,
    Image2DModel,
    X,
    Y,
)
from spatialdata.transformations import Identity

FovName = NewType("FovName", str)


@dataclass
class _fov:
    name: FovName
    image: SpatialImage


def load_cohort(
    cohort_dir: Path,
    array_type: str = "numpy",
) -> sd.SpatialData:
    """Load a cohort of images into a SpatialData object.

    Parameters
    ----------
    cohort_dir : Path
        Path to the directory containing the cohort.
    array_type : str, optional
        Array type to use for the image data, by default "numpy".
        Options are "numpy" or "cupy" if cupy is installed.

    Returns
    -------
    SpatialData
        SpatialData object containing the cohort.
    """
    fovs: Generator[Path] = list(cohort_dir.glob("*/"))

    spatial_data = sd.SpatialData()
    # you have so many cores, why not use them?
    with WorkerPool(n_jobs=None, shared_objects=array_type) as pool:
        for fov in pool.imap(
            func=convert_fov,
            iterable_of_args=fovs,
            progress_bar=True,
            progress_bar_style=None,
            progress_bar_options={"unit": "FOV"},
        ):
            spatial_data.add_image(name=fov.name, image=fov.image)

    return spatial_data


def convert_fov(shared_objects: str, fov: Path) -> _fov:
    """
    Convert a single FOV into a SpatialImage.

    Parameters
    ----------
    shared_objects : str
        The array type to use for the image data. Options are "numpy" or "cupy" if cupy is installed.
    fov : Path
        The path to the FOV

    Returns
    -------
    _fov
        A dataclass containing the FOV name and SpatialImage.
    """
    array_type = shared_objects
    data: da.Array = imread(fname=f"{fov.as_posix()}/*.tiff", arraytype=array_type)
    channels: list[str] = ns.natsorted([f.stem for f in fov.glob("*.tiff")])
    fov_si: SpatialImage = Image2DModel().parse(
        data=data,
        dims=(C, Y, X),
        c_coords=channels,
        transformations={
            fov.stem: Identity(),  # Per FOV coordinate system
            "global": Identity(),  # Global coordinate system
        },
    )

    return _fov(fov.stem, fov_si)
