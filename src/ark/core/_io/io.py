from dataclasses import dataclass
from pathlib import Path

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
from dask import delayed
from dask.distributed import get_client, as_completed
from functools import partial
from tqdm.auto import tqdm


@dataclass
class _fov:
    name: str
    image: SpatialImage


def load_cohort(
    cohort_dir: Path,
    save_dir: Path,
    array_type: str = "numpy",
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

    spatial_data = sd.SpatialData()

    spatial_data.write(save_dir)

    futures = client.map(lambda fov: partial(convert_fov, array_type=array_type)(fov), fovs)

    for future, result in tqdm(as_completed(futures, with_results=True), total=len(fovs)):
        spatial_data.add_image(name=result.name, image=result.image)

    return spatial_data


def convert_fov(fov: Path, array_type: Path) -> _fov:
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
    _fov
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

    return _fov(fov.stem, fov_si)
