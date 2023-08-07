from collections.abc import Generator
from pathlib import Path

import natsort as ns
import spatialdata as sd
from dask_image.imread import imread
from spatialdata.models import (
    C,
    Image2DModel,
    X,
    Y,
)


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
    fovs: Generator[Path] = cohort_dir.glob("*/")
    spatial_data = sd.SpatialData()

    for fov in fovs:
        data = imread(fname=f"{fov.as_posix()}/*.tiff", arraytype=array_type)
        channels: list[str] = ns.natsorted([f.stem for f in fov.glob("*.tiff")])
        spatial_data.add_image(
            name=fov.stem, image=Image2DModel.parse(data=data, dims=(C, Y, X), c_coords=channels)
        )
    return spatial_data
