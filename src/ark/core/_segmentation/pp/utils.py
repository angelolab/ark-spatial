import pathlib
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile

import ray
from fastapi import FastAPI, File
from multiscale_spatial_image.multiscale_spatial_image import MultiscaleSpatialImage
from pydantic import BaseModel
from skimage.io import imsave
from spatial_image import SpatialImage

app = FastAPI()


@ray.remote
def _create_deepcell_input(fov: SpatialImage | MultiscaleSpatialImage):
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_fov_dir = pathlib.Path(tmpdir) / fov.name
        temp_fov_dir.mkdir()

        spaital_data_to_fov(fov, temp_fov_dir, compute=True)

        zip_path = zip_input_files(fov, temp_fov_dir)
        response_json = upload_to_deepcell(zip_path)
    return response_json


def _save_image(fov_chan: SpatialImage, save_dir: pathlib.Path, plugin_args):
    imsave(
        save_dir / f"{fov_chan.c.values.tolist()}.tiff",
        fov_chan.squeeze(),
        check_contrast=False,
        **plugin_args,
    )
    return fov_chan.c


def spaital_data_to_fov(fov: SpatialImage, save_dir: pathlib.Path, compute=False):
    plugin_args: dict[str, str | dict] = {
        "compression": "zlib",
        "compressionargs": {"level": 7},
    }
    for fov_chan in fov:
        _save_image(fov_chan, save_dir, plugin_args)


def zip_input_files(fov: SpatialImage, fov_temp_dir: pathlib.Path):
    # write all files to the zip file
    zip_path = fov_temp_dir.parent / f"{fov_temp_dir.name}.zip"

    # create zip files, skip any existing
    if not zip_path.exists():
        with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as zipObj:
            fov_tiffs = fov_temp_dir.glob("*.tiff")

            for fov_tiff in fov_tiffs:
                zipObj.write(fov_tiff, fov_tiff.stem)

    return zip_path


"""
https://stackoverflow.com/questions/63872924/how-can-i-send-an-http-request-from-my-fastapi-app-to-another-site-api
"""


class FOV_DeepCell_Upload(BaseModel):
    name: str
    file: File
    price: float
    tax: float | None = None


app = FastAPI()


@app.post("/items/")
async def upload_to_deepcell(item: FOV_DeepCell_Upload):
    return item
