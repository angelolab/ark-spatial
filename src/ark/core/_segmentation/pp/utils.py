import pathlib
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile

import httpx
from multiscale_spatial_image.multiscale_spatial_image import MultiscaleSpatialImage
from skimage.io import imsave
from spatial_image import SpatialImage


async def _create_deepcell_input(
    fov: SpatialImage | MultiscaleSpatialImage, dc_session: httpx.AsyncClient
):
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_fov_dir = pathlib.Path(tmpdir) / fov.name
        temp_fov_dir.mkdir()

        spaital_data_to_fov(fov, temp_fov_dir)

        zip_path = zip_input_files(fov, temp_fov_dir)

        output_zip_path = await upload_to_deepcell(zip_path, dc_session)

        temp_extracted_seg_dir: pathlib.Path = temp_fov_dir / "segmentation"
        temp_extracted_seg_dir.mkdir()
        extract_zip(output_zip_path, temp_extracted_seg_dir)

    return 0


def extract_zip(zip_path: pathlib.Path, save_dir: pathlib.Path):
    with ZipFile(zip_path, "r") as zipObj:
        # for name in zipObj.namelist():
        print(zipObj.filelist)
        zipObj.extractall(save_dir)


def _save_image(fov_chan: SpatialImage, save_dir: pathlib.Path, plugin_args):
    imsave(
        save_dir / f"{fov_chan.name}.tiff",
        fov_chan,
        check_contrast=False,
        **plugin_args,
    )
    return fov_chan.name


def spaital_data_to_fov(fov: SpatialImage, save_dir: pathlib.Path):
    plugin_args: dict[str, str | dict] = {
        "compression": "zlib",
        "compressionargs": {"level": 7},
    }
    _save_image(fov, save_dir, plugin_args)


def zip_input_files(fov: SpatialImage, fov_temp_dir: pathlib.Path):
    # write all files to the zip file
    zip_path = fov_temp_dir.parent / f"{fov_temp_dir.name}.zip"

    # create zip files, skip any existing
    if not zip_path.exists():
        with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as zipObj:
            fov_tiffs = fov_temp_dir.glob("*.tiff")

            for fov_tiff in fov_tiffs:
                # file has .tiff extension
                zipObj.write(filename=fov_tiff, arcname=fov_tiff.name)

    return zip_path


async def upload_to_deepcell(zipped_fov: pathlib.Path, dc_session: httpx.AsyncClient):
    upload_url: str = "/api/upload"
    predict_url: str = "/api/predict"
    redis_url: str = "/api/redis"
    expire_url: str = "/expire"

    upload_response = await dc_session.post(
        url=upload_url,
        files={"file": open(zipped_fov, "rb")},
        timeout=30,
    )
    upload_response.raise_for_status()
    upload_response_json = upload_response.json()

    # Call prediction
    predict_payload = {
        "jobForm": {
            "scale": 1.0,
        },
        "imageName": zipped_fov.name,
        "imageUrl": upload_response_json["imageURL"],
        "jobType": "mesmer",
        "uploadedName": upload_response_json["uploadedName"],
    }

    predict_response = await dc_session.post(url=predict_url, json=predict_payload)

    predict_response.raise_for_status()

    predict_response_json = predict_response.json()

    predict_hash = predict_response_json["hash"]

    redis_payload = {
        "hash": predict_hash,
        "key": ["status", "progress", "output_url", "reason", "failures"],
    }

    # Check redis every 3 seconds
    while (total_time := 0) < 300:
        redis_response = await dc_session.post(url=redis_url, json=redis_payload, timeout=3)
        redis_response.raise_for_status()
        redis_response_json = redis_response.json()

        status = redis_response_json["value"][0]
        if status not in ["done", "waiting", "new"]:
            print(status)
        if status == "done":
            break
        total_time += 3

    deepcell_output = await dc_session.get(redis_response_json["value"][2], follow_redirects=True)
    deepcell_output.raise_for_status()

    deepcell_output_path = zipped_fov.parent / f"{zipped_fov.stem}_deepcell_output.zip"

    with open(deepcell_output_path, mode="wb") as f:
        f.write(deepcell_output.content)

    await dc_session.post(url=expire_url, json={"hash": predict_hash, "expireIn": 90})
    return deepcell_output_path
