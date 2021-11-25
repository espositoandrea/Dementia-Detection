from fastapi import FastAPI, Query, UploadFile, File
from fastapi.responses import HTMLResponse
import numpy as np
import nibabel as nib
from pathlib import Path
from tempfile import NamedTemporaryFile
import cv2
import shutil
import tensorflow as tf
from .images2frames import resize_to_input_shape, normalize

model = tf.keras.models.load_model('data/model/memento.h5')


def classify(image):
    return list(map(float, model.predict(image)))


def save_upload_file_tmp(upload_file: UploadFile) -> Path:
    try:
        suffix = Path(upload_file.filename).suffix
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(upload_file.file, tmp)
            tmp_path = Path(tmp.name)
    finally:
        upload_file.file.close()
    return tmp_path


app = FastAPI()


@app.post("/predict")
def predict(
    image: UploadFile = File(...),
    format: str = Query(
        "json",
        regex=r"^txt|json$",
        description="The format in which the output will be returned",
    )
):
    tmp_path = save_upload_file_tmp(image)
    try:
        res = {
            "probability": classify(np.expand_dims(cv2.imread(str(tmp_path), cv2.IMREAD_GRAYSCALE), 0))[0]
        }
    finally:
        tmp_path.unlink()
    return res if format == "json" else res["probability"]


@app.post("/report")
def report(
    scan: UploadFile = File(...),
    format: str = Query(
        "json",
        regex=r"^html|txt|json$",
        description="The format in which the output will be returned",
    )
):
    tmp_path = save_upload_file_tmp(scan)
    try:
        image = nib.load(tmp_path).get_fdata()
        if len(image.shape) == 4:
            image = np.mean(image, axis=3)
        image = normalize(image)
        image = resize_to_input_shape(image, n_frames=20)
        res = {
            "probabilities": classify(np.vstack(np.expand_dims(image[:, :, i], 0) for i in range(20))),
        }
        res["final_probability"] = np.mean(res["probabilities"])
    finally:
        tmp_path.unlink()

    return res


@app.get("/")
def root():
    return HTMLResponse('''
    <form action="/predict" method="post" enctype="multipart/form-data">
        <input type="file" required name="image">
        <input type="submit">
    </form>
    <form action="/report" method="post" enctype="multipart/form-data">
        <input type="file" required name="scan">
        <input type="submit">
    </form>
    ''')
