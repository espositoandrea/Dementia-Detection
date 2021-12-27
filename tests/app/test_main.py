import sys
from pathlib import Path
sys.path.insert(1, str((Path(__file__).parent / '../..').resolve()))
import cv2
from contextlib import contextmanager
import numpy as np
from tempfile import NamedTemporaryFile
import nibabel as nib
from fastapi.testclient import TestClient
import pytest
from src.app.main import app


client = TestClient(app)


@pytest.fixture
@contextmanager
def random_scan():
    with NamedTemporaryFile(delete=True, suffix='.nii') as tmp:
        scan = nib.Nifti1Image(np.random.rand(
            128, 128, 20, 30), affine=np.eye(4))
        nib.save(scan, tmp.name)
        tmp.seek(0)
        yield tmp


@pytest.fixture
@contextmanager
def random_image():
    with NamedTemporaryFile(delete=True, suffix='.png') as tmp:
        scan = np.random.rand(128, 128)
        cv2.imwrite(tmp.name, scan)
        tmp.seek(0)
        yield tmp


def test_root():
    response = client.get('/')
    assert response.status_code == 200
    body = response.json()
    assert "status" in body
    assert "version" in body
    assert body["status"] == "ok"


def test_predict(random_image):
    response = client.get('/predict')
    assert response.status_code == 405

    with random_image as f:
        response = client.post('/predict', files={'image': f})
        assert response.status_code == 200
        json = response.json()
        assert "probability" in json
        assert type(json["probability"]) == float

        f.seek(0)
        response = client.post('/predict?format=txt', files={'image': f})
        assert response.status_code == 200

        f.seek(0)
        response = client.post('/predict?format=html', files={'image': f})
        assert response.status_code == 422


def test_report(random_scan):
    response = client.get('/report')
    assert response.status_code == 405

    with random_scan as f:
        response = client.post('/report', files={'scan': f})
        assert response.status_code == 200
        json = response.json()
        assert "probabilities" in json
        assert "final_probability" in json
        assert type(json['probabilities']) == list
        assert all(type(x) == float for x in json['probabilities'])
        assert type(json['final_probability']) == float

        f.seek(0)
        response = client.post('/report?format=txt', files={'scan': f})
        assert response.status_code == 200

        f.seek(0)
        response = client.post('/report?format=html', files={'scan': f})
        assert response.status_code == 200

        f.seek(0)
        response = client.post('/report?format=html', files={'scan': f})
        assert response.status_code == 200
