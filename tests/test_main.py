import sys
from pathlib import Path
sys.path.insert(1, str((Path(__file__).parent / '../src').resolve()))

from main import app, classify
import pytest
from fastapi.testclient import TestClient
import random
import nibabel as nib
from tempfile import NamedTemporaryFile
import numpy as np
from contextlib import contextmanager

client = TestClient(app)
frames = Path(__file__).parent / '../data/frames'
scans = Path(__file__).parent / '../data/prepared/scans'
avail_frames = list(frames.glob('*/*.png'))
avail_scans = list(scans.glob('*.nii'))

@pytest.fixture
@contextmanager
def random_scan():
    with NamedTemporaryFile(delete=True, suffix='.nii') as tmp:
        scan = nib.Nifti1Image(np.random.rand(128,128,20,30), affine=np.eye(4))
        nib.save(scan, tmp.name)
        tmp.seek(0)
        yield tmp

def test_root():
    response = client.get('/')
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict():
    response = client.get('/predict')
    assert response.status_code == 405
    
    with open(random.choice(avail_frames), 'rb') as f:
        response = client.post('/predict', files={'image': f})
        assert response.status_code == 200
        json = response.json()
        assert "probability" in json
        assert type(json["probability"]) == float

    with open(random.choice(avail_frames), 'rb') as f:
        response = client.post('/predict?format=txt', files={'image': f})
        assert response.status_code == 200
    with open(random.choice(avail_frames), 'rb') as f:
        response = client.post('/predict?format=html', files={'image': f})
        assert response.status_code == 422

def test_report(random_scan):
    response = client.get('/report')
    assert response.status_code == 405

    with open(random.choice(avail_scans), 'rb') as f:
        response = client.post('/report', files={'scan': f})
        assert response.status_code == 200
        json = response.json()
        assert "probabilities" in json
        assert "final_probability" in json
        assert type(json['probabilities']) == list
        assert all(type(x) == float for x in json['probabilities'])
        assert type(json['final_probability']) == float

    with open(random.choice(avail_scans), 'rb') as f:
        response = client.post('/report?format=txt', files={'scan': f})
        assert response.status_code == 200

    with open(random.choice(avail_scans), 'rb') as f:
        response = client.post('/report?format=html', files={'scan': f})
        assert response.status_code == 200

    with random_scan as tmp:
        response = client.post('/report?format=html', files={'scan': tmp})
        assert response.status_code == 200
