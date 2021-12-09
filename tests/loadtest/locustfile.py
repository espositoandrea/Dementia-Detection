"""
Run load tests:

locust -f load-test/locustfile.py --host http://127.0.0.1:8080
"""

from locust import HttpUser, task, between
from pathlib import Path
import random

frames = Path(__file__).parent / '../../data/frames'
scans = Path(__file__).parent / '../../data/prepared/scans'
avail_frames = list(frames.glob('*/*.png'))
avail_scans = list(scans.glob('*.nii'))

class DementiaPredictionUser(HttpUser):
    wait_time = between(1, 5)

    @task(1)
    def healthcheck(self):
        self.client.get("/")

    @task(3)
    def predict(self):
        with open(random.choice(avail_frames), 'rb') as f:
            self.client.post('/predict', files={
                'image': f
            })

    @task(9)
    def report(self):
        with open(random.choice(avail_scans), 'rb') as f:
            self.client.post('/report', files={
                'scan': f
            })
