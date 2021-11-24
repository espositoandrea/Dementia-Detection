from numpy.core.numeric import outer
import yaml
import pandas as pd
import argparse
from pathlib import Path
import random
from scipy.ndimage.interpolation import rotate
import numpy as np
import nibabel as nib
import re
import sys
import cv2
import shutil


POSITIVE_CLS = 'positive'
NEGATIVE_CLS = 'negative'


def find_brain_bounding_box(image):
    x,y,w,h = (image.shape[0]+1, image.shape[1]+1, -1, -1)
    for slice in image.T:
        blurred = cv2.GaussianBlur(slice.T, (13, 13), 150)
        blurred *= 255.0 / blurred.max()
        blurred[np.where(blurred < 0)] = 0
        blurred = blurred.astype(np.uint8)
        ret, thres = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        blurred_thres = cv2.medianBlur(thres, 5)
        contours, hierarchy = cv2.findContours(blurred_thres, 1, 2)
        curr_x, curr_y, curr_w, curr_h = cv2.boundingRect(contours[0])
        w = max(w, curr_x + curr_w)
        h = max(h, curr_y + curr_h)
        x = min(x, curr_x)
        y = min(y, curr_y)
    # provide a small padding around the found box:
    x = max(x - 10, 0)
    y = max(y - 10, 0)
    w = min(w + 10, image.shape[0])
    h = min(h + 10, image.shape[1])
    return x, y, w, h


def normalize(volume):
    """Normalize the volume"""
    volume = np.array(volume)
    min = volume.min()
    max = volume.max()
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume


def resize_to_input_shape(img, n_frames=20):
    shape = (128, 128)
    # Starting index to extract the 40 central frames of the third dimension
    index = int(img.shape[2] / 2) - int(n_frames / 2)
    final_img = img[:,:,index : index + n_frames]
    x, y, w, h = find_brain_bounding_box(final_img)
    square_size = max(w - x, h - y)
    to_square = square_size - w + x, square_size - h + y
    miss_x, miss_y = to_square
    x, y = max(int(x - np.floor(miss_x / 2)), 0), max(int(np.floor(y - miss_y / 2)), 0)
    w, h = int(np.floor(w + miss_x / 2)), int(np.floor(h + miss_y / 2))
    return np.array(
        [
            #cv2.resize(frame, shape)
            cv2.resize(frame[x:w,y:h], shape)
            for frame in final_img.T
        ]
    ).T


def process_scan(path, n_frames=20):
    """Read and resize volume"""
    try:
        # Read scan
        volume = nib.load(path).get_fdata()
        # Normalize
        volume = normalize(volume)
        # Resize width, height and depth
        volume = resize_to_input_shape(volume, n_frames=n_frames)
        return volume
    except OSError:
        print("Error in reading", path, "(file could be damaged)", file=sys.stderr)
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "Convert 3D PET scans to a sequence of 2D images to be used in network training"
    )
    parser.add_argument("csv")
    parser.add_argument("scan_folder")
    parser.add_argument("outdir")
    parser.add_argument("--params", "-p", default='params.yaml')
    args = parser.parse_args()

    with open(args.params, 'r') as f:
        params = yaml.safe_load(f)['extractframes']

    dataset = pd.read_csv(args.csv, index_col=0)
    dataset.replace({True: POSITIVE_CLS, False: NEGATIVE_CLS}, inplace=True)

    scan_dir = Path(args.scan_folder)

    outdir = Path(args.outdir)
    shutil.rmtree(outdir, ignore_errors=True)
    outdir.mkdir(parents=True, exist_ok=True)

    for _, row in dataset.iterrows():
        img = process_scan(scan_dir / row['filename'], n_frames=params['n_frames'])
        fold_dir = outdir / row['label']
        fold_dir.mkdir(exist_ok=True)
        for frame in range(img.shape[2]):
            cv2.imwrite(str(fold_dir / f"{row['filename']}_frame{frame}.tiff"), img[:, :, frame])
