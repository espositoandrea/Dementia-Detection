"""Prepare the dataset
This module contains various functions that are useful to prepare the dataset of
3D images. It also exports a CLI that can be used by simply running this file as
a script.
"""

import argparse
import random
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import yaml
from scipy.ndimage.interpolation import rotate

# This rotation fills the empty space in the corners with the mean of colours from the corner-patch


def rotate_img(img, angle, bg_patch=(5, 5)):
    """Rotate a 3D image
    This function rotates a 3D image along its first two axis, fixing any voxels
    that may have got a negative value
    """
    assert len(img.shape) <= 3, "Incorrect image shape"
    rgb = len(img.shape) == 3
    if rgb:
        bg_color = np.mean(img[: bg_patch[0], : bg_patch[1], :], axis=(0, 1))
    else:
        bg_color = np.mean(img[: bg_patch[0], : bg_patch[1]])
    img = rotate(img, angle, reshape=False)
    mask = [img <= 0, np.any(img <= 0, axis=-1)][rgb]
    img[mask] = bg_color
    return img


def transform(img, functions):
    """Apply a set of transformations to a 3D image.
    This function applies a set of transformations to a 3D image. The available
    transformations are 'flipv', 'fliph' and 'rotate:[ANGLE]', and the list of
    transformation is expressed as a string of transformations concatenated by a
    pipe | symbol.
    """
    transformed = img.copy()
    for func in functions.split("|"):
        grp = re.match(r"^rotate:(-?\d+?)$", func)
        if grp:
            angle = int(grp.group(1))
            for i in range(transformed.shape[0]):
                transformed[i, :, :] = rotate_img(transformed[i, :, :], angle)
        elif func == "flipv":
            transformed = transformed[:, ::-1, :]
        elif func == "fliph":
            transformed = transformed[:, :, ::-1]
        else:
            raise ValueError(f"Illegal 'fn' value '{functions}'")
    return transformed


def get_file_name(dirname):
    subj, tracer, _, date = Path(dirname).name.split('_')
    return Path(dirname, f"{subj}_{tracer}_{date}n_moco.4dfp.hdr")


def main():
    """The main CLI entrypoint"""
    parser = argparse.ArgumentParser("Prepare the dataset")
    parser.add_argument("data_folder")
    parser.add_argument("labels")
    parser.add_argument("outdir")
    parser.add_argument("--params", "-p", default='params.yaml')
    args = parser.parse_args()

    ID_COLUMN = 'PUP_PUPTIMECOURSEDATA ID'
    DATA_DIR = Path(args.data_folder)
    OUT_DIR = Path(args.outdir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(args.params, 'r') as config:
        params = yaml.safe_load(config)['prepare']

    random.seed(params['seed'])

    labeled_dataset = pd.read_csv(args.labels)
    ids = [l.name for l in DATA_DIR.glob('*')]
    labeled_dataset = labeled_dataset.loc[labeled_dataset[ID_COLUMN].isin(
        ids)]

    positive = labeled_dataset.loc[labeled_dataset['Label']]
    common_negative = labeled_dataset.loc[(labeled_dataset['Label'] == False) & (
        labeled_dataset['Subject'].isin(positive['Subject'].unique()))]
    remainder = labeled_dataset[~labeled_dataset[ID_COLUMN].isin(
        pd.concat([positive, common_negative])[ID_COLUMN])]
    negative = pd.concat([
        common_negative,
        remainder.sample(max(0, params['num_samples'] - common_negative.shape[0]),
                         random_state=params['seed'])
    ])

    sampled = pd.concat([positive, negative])

    images = [DATA_DIR / get_file_name(i) for i in positive[ID_COLUMN]]
    paths = random.choices(images,
                           k=max(0, negative.shape[0] - positive.shape[0]))
    modimg_df = pd.DataFrame(
        columns=['image', 'transform'], index=np.arange(0, len(paths)))

    (OUT_DIR / 'scans').mkdir(exist_ok=True, parents=True)
    labelled_files = []

    for i, path in enumerate(paths):
        # Need to store both the .nii img and the one with mean,
        # so we can use the .nii for saving later
        nii_img = nib.load(path)
        img = np.mean(nii_img.dataobj, axis=3)
        transformation_functions = random.choices(['rotate', 'flipv', 'fliph'],
                                                  weights=[0.5, 0.25, 0.25],
                                                  k=random.randint(1, 5))
        transformation_functions = [
            s + f':{random.randrange(-30, 30)}' if s == 'rotate' else s for s in transformation_functions]
        transform_string = '|'.join(transformation_functions)
        modimg_df.loc[i] = [path.name, transform_string]
        # Apply the transformation function(s)
        transformed_img = transform(np.array(img), transform_string)
        to_save_img = nib.Nifti1Image(transformed_img, nii_img.affine)
        # Save the new generated image (as: ORIGINALE_modID.4dfp.img)
        nib.save(to_save_img, OUT_DIR / 'scans' /
                 f"{str(path.name).split('.')[0]}_mod{i}.4dfp.nii")
        labelled_files.append(
            (f"{str(path.name).split('.')[0]}_mod{i}.4dfp.nii", True))

    images = [DATA_DIR / get_file_name(i) for i in sampled[ID_COLUMN]]
    for i, path in enumerate(images):
        nii_img = nib.load(path)
        img = np.mean(nii_img.dataobj, axis=3)
        to_save_img = nib.Nifti1Image(transformed_img, nii_img.affine)
        nib.save(to_save_img, OUT_DIR / 'scans' / f"{Path(path).stem}.nii")
        labelled_files.append(
            (f"{Path(path).stem}.nii", sampled.iloc[i]['Label']))

    pd.DataFrame(labelled_files, columns=['filename', 'label']).to_csv(
        OUT_DIR / 'labelled-images.csv')
    modimg_df.to_csv(OUT_DIR / 'modified-images.csv')


if __name__ == "__main__":
    main()
