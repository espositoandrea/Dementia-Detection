from model import get_model
import tensorflow as tf
from tensorflow import keras
import argparse
import shutil
from pathlib import Path
import yaml


model = get_model()

parser = argparse.ArgumentParser("Defining and training the model")
parser.add_argument('data')
parser.add_argument("outdir")
parser.add_argument("--params", "-p", default='params.yaml')
args = parser.parse_args()

outdir = Path(args.outdir)
shutil.rmtree(outdir, ignore_errors=True)
outdir.mkdir(parents=True, exist_ok=True)

with open(args.params, 'r') as f:
    params = yaml.safe_load(f)['train']

train_ds = tf.keras.utils.image_dataset_from_directory(
    args.data,
    seed=params['seed'],
    shuffle=True,
    image_size=(128, 128),
    color_mode="grayscale",
    label_mode="binary",
    batch_size=params['batch_size']
)

train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# mlflow.log_param("initial_learning_rate", initial_learning_rate)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    params['initial_learning_rate'],
    decay_steps=100000,
    decay_rate=params['decay_rate'],
    staircase=True
)
model.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=['accuracy'],
)

# Train the model, doing validation at the end of each epoch
history = model.fit(
    train_ds,
    epochs=params["epochs"],
    shuffle=True,
    verbose=1,
)

# Save the model
model.save(outdir / 'memento.h5')
