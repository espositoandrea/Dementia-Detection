from model import get_model
import tensorflow as tf
from tensorflow import keras
import argparse
import shutil
from pathlib import Path
import yaml
import numpy as np
import json
import pytest


model = get_model()

parser = argparse.ArgumentParser(
    "Defining and training the model, tracking metrics"
)
parser.add_argument('data')
parser.add_argument("outdir")
parser.add_argument("--params", "-p", default='params.yaml')
args = parser.parse_args()

outdir = Path(args.outdir)
#shutil.rmtree(outdir, ignore_errors=True)
#outdir.mkdir(parents=True, exist_ok=True)

filename = outdir / "weights.h5"

with open(args.params, 'r') as f:
    params = yaml.safe_load(f)['evaluate']


val_ds = tf.keras.utils.image_dataset_from_directory(
    args.data,
    validation_split=params['validation_split'],
    subset="validation",
    seed=params['seed'],
    shuffle=True,
    image_size=(128, 128),
    color_mode="grayscale",
    label_mode="binary",
    batch_size=params['batch_size']
)

val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

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

# Define callbacks.
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    filename, save_best_only=True
)
early_stopping_cb = keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=params["patience"]
)


# tf.keras.backend.clear_session()
# mlflow.log_artifacts(str(Path(filename).parent))

model.load_weights(filename)
metrics = model.evaluate(val_ds)
x_test = np.concatenate([x for x,y in val_ds], axis=0)
y_test = np.concatenate([y for x,y in val_ds], axis=0)
predictions = model.predict(x_test)
matrix = tf.math.confusion_matrix(y_test.flatten(), tf.round(predictions.flatten()))
t_n, f_p, f_n, t_p = matrix.numpy().ravel()
precision = t_p / (t_p + f_p)
recall = t_p / (t_p + f_n)
f1 = 2*(precision*recall)/(precision+recall)

metrics_dictionary = dict(zip(model.metrics_names, metrics))
metrics_dictionary['precision'] = precision
metrics_dictionary['recall'] = recall
metrics_dictionary['f1'] = f1
with open(outdir / "scores.json", "w") as f:
    json.dump(metrics_dictionary, f, indent=4)
    print(metrics_dictionary)

# Assert metrics
assert metrics_dictionary['accuracy'] >= 0.5
assert metrics_dictionary['precision'] >= 0.45
assert metrics_dictionary['recall'] >= 0.7
assert metrics_dictionary['f1'] >= 0.5

# Check precision, recall and f1
#assert precision >= x
#assert recall >= x
#assert f1 >= x

# mlflow.log_metric("val_loss", metrics[0])
# mlflow.log_metric("val_accuracy", metrics[1])

# mlflow.log_metric("confusion.tn", t_n)
# mlflow.log_metric("confusion.fp", f_p)
# mlflow.log_metric("confusion.fn", f_n)
# mlflow.log_metric("confusion.tp", t_p)