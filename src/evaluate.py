from model import get_model
import tensorflow as tf
from tensorflow import keras
import argparse
import shutil
from pathlib import Path
import yaml


model = get_model()

parser = argparse.ArgumentParser("Defining and training the model, tracking metrics with MLFlow.")
parser.add_argument('data')
parser.add_argument("outdir")
parser.add_argument("--params", "-p", default='params.yaml')
args = parser.parse_args()

outdir = Path(args.outdir)
shutil.rmtree(outdir, ignore_errors=True)
outdir.mkdir(parents=True, exist_ok=True)

filename = outdir / "weights.h5"

with open(args.params, 'r') as f:
    params = yaml.safe_load(f)['evaluate']

train_ds = tf.keras.utils.image_dataset_from_directory(
  args.data,
  validation_split= params['validation_split'],
  subset="training",
  seed=params['seed'],
  image_size=(128, 128),
  color_mode = "grayscale",
  label_mode = "binary",
  batch_size=params['batch_size'])

val_ds = tf.keras.utils.image_dataset_from_directory(
  args.data,
  validation_split=params['validation_split'],
  subset="validation",
  seed=params['seed'],
  image_size=(128, 128),
  color_mode = "grayscale",
  label_mode = "binary",
  batch_size=params['batch_size'])

train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

initial_learning_rate = params['initial_learning_rate'] # from https://doi.org/10.3938/jkps.75.597
#mlflow.log_param("initial_learning_rate", initial_learning_rate)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=params['decay_rate'], staircase=True
)
model.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=['acc'],
)

# Define callbacks.
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    filename, save_best_only=True
)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=params["patience"])

# Train the model, doing validation at the end of each epoch
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=params["max_epochs"],
    shuffle=True,
    verbose=1,
    callbacks=[checkpoint_cb, early_stopping_cb],
)
#tf.keras.backend.clear_session()
#mlflow.log_artifacts(str(Path(filename).parent))

#model.load_weights(filename)
#metrics = model.evaluate(x_test, y_test)
#mlflow.log_metric("val_loss", metrics[0])
#mlflow.log_metric("val_accuracy", metrics[1])

#predictions = model.predict(x_test)
#matrix = tf.math.confusion_matrix(y_test, predictions)
#t_n, f_p, f_n, t_p = matrix.numpy().ravel()
#mlflow.log_metric("confusion.tn", t_n)
#mlflow.log_metric("confusion.fp", f_p)
#mlflow.log_metric("confusion.fn", f_n)
#mlflow.log_metric("confusion.tp", t_p)