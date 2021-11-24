import argparse
import pandas as pd
import re
import yaml
import numpy as np
from pathlib import Path
import random

#%tensorflow_version 2.x
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

parser = argparse.ArgumentParser("Defining and training the model")
parser.add_argument('data')
parser.add_argument("outdir")
parser.add_argument("--params", "-p", default='params.yaml')
args = parser.parse_args()

DATA_DIR = Path(args.prepared_data_folder)
OUT_DIR = Path(args.outdir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(args.params, 'r') as f:
    params = yaml.safe_load(f)['train']

random.seed(params['seed'])

def get_model(width=128, height=128):
    """Build a convolutional neural network model based on the Zunhair et al model.
    References
    ----------
    - https://arxiv.org/abs/2007.13224
    """
    model = tf.keras.Sequential([
        keras.Input((width, height, 1)),    
        layers.Conv2D(filters=64, kernel_size=3, activation="relu"),
        layers.MaxPool2D(pool_size=2),
        layers.BatchNormalization(),

        layers.Conv2D(filters=64, kernel_size=3, activation="relu"),
        layers.MaxPool2D(pool_size=2),
        layers.BatchNormalization(),

        layers.Conv2D(filters=128, kernel_size=3, activation="relu"),
        layers.MaxPool2D(pool_size=2),
        layers.BatchNormalization(),

        layers.Conv2D(filters=256, kernel_size=3, activation="relu"),
        layers.MaxPool2D(pool_size=2),
        layers.BatchNormalization(),

        layers.GlobalAveragePooling2D(),
        layers.Dense(units=512, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(units=1, activation="sigmoid")
    ])

    return model

# Compile model.
def train_model(x_train, y_train, x_test, y_test, epochs=10000, verbose=2, batch_size=32, run_name="run", filename="3d_image_classification.h5"):
    # Build model.
    #mlflow.tensorflow.autolog()
    model = get_model(width=128, height=128)

    train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    validation_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    #mlflow.log_param("batch_size", batch_size)
    train_dataset = (
        train_loader.shuffle(len(x_train), seed=RANDOM_SEED)
        .batch(batch_size)
        .prefetch(128)
    )
    validation_dataset = (
        validation_loader.shuffle(len(x_test), seed=RANDOM_SEED)
        .batch(batch_size)
        .prefetch(128)
    )

    initial_learning_rate = 5e-5 # from https://doi.org/10.3938/jkps.75.597
    #mlflow.log_param("initial_learning_rate", initial_learning_rate)
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
    )
    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=['acc'],
    )

    # Define callbacks.
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        filename, save_best_only=True
    )
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor="acc", patience=35)

    # Train the model, doing validation at the end of each epoch
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        shuffle=True,
        verbose=verbose,
        callbacks=[checkpoint_cb, early_stopping_cb],
    )
    tf.keras.backend.clear_session()
    #mlflow.log_artifacts(str(Path(filename).parent))

    model.load_weights(filename)
    metrics = model.evaluate(x_test, y_test)
    #mlflow.log_metric("val_loss", metrics[0])
    #mlflow.log_metric("val_accuracy", metrics[1])
    
    predictions = model.predict(x_test)
    matrix = tf.math.confusion_matrix(y_test, predictions)
    t_n, f_p, f_n, t_p = matrix.numpy().ravel()
    #mlflow.log_metric("confusion.tn", t_n)
    #mlflow.log_metric("confusion.fp", f_p)
    #mlflow.log_metric("confusion.fn", f_n)
    #mlflow.log_metric("confusion.tp", t_p)

    return history
    return history, metrics, matrix

batchsize = args.batch_size
for k in range(1, 11):
    base = DATA_DIR / f'both-folds/fold-{k}'
    xtrain = np.load(base / 'xtrain.npy')
    ytrain = np.load(base / 'ytrain.npy')
    xtest = np.load(base / 'xtest.npy')
    ytest = np.load(base / 'ytest.npy')
    tf.keras.backend.clear_session()

    results = DATA_DIR / f'both-folds/results/fold-{k}'
    results.mkdir(exist_ok=True, parents=True)
    train_model(xtrain, ytrain, xtest, ytest, batch_size=batchsize, run_name=f"batch{batchsize}-fold{k}", filename=str(results / f"model.h5"), verbose=0)
