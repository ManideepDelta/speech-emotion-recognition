"""CNN + LSTM hybrid for speech emotion + gender classification (Keras).

Input:  (batch, 20, 1) -- the 20-dim mean-MFCC vector, reshaped for Conv1D
Conv1D x3: learns spectral patterns at shrinking resolution (2048->1024->512)
LSTM x2:   models how those patterns relate across the coefficient axis
Dense:     narrows down to a 16-way softmax over gender_emotion labels
"""

import tensorflow as tf
from keras import Sequential
from keras.layers import BatchNormalization, Conv1D, Dense, Dropout, LSTM, MaxPooling1D

from models.labels import LABELS

N_FEATURES = 20


def build_model(num_classes: int = len(LABELS)) -> Sequential:
    model = Sequential([
        Conv1D(2048, kernel_size=5, strides=1, padding="same", activation="relu", input_shape=(N_FEATURES, 1)),
        MaxPooling1D(pool_size=2, strides=2, padding="same"),
        BatchNormalization(),

        Conv1D(1024, kernel_size=5, strides=1, padding="same", activation="relu"),
        MaxPooling1D(pool_size=2, strides=2, padding="same"),
        BatchNormalization(),

        Conv1D(512, kernel_size=5, strides=1, padding="same", activation="relu"),
        MaxPooling1D(pool_size=2, strides=2, padding="same"),
        BatchNormalization(),

        LSTM(256, return_sequences=True),
        LSTM(128),

        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(32, activation="relu"),
        Dropout(0.2),

        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
