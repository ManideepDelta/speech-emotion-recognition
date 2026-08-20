"""Run inference on a single audio clip with the CNN+LSTM model.

    python predict_cnn_lstm.py --audio samples/sample.wav --checkpoint checkpoints/model.keras
"""

import argparse
import os

import joblib
import keras

from models.labels import LABELS
from preprocessing.features_flat import featurize_file


def predict(audio_path: str, checkpoint: str):
    model = keras.models.load_model(checkpoint)  # trusted: trained locally by train_cnn_lstm.py

    scaler_path = os.path.join(os.path.dirname(checkpoint) or ".", "scaler.joblib")
    scaler = joblib.load(scaler_path)["scaler"]  # trusted: trained locally by train_cnn_lstm.py

    features = featurize_file(audio_path).reshape(1, -1)
    features = scaler.transform(features).reshape(1, -1, 1)

    probs = model.predict(features, verbose=0)[0]
    return sorted(zip(LABELS, probs), key=lambda x: -x[1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/model.keras")
    args = parser.parse_args()

    results = predict(args.audio, args.checkpoint)
    print(f"\npredicted: {results[0][0]}  ({results[0][1]*100:.1f}%)\n")
    for label, prob in results:
        bar = "█" * int(prob * 30)
        print(f"  {label:>16} {prob*100:5.1f}%  {bar}")
