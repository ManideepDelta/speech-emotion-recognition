"""Run inference on a single audio clip with the MLP model.

    python predict_mlp.py --audio samples/sample.wav --model model/emotion_model.joblib
"""

import argparse

import joblib

from preprocessing.features_flat import featurize_file


def predict(audio_path: str, model_path: str):
    bundle = joblib.load(model_path)  # trusted: file trained locally by train_mlp.py, not user-supplied
    model, scaler, labels = bundle["model"], bundle["scaler"], bundle["labels"]

    features = featurize_file(audio_path).reshape(1, -1)
    features = scaler.transform(features)
    probs = model.predict_proba(features)[0]

    return sorted(zip(labels, probs), key=lambda x: -x[1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model", default="model/emotion_model.joblib")
    args = parser.parse_args()

    results = predict(args.audio, args.model)
    print(f"\npredicted: {results[0][0]}  ({results[0][1]*100:.1f}%)\n")
    for label, prob in results:
        bar = "█" * int(prob * 30)
        print(f"  {label:>16} {prob*100:5.1f}%  {bar}")
