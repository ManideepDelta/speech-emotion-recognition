"""Train the MLP gender+emotion classifier.

    python train_mlp.py --data_dir data/ravdess --out model/emotion_model.joblib
"""

import argparse
import os

import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models.labels import LABELS
from models.mlp import build_model
from preprocessing.dataset import load_dataset


def train(data_dir: str, test_size: float, out_path: str):
    X, y = load_dataset(data_dir)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    model = build_model()
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"test accuracy: {accuracy:.4f}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler, "labels": LABELS, "accuracy": accuracy}, out_path)
    print(f"saved model to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--out", default="model/emotion_model.joblib")
    args = parser.parse_args()

    train(args.data_dir, args.test_size, args.out)
