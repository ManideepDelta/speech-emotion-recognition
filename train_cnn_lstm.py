"""Train the CNN+LSTM gender+emotion classifier (Keras).

    python train_cnn_lstm.py --data_dir data/ravdess --epochs 50 --out checkpoints/model.keras
"""

import argparse
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models.cnn_lstm import build_model
from preprocessing.dataset import load_dataset


def train(data_dir: str, epochs: int, batch_size: int, out_path: str):
    X, y = load_dataset(data_dir)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train).reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = scaler.transform(X_test).reshape(X_test.shape[0], X_test.shape[1], 1)

    model = build_model()
    model.summary()
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"test accuracy: {accuracy:.4f}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    model.save(out_path)

    scaler_path = os.path.join(os.path.dirname(out_path) or ".", "scaler.joblib")
    import joblib
    joblib.dump({"scaler": scaler, "accuracy": float(accuracy)}, scaler_path)
    print(f"saved model to {out_path} and scaler to {scaler_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--out", default="checkpoints/model.keras")
    args = parser.parse_args()

    train(args.data_dir, args.epochs, args.batch_size, args.out)
