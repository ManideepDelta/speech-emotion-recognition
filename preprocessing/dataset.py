"""Dataset loader shared by both models. Expects a directory of .wav files
where gender and emotion are encoded in the filename, following the
RAVDESS convention:

    03-01-05-01-02-01-12.wav
             ^^          ^^ emotion code (1=neutral..8=surprise), actor number
                            (even actor number = female, odd = male)
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np

from models.labels import EMOTIONS, LABELS
from preprocessing.features_flat import featurize_file

EMOTION_CODE_MAP = {str(i + 1).zfill(2): emotion for i, emotion in enumerate(EMOTIONS)}


def parse_label(filename: str) -> str:
    """Returns a combined "gender_emotion" label, e.g. "male_happy"."""
    parts = filename.split("-")
    emotion_code = parts[2] if len(parts) > 2 else "01"
    emotion = EMOTION_CODE_MAP.get(emotion_code, "neutral")

    actor_part = parts[6].split(".")[0] if len(parts) > 6 else "01"
    actor_num = int("".join(c for c in actor_part if c.isdigit()) or "1")
    gender = "female" if actor_num % 2 == 0 else "male"

    return f"{gender}_{emotion}"


def load_dataset(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (X, y): X is (n_samples, 20) mean-MFCC vectors, y is integer
    label indices into models.labels.LABELS."""
    features, labels = [], []
    for path in Path(data_dir).rglob("*.wav"):
        label = parse_label(path.name)
        if label not in LABELS:
            continue
        features.append(featurize_file(str(path)))
        labels.append(LABELS.index(label))

    if not features:
        raise RuntimeError(f"No labeled .wav files found under {data_dir}")

    return np.stack(features), np.array(labels)
