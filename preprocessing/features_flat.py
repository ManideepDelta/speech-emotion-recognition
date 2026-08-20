"""Audio -> flat MFCC feature vector, matching the original project's
feature extraction exactly: a fixed 2.5s window starting at a 0.6s offset,
mean-pooled over time, using librosa's default 20 MFCC coefficients.
"""

import numpy as np
import librosa

DURATION = 2.5
OFFSET = 0.6
FEATURE_SIZE = 20  # librosa.feature.mfcc default n_mfcc


def load_audio(path: str) -> tuple[np.ndarray, int]:
    audio, sample_rate = librosa.load(path, duration=DURATION, offset=OFFSET)
    return audio, sample_rate


def extract_features(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate)
    return mfcc.T.mean(axis=0).astype(np.float32)


def featurize_file(path: str) -> np.ndarray:
    audio, sample_rate = load_audio(path)
    return extract_features(audio, sample_rate)
