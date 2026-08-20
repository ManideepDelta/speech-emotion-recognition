# Speech Emotion Recognition

Two independently trained models that classify gender + emotion from a short voice clip: a **CNN+LSTM** built in Keras/TensorFlow, and an **MLP** built in scikit-learn. Both read the same 20-value mean-MFCC summary of the clip and score all 16 combined classes (e.g. `male_happy`, `female_calm`) independently.

## Pipeline

```
raw audio → mean-MFCC vector (20 values, 2.5s window) → [CNN+LSTM  or  MLP] → 16-class softmax (gender x emotion)
```

## Models

**CNN+LSTM** (Keras) — 3x Conv1D (2048 → 1024 → 512 filters) with BatchNorm/MaxPool after each, then 2x LSTM (256 → 128), then Dense layers (128 → 64 → 32) with dropout, ending in a 16-way softmax.

**MLP** (scikit-learn) — a single `MLPClassifier(hidden_layer_sizes=(2300,), alpha=0.01, batch_size=256, learning_rate='adaptive', max_iter=800)` on the same feature vector, standardized with `StandardScaler`.

## Results

Trained on the full RAVDESS speech set (1,440 clips, 24 actors, 16-class gender+emotion split):

| Model | Test accuracy |
|---|---|
| MLP | 75.7% |
| CNN+LSTM | 63.9% |

The MLP's trained weights (`model/emotion_model.joblib`, 1.3MB) are bundled in this repo, so `/predict/mlp` and the frontend's MLP toggle work immediately on clone. The CNN+LSTM checkpoint is ~170MB — over GitHub's 100MB push limit — so it isn't bundled; run `train_cnn_lstm.py` to reproduce it (takes a few minutes on CPU).

## Getting started

Training requires a labeled dataset with the RAVDESS filename convention (emotion code + actor number in the filename). [RAVDESS](https://zenodo.org/records/1188976) itself is free to download (CC BY-NC-SA 4.0), just not bundled in this repo — download `Audio_Speech_Actors_01-24.zip` and point `--data_dir` at the extracted folder.

```bash
git clone https://github.com/ManideepDelta/speech-emotion-recognition.git
cd speech-emotion-recognition
pip install -r requirements.txt

# train either or both models against your own copy of the dataset
python train_cnn_lstm.py --data_dir /path/to/ravdess --out checkpoints/model.keras
python train_mlp.py --data_dir /path/to/ravdess --out model/emotion_model.joblib

# run inference on a single clip from the CLI
python predict_cnn_lstm.py --audio samples/sample.wav
python predict_mlp.py --audio samples/sample.wav

# or serve both models over HTTP
uvicorn api:app --reload
```

`GET /health` reports each model's load state and test accuracy independently. `POST /predict/cnn-lstm` and `POST /predict/mlp` each accept a `file` upload and return `{top_label, confidence, scores}`.

## Tech stack

`Keras` · `TensorFlow` · `scikit-learn` · `Librosa` · `FastAPI` · vanilla `HTML/CSS/JS` frontend

## Project structure

```
├── models/
│   ├── labels.py          # shared 16-class gender_emotion label set
│   ├── cnn_lstm.py         # Keras CNN+LSTM architecture
│   └── mlp.py               # scikit-learn MLPClassifier config
├── preprocessing/
│   ├── features_flat.py    # mean-MFCC feature extraction
│   └── dataset.py            # RAVDESS filename parsing (gender + emotion)
├── train_cnn_lstm.py / train_mlp.py
├── predict_cnn_lstm.py / predict_mlp.py
├── api.py                     # FastAPI serving both models
└── static/index.html          # frontend with a model toggle
```

## License

MIT
