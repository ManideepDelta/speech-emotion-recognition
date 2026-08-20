"""Shared label set: gender + emotion combined, as in the original project.

RAVDESS encodes 8 emotions per clip; gender is derived from the actor
number (even = female, odd = male). Combined, that's 16 classes like
"male_happy" or "female_surprise".
"""

GENDERS = ["female", "male"]
EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fear", "disgust", "surprise"]

# Sorted to match sklearn's OneHotEncoder category ordering (lexicographic),
# so training and inference always agree on class order.
LABELS = sorted(f"{gender}_{emotion}" for gender in GENDERS for emotion in EMOTIONS)
