"""MLPClassifier for speech emotion + gender classification.

Input:  a 20-dim mean-MFCC feature vector
Output: a probability distribution over the 16 gender_emotion labels
"""

from sklearn.neural_network import MLPClassifier


def build_model() -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=(2300,),
        alpha=0.01,
        batch_size=256,
        learning_rate="adaptive",
        max_iter=800,
    )
