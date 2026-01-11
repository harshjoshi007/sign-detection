from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os
from config import actions
from utils import extract_semantic_features

# Load data (same as train_model.py)
DATA_PATH = os.path.join('MP_Data') 
no_sequences = 30
sequence_length = 30
label_map = {label:num for num, label in enumerate(actions)}
sequences, labels = [], []

print("Loading data...")
for action in actions:
    for sequence in range(no_sequences):
        file_path = os.path.join(DATA_PATH, action, "{}.npy".format(sequence))
        if os.path.exists(file_path):
            res = np.load(file_path)
            # Apply semantic feature extraction
            features = np.array([extract_semantic_features(frame) for frame in res])
            sequences.append(features)
            labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

print("Loading model...")
model = load_model('action.h5')

print("Predicting...")
yhat = model.predict(X_test)
ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

print("\nAccuracy Score:", accuracy_score(ytrue, yhat))

print("\nConfusion Matrix:")
cm = multilabel_confusion_matrix(ytrue, yhat, labels=range(len(actions)))
# print(cm)

# Print per-class accuracy
for i, action in enumerate(actions):
    # True Positives
    tp = cm[i][1][1]
    # False Positives
    fp = cm[i][0][1]
    # False Negatives
    fn = cm[i][1][0]
    # True Negatives
    tn = cm[i][0][0]
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"{action}: Precision={precision:.2f}, Recall={recall:.2f}")

print("\nCommon Confusions:")
# Simple confusion check
confusions = {}
for true, pred in zip(ytrue, yhat):
    if true != pred:
        pair = (actions[true], actions[pred])
        confusions[pair] = confusions.get(pair, 0) + 1

for (true, pred), count in sorted(confusions.items(), key=lambda x: x[1], reverse=True):
    print(f"True: {true} -> Predicted: {pred} ({count} times)")
