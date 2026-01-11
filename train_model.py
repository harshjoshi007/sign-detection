from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
import numpy as np
import os

from config import actions
from utils import normalize_keypoints, extract_semantic_features

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data') 

# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30

label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []

for action in actions:
    for sequence in range(no_sequences):
        file_path = os.path.join(DATA_PATH, action, "{}.npy".format(sequence))
        if not os.path.exists(file_path):
            print(f"Error: Data missing for action '{action}', sequence {sequence}.")
            print("Please run 'collect_data.py' to collect all required data.")
            exit()
        res = np.load(file_path)
        # Apply semantic feature extraction to the FULL raw frame
        # res shape is (30, 1662)
        features = np.array([extract_semantic_features(frame) for frame in res])
        
        sequences.append(features)
        labels.append(label_map[action])

        # AUGMENTATION: Add noise to create variations
        # Create 2 augmented versions per sequence
        for _ in range(2):
            noise = np.random.normal(0, 0.05, features.shape)
            augmented_features = features + noise
            sequences.append(augmented_features)
            labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)
es_callback = EarlyStopping(monitor='val_categorical_accuracy', patience=20, verbose=1, mode='max', restore_best_weights=True)
lr_callback = ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.5, patience=5, verbose=1, mode='max', min_lr=0.00001)

model = Sequential()
model.add(Bidirectional(LSTM(64, return_sequences=True, activation='tanh'), input_shape=(30,282)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(128, return_sequences=True, activation='tanh')))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=False, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test), callbacks=[tb_callback, es_callback, lr_callback])

model.summary()

model.save('action.h5')
