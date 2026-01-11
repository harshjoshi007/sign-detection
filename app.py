# =======================
# MUST BE FIRST (NO EXCEPTIONS)
# =======================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# =======================
# STANDARD IMPORTS
# =======================
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import av

from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# =======================
# SAFE CONFIG IMPORT
# =======================
try:
    from config import actions
except Exception as e:
    st.error(f"Failed to load config/actions: {e}")
    st.stop()

from utils import (
    extract_semantic_features,
    check_gesture_rules,
    calculate_sequence_variance
)

from inference import (
    mediapipe_detection,
    draw_styled_landmarks,
    extract_keypoints
)

# =======================
# LOAD MODEL (SAFE)
# =======================
@st.cache_resource
def load_seq_model():
    if not os.path.exists("action.h5"):
        st.error("Model file 'action.h5' not found.")
        return None
    return load_model("action.h5")

model = load_seq_model()
if model is None:
    st.stop()

# =======================
# MEDIAPIPE SINGLETON (CRITICAL FIX)
# =======================
mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =======================
# VIDEO PROCESSOR
# =======================
class SignLanguageProcessor(VideoTransformerBase):
    def __init__(self):
        self.sequence = []
        self.sentence = []
        self.predictions = []
        self.threshold = 0.95
        self.frames_without_hands = 0
        self.motion_threshold = 0.001
        self.holistic = holistic_model

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # MediaPipe detection
        image, results = mediapipe_detection(img, self.holistic)
        draw_styled_landmarks(image, results)

        # Feature extraction
        raw_keypoints = extract_keypoints(results)
        processed_keypoints = extract_semantic_features(raw_keypoints)

        self.sequence.append(processed_keypoints)
        self.sequence = self.sequence[-30:]

        # Hand presence check
        if results.left_hand_landmarks or results.right_hand_landmarks:
            self.frames_without_hands = 0
        else:
            self.frames_without_hands += 1

        # Prediction logic
        if len(self.sequence) == 30 and self.frames_without_hands < 10:
            variance = calculate_sequence_variance(self.sequence)

            if variance > self.motion_threshold:
                res = model.predict(
                    np.expand_dims(self.sequence, axis=0),
                    verbose=0
                )[0]

                self.predictions.append(np.argmax(res))

                unique, counts = np.unique(
                    self.predictions[-10:], return_counts=True
                )

                if len(unique) > 0:
                    best_idx = unique[np.argmax(counts)]
                    count = counts.max()

                    if count > 8 and res[best_idx] > self.threshold:
                        action = actions[best_idx]
                        if check_gesture_rules(action, raw_keypoints):
                            if not self.sentence or action != self.sentence[-1]:
                                self.sentence.append(action)

                # Probability overlay
                top_3 = np.argsort(res)[-3:][::-1]
                y = 100
                for idx in top_3:
                    prob = res[idx]
                    label = actions[idx]
                    color = (0, 255, 0) if prob > self.threshold else (255, 255, 255)
                    cv2.putText(
                        image,
                        f"{label}: {prob:.2f}",
                        (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2
                    )
                    y += 30

                cv2.putText(
                    image,
                    f"Motion: {variance:.4f}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )
            else:
                cv2.putText(
                    image,
                    "STATIC",
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
                self.predictions.clear()
        else:
            cv2.putText(
                image,
                "NO HANDS",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        # Reset logic
        if self.frames_without_hands > 60:
            self.sequence.clear()
            self.predictions.clear()

        if len(self.sentence) > 5:
            self.sentence = self.sentence[-5:]

        # Sentence overlay
        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(
            image,
            " ".join(self.sentence),
            (3, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        return av.VideoFrame.from_ndarray(image, format="bgr24")

# =======================
# STREAMLIT UI
# =======================
st.set_page_config(page_title="Sign Language Detection", layout="centered")

st.title("🤟 Sign Language Detection")
st.write("Real-time sign language recognition using MediaPipe + LSTM")

webrtc_streamer(
    key="sign-language",
    video_processor_factory=SignLanguageProcessor,
    async_processing=True
)
