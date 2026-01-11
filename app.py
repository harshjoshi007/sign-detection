import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
from config import actions
from utils import extract_semantic_features, check_gesture_rules, calculate_sequence_variance

import os

# Load model
@st.cache_resource
def load_seq_model():
    if not os.path.exists('action.h5'):
        st.error("Model file 'action.h5' not found. Please train the model first.")
        return None
    return load_model('action.h5')

model = load_seq_model()

if model is None:
    st.stop()

from inference import mediapipe_detection, draw_styled_landmarks, extract_keypoints, mp_holistic

class SignLanguageProcessor(VideoTransformerBase):
    def __init__(self):
        self.sequence = []
        self.sentence = []
        self.predictions = []
        self.threshold = 0.95
        self.frames_without_hands = 0
        self.motion_threshold = 0.001
        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Make detections
        image, results = mediapipe_detection(img, self.holistic)
        
        # Draw landmarks
        draw_styled_landmarks(image, results)
        
        # Prediction logic
        raw_keypoints = extract_keypoints(results)
        processed_keypoints = extract_semantic_features(raw_keypoints)
        
        self.sequence.append(processed_keypoints)
        self.sequence = self.sequence[-30:]
        
        if results.left_hand_landmarks or results.right_hand_landmarks:
            self.frames_without_hands = 0
        else:
            self.frames_without_hands += 1
            
        if len(self.sequence) == 30 and self.frames_without_hands < 10:
            variance = calculate_sequence_variance(self.sequence)
            
            if variance > self.motion_threshold:
                res = model.predict(np.expand_dims(self.sequence, axis=0), verbose=0)[0]
                self.predictions.append(np.argmax(res))
                
                unique, counts = np.unique(self.predictions[-10:], return_counts=True)
                if len(unique) > 0:
                    best_idx_vote = np.argmax(counts)
                    best_class = unique[best_idx_vote]
                    count = counts[best_idx_vote]
                    
                    if count > 8:
                        if res[best_class] > self.threshold:
                            current_action = actions[best_class]
                            is_valid_gesture = check_gesture_rules(current_action, raw_keypoints)
                            
                            if is_valid_gesture:
                                if len(self.sentence) > 0:
                                    if current_action != self.sentence[-1]:
                                        self.sentence.append(current_action)
                                else:
                                    self.sentence.append(current_action)
                
                # Viz probabilities
                top_3_idx = np.argsort(res)[-3:][::-1]
                y_offset = 100
                for idx in top_3_idx:
                     prob = res[idx]
                     label = actions[idx]
                     color = (0, 255, 0) if prob > self.threshold else (255, 255, 255)
                     text = f"{label}: {prob:.2f}"
                     cv2.putText(image, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                     y_offset += 30
                     
                cv2.putText(image, f"Motion: {variance:.4f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            else:
                cv2.putText(image, "STATIC", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                self.predictions = []
        else:
            cv2.putText(image, "NO HANDS", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if self.frames_without_hands > 60:
             self.sequence = []
             self.predictions = []

        if len(self.sentence) > 5:
            self.sentence = self.sentence[-5:]

        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(self.sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(image, format="bgr24")

st.title("Sign Language Detection")
st.text("Uses MediaPipe and LSTM to detect sign language actions.")

webrtc_streamer(key="example", video_processor_factory=SignLanguageProcessor)
