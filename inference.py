import cv2
import numpy as np
import os
import mediapipe as mp
from tensorflow.keras.models import load_model

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             )
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

from config import actions
from utils import normalize_keypoints, calculate_sequence_variance, extract_semantic_features, check_gesture_rules

# Actions that we try to detect


if not os.path.exists('action.h5'):
    print("Error: 'action.h5' not found. Please run 'train_model.py' first to train the model.")
    exit()

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame

if __name__ == '__main__':
    model = load_model('action.h5')

    # 1. New detection variables
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.95 # Increased confidence threshold
    frames_without_hands = 0
    motion_threshold = 0.001 # Minimum variance to consider it a gesture

    # Generate random colors for visualization
    colors = [(245,117,16)] * len(actions)

    cap = cv2.VideoCapture(0)
    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()
            if not ret:
                break

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            
            # Draw landmarks
            draw_styled_landmarks(image, results)
            
            # 2. Prediction logic
            raw_keypoints = extract_keypoints(results)
            
            # Apply normalization (Make sure to match training logic)
            # We now use extract_semantic_features which expects the FULL raw frame
            # extract_keypoints now returns Pose+Face+LH+RH
            processed_keypoints = extract_semantic_features(raw_keypoints)
        
            sequence.append(processed_keypoints)
            sequence = sequence[-30:]

            # Smart Hand Check (Relaxed)
            # Just check if hands are present to allow "update"
            if results.left_hand_landmarks or results.right_hand_landmarks:
                frames_without_hands = 0
            else:
                frames_without_hands += 1
            
            # Only predict if hands are present or were recently present
            if len(sequence) == 30 and frames_without_hands < 10:
                # Check for movement/variance
                variance = calculate_sequence_variance(sequence)
                
                # Only predict if there is enough movement (not just static hands)
                if variance > motion_threshold:
                    res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                    predictions.append(np.argmax(res))
                    
                    # Print top prediction to terminal for debugging
                    best_idx = np.argmax(res)
                    # print(f"Pred: {actions[best_idx]} ({res[best_idx]:.2f}) - Var: {variance:.4f}")

                    # 3. Viz logic
                    unique, counts = np.unique(predictions[-10:], return_counts=True)
                    if len(unique) > 0:
                        best_idx_vote = np.argmax(counts)
                        best_class = unique[best_idx_vote]
                        count = counts[best_idx_vote]
                        
                        if count > 8: # Increased stability check (8 out of 10 frames)
                            if res[best_class] > threshold: 
                                current_action = actions[best_class]
                                
                                # NEW: Validate prediction with gesture rules
                                # If the model predicts 'Thanks', we double check if hand is near chin
                                is_valid_gesture = check_gesture_rules(current_action, raw_keypoints)
                                
                                if is_valid_gesture:
                                    if len(sentence) > 0: 
                                        if current_action != sentence[-1]:
                                            sentence.append(current_action)
                                    else:
                                        sentence.append(current_action)
                                else:
                                    # Feedback for debugging
                                    cv2.putText(image, f"RULE FAILED: {current_action}", (10, 140), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Show probabilities on screen (Top 3)
                    # Sort indices by probability
                    top_3_idx = np.argsort(res)[-3:][::-1]
                    y_offset = 100
                    for idx in top_3_idx:
                         prob = res[idx]
                         label = actions[idx]
                         # Color code: Green if > threshold, White otherwise
                         color = (0, 255, 0) if prob > threshold else (255, 255, 255)
                         text = f"{label}: {prob:.2f}"
                         cv2.putText(image, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                         y_offset += 30
                         
                    # Show variance for debugging
                    cv2.putText(image, f"Motion: {variance:.4f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
                else:
                    cv2.putText(image, "STATIC (NO GESTURE)", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    predictions = [] # Reset predictions if static
                    
            else:
                # If no hands, show status
                cv2.putText(image, "NO HANDS DETECTED", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Reset sequence if hands gone for > 2 sec (60 frames)
            # This prevents "ghost" data from staying in the buffer too long
            if frames_without_hands > 60:
                 sequence = []
                 predictions = []

            if len(sentence) > 5: 
                sentence = sentence[-5:]

                # Viz probabilities
                # image = prob_viz(res, actions, image, colors)
                
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Show to screen
            cv2.imshow('OpenCV Feed', image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()
