import numpy as np

def normalize_keypoints(keypoints):
    """
    Normalizes the keypoints to be invariant to translation and scale.
    Input: flattened array of size 258 (132 pose + 63 lh + 63 rh)
    Output: flattened array of size 258
    """
    # Reshape
    # Pose: 33 landmarks * 4 (x, y, z, visibility)
    pose = keypoints[:132].reshape(33, 4)
    # LH: 21 landmarks * 3 (x, y, z)
    lh = keypoints[132:195].reshape(21, 3)
    # RH: 21 landmarks * 3 (x, y, z)
    rh = keypoints[195:258].reshape(21, 3)
    
    # Check if pose is detected (if all zeros, return as is)
    if np.all(pose == 0):
        return keypoints

    # Define reference points (Shoulders)
    # 11: left_shoulder, 12: right_shoulder
    left_shoulder = pose[11, :2] # x, y
    right_shoulder = pose[12, :2] # x, y
    
    # Calculate center (midpoint of shoulders)
    center = (left_shoulder + right_shoulder) / 2
    
    # Calculate scale (distance between shoulders)
    scale = np.linalg.norm(left_shoulder - right_shoulder)
    
    # Avoid division by zero
    if scale < 0.001:
        scale = 1.0
        
    # Normalize Pose
    pose_norm = pose.copy()
    pose_norm[:, :2] = (pose[:, :2] - center) / scale
    pose_norm[:, 2] = pose[:, 2] / scale # Normalize z with the same scale
    # Visibility (index 3) remains unchanged
    
    # Normalize LH
    lh_norm = lh.copy()
    if not np.all(lh == 0):
        lh_norm[:, :2] = (lh[:, :2] - center) / scale
        lh_norm[:, 2] = lh[:, 2] / scale
        
    # Normalize RH
    rh_norm = rh.copy()
    if not np.all(rh == 0):
        rh_norm[:, :2] = (rh[:, :2] - center) / scale
        rh_norm[:, 2] = rh[:, 2] / scale
        
    # Flatten and concatenate
    return np.concatenate([pose_norm.flatten(), lh_norm.flatten(), rh_norm.flatten()])

def calculate_sequence_variance(sequence):
    """
    Calculates the variance of the keypoints sequence to detect movement.
    Input: List of 30 frames (each frame is a flattened array)
    Output: float (mean variance across dimensions)
    """
    if len(sequence) < 2:
        return 0.0
    
    # Convert to numpy array: (30, 258)
    data = np.array(sequence)
    
    # Calculate variance along the time axis (axis 0) for each feature
    # We only care about x,y coordinates, not z or visibility really, but z is fine too.
    # The keypoints are [x, y, z, v, x, y, z, ...]
    
    # Simple variance of the entire block along time
    # var shape: (258,)
    var = np.var(data, axis=0)
    
    # Return the mean variance across all features
    # (High movement = high mean variance)
    return np.mean(var)

def extract_semantic_features(raw_frame):
    """
    Extracts semantic features from the raw frame (Pose + Face + LH + RH).
    Input: flattened array of size 1662 (132 Pose + 1404 Face + 63 LH + 63 RH)
    Output: flattened array of size 258 (Normalized) + 12 (Semantic) = 270
    """
    # 1. Parse raw frame
    # Pose: 33*4 = 132
    pose = raw_frame[:132].reshape(33, 4)
    # Face: 468*3 = 1404
    face = raw_frame[132:1536].reshape(468, 3)
    # LH: 21*3 = 63
    lh = raw_frame[1536:1599].reshape(21, 3)
    # RH: 21*3 = 63
    rh = raw_frame[1599:1662].reshape(21, 3)
    
    # 2. Get normalized keypoints (Standard 258 features)
    # Construct the input for normalize_keypoints (Pose + LH + RH)
    base_input = np.concatenate([pose.flatten(), lh.flatten(), rh.flatten()])
    normalized_base = normalize_keypoints(base_input)
    
    # 3. Calculate Semantic Features (Distances)
    # We need a scale factor to make these distances invariant
    left_shoulder = pose[11, :2]
    right_shoulder = pose[12, :2]
    scale = np.linalg.norm(left_shoulder - right_shoulder)
    if scale < 0.001: scale = 1.0
    
    features = []
    
    # Helpers
    def get_dist(p1, p2):
        return np.linalg.norm(p1[:2] - p2[:2]) / scale
    
    # Landmarks of interest
    nose = pose[0]
    shoulder_center = (left_shoulder + right_shoulder) / 2
    # Face landmarks (Mediapipe Face Mesh)
    # 1: Nose tip, 61: Mouth corner left, 291: Mouth corner right, 152: Chin, 10: Forehead
    # Note: Face mesh indices might vary, but 152 is usually chin, 10 is forehead.
    # If face is all zeros, use Pose Nose as fallback
    
    has_face = not np.all(face == 0)
    chin = face[152] if has_face else nose # Fallback to nose
    forehead = face[10] if has_face else nose # Fallback
    
    # Hand locations
    lh_wrist = lh[0]
    rh_wrist = rh[0]
    
    has_lh = not np.all(lh == 0)
    has_rh = not np.all(rh == 0)
    
    # Feature 1-4: Wrist to Head locations
    features.append(get_dist(lh_wrist, nose) if has_lh else 0)
    features.append(get_dist(rh_wrist, nose) if has_rh else 0)
    features.append(get_dist(lh_wrist, chin) if has_lh else 0)
    features.append(get_dist(rh_wrist, chin) if has_rh else 0)
    features.append(get_dist(lh_wrist, forehead) if has_lh else 0)
    features.append(get_dist(rh_wrist, forehead) if has_rh else 0)
    
    # Feature 7-8: Wrist to Chest (Shoulder Center)
    # Expand shoulder_center to 3D (z=0 for simplicity or use mean z)
    sc_3d = np.array([shoulder_center[0], shoulder_center[1], 0]) 
    features.append(get_dist(lh_wrist, sc_3d) if has_lh else 0)
    features.append(get_dist(rh_wrist, sc_3d) if has_rh else 0)
    
    # Feature 9-10: Pinch Distance (Thumb Tip 4 to Index Tip 8)
    features.append(get_dist(lh[4], lh[8]) if has_lh else 0)
    features.append(get_dist(rh[4], rh[8]) if has_rh else 0)
    
    # Feature 11-12: Fist vs Open (Mean dist of tips to wrist)
    # Tips: 8, 12, 16, 20
    def get_hand_openness(hand):
        if np.all(hand == 0): return 0
        tips = [8, 12, 16, 20]
        dists = [get_dist(hand[i], hand[0]) for i in tips]
        return np.mean(dists)
        
    features.append(get_hand_openness(lh))
    features.append(get_hand_openness(rh))
    
    # Feature 13-22: Individual Finger States (Dist Tip to Wrist)
    # Thumb: 4, Index: 8, Middle: 12, Ring: 16, Pinky: 20
    finger_tips = [4, 8, 12, 16, 20]
    
    def get_finger_states(hand):
        if np.all(hand == 0): return [0.0] * 5
        states = []
        for tip in finger_tips:
            states.append(get_dist(hand[tip], hand[0]))
        return states

    features.extend(get_finger_states(lh))
    features.extend(get_finger_states(rh))
    
    # Feature 23-24: Arm Angles (Elbow Extension)
    # Pose indices: Left(11,13,15), Right(12,14,16) (Shoulder, Elbow, Wrist)
    def get_angle(a, b, c):
        # Angle at b
        v1 = a[:2] - b[:2]
        v2 = c[:2] - b[:2]
        # Normalize
        nm1 = np.linalg.norm(v1)
        nm2 = np.linalg.norm(v2)
        if nm1 < 0.001 or nm2 < 0.001: return 0.0
        
        cosine_angle = np.dot(v1, v2) / (nm1 * nm2)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle) / 180.0 # Normalize to 0-1
        
    features.append(get_angle(pose[11], pose[13], pose[15])) # Left Arm
    features.append(get_angle(pose[12], pose[14], pose[16])) # Right Arm
    
    # Total new features: 12 (previous) + 10 (fingers) + 2 (arms) = 24
    # Combined size: 258 + 24 = 282
    return np.concatenate([normalized_base, np.array(features)])

def check_gesture_rules(action, raw_frame):
    """
    Validates if the current frame physically matches the expected gesture rules.
    Returns True if valid or no rule exists, False if rule violated.
    """
    # Parse frame (Same as extract_semantic_features)
    pose = raw_frame[:132].reshape(33, 4)
    face = raw_frame[132:1536].reshape(468, 3)
    lh = raw_frame[1536:1599].reshape(21, 3)
    rh = raw_frame[1599:1662].reshape(21, 3)
    
    has_lh = not np.all(lh == 0)
    has_rh = not np.all(rh == 0)
    has_face = not np.all(face == 0)
    
    if not (has_lh or has_rh):
        return False # No hands, no gesture
        
    # Helpers
    def get_dist_2d(p1, p2):
        return np.linalg.norm(p1[:2] - p2[:2])
        
    def is_hand_near(target, threshold=0.15):
        if has_lh and get_dist_2d(lh[0], target) < threshold: return True
        if has_rh and get_dist_2d(rh[0], target) < threshold: return True
        return False

    def is_above_shoulder():
        # y increases downwards. Lower y is higher.
        ls_y = pose[11][1]
        rs_y = pose[12][1]
        if has_lh and lh[0][1] < ls_y: return True
        if has_rh and rh[0][1] < rs_y: return True
        return False
        
    def is_hand_open(threshold=0.1): # High value = open
        # Simple check: Avg dist of tips to wrist
        tips = [8, 12, 16, 20]
        if has_lh:
            d = np.mean([get_dist_2d(lh[i], lh[0]) for i in tips])
            if d > threshold: return True
        if has_rh:
            d = np.mean([get_dist_2d(rh[i], rh[0]) for i in tips])
            if d > threshold: return True
        return False

    # Key Landmarks
    chin = face[152] if has_face else pose[0]
    nose = face[1] if has_face else pose[0]
    forehead = face[10] if has_face else pose[0]
    # Mouth (Upper lip 13, Lower 14 in FaceMesh) - roughly center of face bottom
    mouth = face[13] if has_face else nose
    
    # Chest approx (midpoint of shoulders)
    chest = (pose[11] + pose[12]) / 2

    # Rules
    if action == 'hello':
        return is_above_shoulder() and is_hand_open(0.1)
        
    elif action == 'thanks':
        return is_hand_near(chin, 0.2)
        
    elif action == 'stupid':
        return is_hand_near(forehead, 0.2)
        
    elif action == 'drink':
        return is_hand_near(mouth, 0.2)
        
    elif action == 'please':
        return is_hand_near(chest, 0.25) and is_hand_open(0.08)
        
    elif action == 'sorry':
        return is_hand_near(chest, 0.25) and not is_hand_open(0.08) # Fist on chest
        
    elif action == 'fine':
        # Thumbs up: Thumb tip higher than other tips?
        # Or just checking if hand is present. 'fine' is static.
        return True # Hard to define strictly without rotation
        
    elif action == 'yes':
        return not is_hand_open(0.08) # Fist
        
    elif action == 'no':
        # Pinching?
        return True
        
    elif action == 'love':
        # Pinch thumb and index
        # Dist between tip 4 and 8
        def is_pinch(threshold=0.05):
            if has_lh and get_dist_2d(lh[4], lh[8]) < threshold: return True
            if has_rh and get_dist_2d(rh[4], rh[8]) < threshold: return True
            return False
        return is_pinch()
        
    elif action == 'look':
        return is_hand_near(nose, 0.2) # Eyes/Nose area
        
    elif action == 'I':
        # Pointing to body (chest)
        return is_hand_near(chest, 0.25)
        
    elif action == 'you':
        # Pointing to camera (Arm extended)
        # Elbow angle should be large (close to 180 aka 1.0)
        # We calculated angles in features, but here we have raw frames.
        # Let's trust the model for 'you' but maybe check if hand is NOT near body
        return not is_hand_near(chest, 0.2) and not is_hand_near(chin, 0.2)
        
    elif action == 'am':
         # Pinky check? (Difficult with simple distance)
         return True
         
    elif action == 'are':
         # Index check?
         return True
         
    return True
