#!/usr/bin/env python3
"""
Simple test script for the pose detection model without Flask
This script tests the core functionality using your webcam directly
"""

import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
import time

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Load the trained model
MODEL_PATH = os.path.join('models', 'pose_classification_model.joblib')
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully!")
    print(f"Model classes: {model.classes_}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# Define landmarks indices
LANDMARKS = {
    'LEFT_SHOULDER': 11, 'RIGHT_SHOULDER': 12,
    'LEFT_ELBOW': 13,    'RIGHT_ELBOW': 14,
    'LEFT_WRIST': 15,    'RIGHT_WRIST': 16,
    'LEFT_HIP': 23,      'RIGHT_HIP': 24,
    'LEFT_KNEE': 25,     'RIGHT_KNEE': 26,
    'LEFT_ANKLE': 27,    'RIGHT_ANKLE': 28,
}

def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    dot_product = np.dot(ba, bc)
    norm_product = np.linalg.norm(ba) * np.linalg.norm(bc)

    epsilon = 1e-7
    cosine_angle = dot_product / (norm_product + epsilon)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def extract_features(landmarks):
    """Extract angle features from pose landmarks"""
    if not landmarks:
        return None
    
    try:
        # Extract coordinates for each landmark
        coords = {}
        for name, idx in LANDMARKS.items():
            landmark = landmarks.landmark[idx]
            coords[name] = [landmark.x, landmark.y, landmark.z]
        
        # Calculate angles
        angles = {
            'angle_left_elbow': calculate_angle(coords['LEFT_SHOULDER'], coords['LEFT_ELBOW'], coords['LEFT_WRIST']),
            'angle_right_elbow': calculate_angle(coords['RIGHT_SHOULDER'], coords['RIGHT_ELBOW'], coords['RIGHT_WRIST']),
            'angle_left_shoulder': calculate_angle(coords['LEFT_ELBOW'], coords['LEFT_SHOULDER'], coords['LEFT_HIP']),
            'angle_right_shoulder': calculate_angle(coords['RIGHT_ELBOW'], coords['RIGHT_SHOULDER'], coords['RIGHT_HIP']),
            'angle_left_hip': calculate_angle(coords['LEFT_SHOULDER'], coords['LEFT_HIP'], coords['LEFT_KNEE']),
            'angle_right_hip': calculate_angle(coords['RIGHT_SHOULDER'], coords['RIGHT_HIP'], coords['RIGHT_KNEE']),
            'angle_left_knee': calculate_angle(coords['LEFT_HIP'], coords['LEFT_KNEE'], coords['LEFT_ANKLE']),
            'angle_right_knee': calculate_angle(coords['RIGHT_HIP'], coords['RIGHT_KNEE'], coords['RIGHT_ANKLE']),
        }
        
        return list(angles.values())
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def predict_activity(features):
    """Predict activity from features"""
    if features is None:
        return "No pose detected", 0.0
    
    try:
        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)[0]
        probabilities = model.predict_proba(features_array)[0]
        confidence = np.max(probabilities)
        
        return prediction, confidence
    except Exception as e:
        print(f"Error in prediction: {e}")
        return "Error", 0.0

def main():
    print("🎥 Starting webcam test...")
    print("Press 'q' to quit, 's' to take a screenshot")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return
    
    print("✅ Webcam initialized successfully")
    print("✅ MediaPipe pose detection ready")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            frame_count += 1
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = pose.process(image_rgb)
            
            # Convert back to BGR for OpenCV display
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            # Draw pose landmarks
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                # Extract features and predict
                features = extract_features(results.pose_landmarks)
                if features:
                    activity, confidence = predict_activity(features)
                    
                    # Display prediction on image
                    text = f"Activity: {activity}"
                    conf_text = f"Confidence: {confidence:.2f}"
                    
                    cv2.putText(image_bgr, text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(image_bgr, conf_text, (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Print to console every 30 frames
                    if frame_count % 30 == 0:
                        print(f"🎯 Detected: {activity} (confidence: {confidence:.2f})")
            else:
                cv2.putText(image_bgr, "No pose detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display FPS
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            cv2.putText(image_bgr, f"FPS: {fps:.1f}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Display the frame
            cv2.imshow('Human Activity Recognition Test', image_bgr)
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_name = f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(screenshot_name, image_bgr)
                print(f"📸 Screenshot saved as {screenshot_name}")
    
    except KeyboardInterrupt:
        print("\n⏹️ Stopping by user request...")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("🏁 Test completed")

if __name__ == "__main__":
    main()