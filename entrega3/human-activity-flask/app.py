from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import joblib
import json
import os
import base64
from datetime import datetime
import config

app = Flask(__name__)
CORS(app)

# MediaPipe setup using config
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=config.MODEL_COMPLEXITY,
    enable_segmentation=False,
    min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
)

# Load the trained model
try:
    model = joblib.load(config.MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define landmarks indices based on the training notebook
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
    if model is None or features is None:
        return "Model not loaded", 0.0
    
    try:
        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)[0]
        probabilities = model.predict_proba(features_array)[0]
        confidence = np.max(probabilities)
        
        return prediction, confidence
    except Exception as e:
        print(f"Error in prediction: {e}")
        return "Error", 0.0

# No server-side camera needed for client-only implementation

@app.route('/')
def index():
    return render_template('client.html')



@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """Endpoint for processing client-side camera frames"""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'})
        
        # Decode base64 image
        image_data = data['image'].split(',')[1]  # Remove data:image/jpeg;base64, prefix
        image_bytes = base64.b64decode(image_data)
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image_bgr is None:
            return jsonify({'error': 'Failed to decode image'})
        
        # Convert BGR to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = pose.process(image_rgb)
        
        if results.pose_landmarks:
            features = extract_features(results.pose_landmarks)
            if features:
                activity, confidence = predict_activity(features)
                
                # Extract landmark coordinates for client-side drawing
                landmarks_data = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks_data.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    })
                
                return jsonify({
                    'activity': activity,
                    'confidence': float(confidence),
                    'timestamp': datetime.now().isoformat(),
                    'landmarks': landmarks_data,
                    'pose_detected': True
                })
        
        return jsonify({
            'pose_detected': False,
            'error': 'No pose detected',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Error in client prediction: {e}")
        return jsonify({'error': f'Processing error: {str(e)}'})



if __name__ == '__main__':
    print(f"🚀 Starting Human Activity Recognition Flask App")
    print(f"📹 Camera: {config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT}")
    print(f"🌐 Server: http://{config.FLASK_HOST}:{config.FLASK_PORT}")
    print(f"🎯 Prediction threshold: {config.PREDICTION_THRESHOLD}")
    print(f"⚡ Update interval: {config.UPDATE_INTERVAL_MS}ms")
    
    app.run(debug=config.FLASK_DEBUG, host=config.FLASK_HOST, port=config.FLASK_PORT)
