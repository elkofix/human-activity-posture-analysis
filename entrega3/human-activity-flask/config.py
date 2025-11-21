# Configuration file for Human Activity Recognition Flask App
# Modify these settings to customize the application behavior

# Flask Configuration
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = True

# Camera Configuration
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_INDEX = 0  # Usually 0 for built-in webcam, 1 for external

# MediaPipe Configuration
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
MODEL_COMPLEXITY = 1  # 0 for light, 1 for full, 2 for heavy

# Prediction Configuration
PREDICTION_THRESHOLD = 0.5  # Minimum confidence to display prediction
UPDATE_INTERVAL_MS = 1000   # How often to update predictions (milliseconds)

# Model Configuration
MODEL_PATH = "models/pose_classification_model.joblib"

# Activity Labels (for translation to Spanish)
ACTIVITY_TRANSLATIONS = {
    "clapping": "👏 Aplaudiendo",
    "meeting_and_splitting": "🤝 Encontrándose y separándose", 
    "sitting": "💺 Sentado",
    "standing_still": "🧍 De pie quieto",
    "walking": "🚶‍♀️ Caminando",
    "walking_while_reading_book": "📚 Caminando leyendo libro",
    "walking_while_using_phone": "📱 Caminando usando teléfono"
}