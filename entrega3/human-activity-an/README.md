# Human Activity & Posture Analysis App 🤸‍♂️

A Streamlit web application for real-time human activity and posture classification using AI.

## Features

- **Image Analysis**: Upload an image to detect and classify human activities
- **Video Analysis**: Process video files to track activities over time
- **Real-time Pose Detection**: Uses MediaPipe for accurate pose landmark detection
- **AI Classification**: Random Forest model trained on pose features with 98% accuracy
- **Interactive Visualizations**: Confidence scores, activity timelines, and pose landmarks

## Supported Activities

- 🚶 Caminando de frente (Walking forward)
- 👏 Aplaudiendo (Clapping)  
- 🔄 Girando (Turning)
- 💺 Sentandome (Sitting down)
- 🏃 Levantarme (Standing up)

## How It Works

### 1. Pose Detection
- Uses **MediaPipe** to detect 33 pose landmarks in images/videos
- Extracts key joint positions (shoulders, elbows, hips, knees, etc.)

### 2. Feature Engineering
- Calculates 8 joint angles from pose landmarks:
  - Left/Right elbow angles
  - Left/Right shoulder angles  
  - Left/Right hip angles
  - Left/Right knee angles

### 3. Activity Classification
- **Random Forest Classifier** trained on extracted angle features
- Model achieves **98% accuracy** on test dataset
- Provides confidence scores for all activity classes

## File Structure

```
human-activity-an/
├── app.py              # Main Streamlit application
├── run_app.py          # Launcher script
├── models/             # Trained model files
│   └── pose_classification_model.joblib
└── README.md           # This file
```

## Model Information

- **Algorithm**: Random Forest Classifier
- **Input Features**: 8 joint angles extracted from MediaPipe pose landmarks
- **Training Data**: Human Activity Recognition Video Dataset from Kaggle
- **Performance**: 98% accuracy on test set
- **Optimization**: GridSearchCV with cross-validation

## Usage Tips

1. **For best results with images:**
   - Use clear images with good lighting
   - Person should be fully visible in the frame
   - Avoid cluttered backgrounds when possible

2. **For video analysis:**
   - Short videos work best (app processes max 100 frames for demo)
   - Supported formats: MP4, AVI, MOV
   - Person should be the main subject in the video

3. **Troubleshooting:**
   - If pose detection fails, try images/videos with better lighting
   - Ensure the person is clearly visible and not occluded
   - Check that all dependencies are installed correctly

## Technical Details

The app implements the same preprocessing pipeline as the Jupyter notebook:

1. **MediaPipe Pose Detection**: Extracts 33 pose landmarks
2. **Angle Calculation**: Computes joint angles using vector mathematics
3. **Feature Vector**: Creates 8-dimensional feature vector
4. **Classification**: Uses trained Random Forest model for prediction
5. **Visualization**: Displays results with confidence scores and pose overlays

## Requirements

- Python 3.8+
- Streamlit
- OpenCV
- MediaPipe
- scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn

See `requirements.txt` for exact versions.