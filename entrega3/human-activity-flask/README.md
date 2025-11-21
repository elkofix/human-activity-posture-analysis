# 🤸‍♀️ Real-Time Human Activity Recognition Flask App

This Flask application provides real-time human activity recognition using MediaPipe for pose detection and a trained Random Forest model for activity classification.

## Features

- **Real-time pose detection** using MediaPipe
- **Activity classification** using a pre-trained Random Forest model
- **Web-based interface** with live video streaming
- **Real-time predictions** with confidence scores
- **Modern and responsive UI** with gradient backgrounds and animations

## Detected Activities

The model can detect the following activities:
- 🚶‍♀️ Walking
- 👏 Clapping
- 🤝 Meeting and splitting
- 💺 Sitting
- 🧍 Standing still
- 📱 Walking while using phone
- 📚 Walking while reading book

## Prerequisites

- Python 3.12 or higher
- Webcam connected to your computer
- Virtual environment (recommended)

## Installation and Setup

### 1. Clone or navigate to the project directory
```bash
cd entrega3/human-activity-flask
```

### 2. Create a virtual environment
```bash
# Windows
py -3.12 -m venv venv

# Linux/Mac
python3.12 -m venv venv
```

### 3. Activate the virtual environment
```bash
# Windows
./venv/Scripts/activate

# Linux/Mac
source venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Ensure the model file exists
Make sure the file `models/pose_classification_model.joblib` exists in the project directory. This is the trained model from the Jupyter notebook.

## Running the Application

### Method 1: Direct execution
```bash
python app.py
```

### Method 2: Using module execution (if you have multiple Python versions)
```bash
py -m flask run
```

## Usage

1. **Start the application**: Run the Flask app using one of the methods above
2. **Open your browser**: Navigate to `http://localhost:5000`
3. **Allow camera access**: Your browser will request permission to access your webcam
4. **View real-time predictions**: The application will show:
   - Live video stream with pose landmarks
   - Current activity prediction
   - Confidence score
   - Statistics and system status

## Project Structure

```
human-activity-flask/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── models/               # Directory containing the trained model
│   └── pose_classification_model.joblib
├── templates/           # HTML templates
│   └── index.html      # Main web interface
└── README.md           # This file
```

## How It Works

1. **Video Capture**: The application captures video from your webcam using OpenCV
2. **Pose Detection**: MediaPipe processes each frame to detect human pose landmarks
3. **Feature Extraction**: Key angles between body joints are calculated from the pose landmarks
4. **Activity Prediction**: The trained Random Forest model predicts the activity based on the extracted features
5. **Real-time Display**: Results are displayed on the web interface with confidence scores

## Technical Details

### Model Input Features
The model uses 8 angle features calculated from pose landmarks:
- Left and right elbow angles
- Left and right shoulder angles  
- Left and right hip angles
- Left and right knee angles

### API Endpoints
- `GET /`: Main web interface
- `GET /video_feed`: Video streaming endpoint
- `POST /predict`: JSON API for getting predictions
- `GET /stop_camera`: Endpoint to stop the camera

## Troubleshooting

### Common Issues

1. **Camera not detected**
   - Ensure your webcam is properly connected
   - Check if other applications are using the camera
   - Try restarting the application

2. **Model not found error**
   - Ensure `pose_classification_model.joblib` exists in the `models/` directory
   - Check the file permissions

3. **Installation issues**
   - Make sure you're using Python 3.12 or higher
   - Try upgrading pip: `pip install --upgrade pip`
   - Install dependencies one by one if bulk installation fails

4. **Performance issues**
   - Close other applications to free up system resources
   - Ensure good lighting for better pose detection
   - Check your webcam resolution settings

## Browser Compatibility

The application works best with modern browsers:
- Chrome 90+ (recommended)
- Firefox 88+
- Safari 14+
- Edge 90+

## Security Notes

- The application runs locally on your machine
- Video data is processed in real-time and not stored
- No data is sent to external servers

## Contributing

To improve this application:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is part of the Human Activity Recognition research project and follows the same licensing terms.