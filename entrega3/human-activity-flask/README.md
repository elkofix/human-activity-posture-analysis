# 🤸‍♀️ Real-Time Human Activity Recognition Flask App

This Flask application provides real-time human activity recognition using MediaPipe for pose detection and a trained Random Forest model for activity classification.

## Features

- **Real-time pose detection** using MediaPipe
- **Activity classification** using a pre-trained Random Forest model
- **Web-based interface** with live video streaming
- **Real-time predictions** with confidence scores
- **Modern and responsive UI** with gradient backgrounds and animations

## Detected Activities

The model can detect the following 5 human activities with high accuracy:
- 🚶‍♂️ **Caminar hacia la cámara** - Walking towards the camera
- 🔙 **Caminar de regreso** - Walking away from the camera
- 🔄 **Girar** - Turning/rotating movement
- 💺 **Sentarse** - Sitting down movement
- 🧍 **Ponerse de pie** - Standing up movement

### Model Performance
- **Training Data**: Extracted from video sequences with pose landmark analysis
- **Algorithm**: Random Forest Classifier
- **Features**: 8 joint angles (elbows, shoulders, hips, knees)
- **Accuracy**: Optimized for real-time inference with confidence scoring
- **Real-time Processing**: Optimized for live camera feed analysis

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

### Method 1: Docker (Recommended)

#### Quick Start with Docker Compose
```bash
# Build and run the application
docker-compose up --build

# Access the app at http://localhost:5000

# Stop the application
docker-compose down
```

#### Manual Docker Commands
```bash
# Development version (Flask dev server)
docker build -t human-activity-dev .
docker run -p 5000:5000 --name activity-app human-activity-dev

# Production version (Gunicorn server)
docker build -f Dockerfile.production -t human-activity-prod .
docker run -p 5000:5000 --name activity-app-prod human-activity-prod

# Stop and cleanup
docker stop activity-app
docker rm activity-app
```

### Method 2: Direct Python execution
```bash
python app.py
```

### Method 3: Using module execution (if you have multiple Python versions)
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

## Docker Support 🐳

This application is fully containerized for easy deployment and consistent environments.

### Available Docker Configurations

1. **Development Dockerfile** (`Dockerfile`)
   - Uses Flask development server
   - Hot reloading and debugging support
   - Optimized for development workflow

2. **Production Dockerfile** (`Dockerfile.production`)
   - Uses Gunicorn WSGI server
   - Multiple worker processes
   - Production-ready with proper logging
   - Better performance and stability

3. **Docker Compose** (`docker-compose.yml`)
   - Complete orchestration setup
   - Automatic container management
   - Volume mounting for models
   - Health checks and restart policies

### Docker Features
- **Headless OpenCV**: Optimized for container environments
- **Software rendering**: No GPU dependencies required
- **Multi-architecture**: Works on x86_64 and ARM64
- **Security**: Non-root user execution
- **Health monitoring**: Built-in health checks
- **Easy scaling**: Ready for production deployment

### Quick Scripts
- **Windows**: Use `start_app.bat` or PowerShell scripts
- **Linux/Mac**: Use the provided shell scripts (see Script Usage Guide below)

## Script Usage Guide 📜

We provide convenient scripts to run the application across different platforms:

### Windows Scripts

#### 1. Batch Script (`start_app.bat`)
```cmd
# Simple double-click execution
start_app.bat

# Or from command prompt
.\start_app.bat
```

#### 2. PowerShell Script (`start_app.ps1`)
```powershell
# Run from PowerShell
.\start_app.ps1

# If execution policy prevents running:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\start_app.ps1
```

### Linux/Mac Scripts

#### 1. Main Application Script (`start_app.sh`)
```bash
# Make executable (first time only)
chmod +x start_app.sh

# Auto-detect and run (recommended)
./start_app.sh

# Specific modes
./start_app.sh docker     # Force Docker Compose
./start_app.sh python     # Force Python setup
./start_app.sh help       # Show all options
```

**Features:**
- ✅ Auto-detects Docker or Python availability
- 🔧 Creates Python virtual environment automatically
- 📦 Installs dependencies automatically
- ❌ Validates model file existence
- 🎯 Multiple execution modes

#### 2. Docker Management Script (`docker_manager.sh`)
```bash
# Make executable (first time only)
chmod +x docker_manager.sh

# Docker operations
./docker_manager.sh dev        # Development mode (Flask)
./docker_manager.sh prod       # Production mode (Gunicorn)
./docker_manager.sh compose    # Docker Compose
./docker_manager.sh build      # Build + Docker Compose

# Management operations
./docker_manager.sh logs dev   # View development logs
./docker_manager.sh logs prod  # View production logs
./docker_manager.sh logs       # View compose logs
./docker_manager.sh cleanup    # Stop and clean containers
./docker_manager.sh help       # Show all commands
```

**Features:**
- 🐳 Complete Docker container management
- 📊 Easy log viewing and monitoring
- 🧹 Automatic cleanup operations
- 🚀 Separate development and production modes
- 🔄 Container restart and rebuild capabilities

### Script Comparison

| Script | Platform | Best For | Requirements |
|--------|----------|----------|--------------|
| `start_app.bat` | Windows | Simple execution | Windows, Python |
| `start_app.ps1` | Windows | Advanced features | PowerShell |
| `start_app.sh` | Linux/Mac | Auto-detection | Bash, Docker/Python |
| `docker_manager.sh` | Linux/Mac | Docker management | Bash, Docker |

### First Time Setup (Linux/Mac)
```bash
# 1. Make scripts executable
chmod +x *.sh

# 2. Run the application
./start_app.sh

# 3. Or use Docker specifically
./docker_manager.sh compose
```

### Troubleshooting Scripts

**Permission Issues (Linux/Mac):**
```bash
# If permission denied
chmod +x start_app.sh docker_manager.sh
```

**PowerShell Execution Policy (Windows):**
```powershell
# If script execution is disabled
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Dependencies Missing:**
- The scripts will automatically detect and install missing dependencies
- For manual setup, ensure Docker or Python 3.8+ is installed

## Project Structure

```
human-activity-flask/
├── app.py                          # Main Flask application
├── config.py                       # Configuration settings
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Development Docker image
├── Dockerfile.production          # Production Docker image
├── docker-compose.yml             # Docker orchestration
├── .dockerignore                   # Docker build exclusions
├── DOCKER_README.md               # Detailed Docker documentation
├── models/                        # Directory containing the trained model
│   └── pose_classification_model.joblib
├── templates/                     # HTML templates
│   ├── index.html                # Original server-side interface
│   └── client.html               # Client-side camera interface
├── start_app.bat                 # Windows startup script
├── start_app.ps1                 # PowerShell startup script
├── start_app.sh                  # Linux/Mac startup script
└── README.md                     # This file
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
- `GET /`: Main web interface (client-side camera)
- `POST /predict`: JSON API for pose analysis and activity prediction
- Health check endpoints for Docker monitoring

**Note**: This version uses client-side camera access, so no video streaming endpoints are needed.

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

The application works best with modern browsers that support WebRTC and Canvas API:
- **Chrome 90+** (recommended) - Best performance and compatibility
- **Firefox 88+** - Full feature support
- **Safari 14+** - Good compatibility (some WebRTC limitations)
- **Edge 90+** - Full Chromium-based support

### Camera Requirements
- Modern web browsers with WebRTC support
- User permission for camera access
- Adequate lighting for pose detection
- Minimum 640x480 camera resolution

## Deployment Options

### Local Development
```bash
# Traditional Python setup
python app.py

# Docker development
docker-compose up --build
```

### Production Deployment

#### Option 1: Docker Production
```bash
# Using production Dockerfile with Gunicorn
docker build -f Dockerfile.production -t activity-recognition .
docker run -d -p 80:5000 --restart unless-stopped activity-recognition
```

#### Option 2: Reverse Proxy Setup
```bash
# With nginx reverse proxy
docker-compose -f docker-compose.production.yml up -d
```

#### Option 3: Cloud Deployment
- **AWS ECS/Fargate**: Use the production Docker image
- **Google Cloud Run**: Deploy with the containerized application
- **Azure Container Instances**: Direct Docker deployment
- **Heroku**: Use Docker deployment method

### Environment Variables

Customize the application behavior:
```bash
FLASK_ENV=production          # Set environment mode
FLASK_DEBUG=false            # Disable debug mode
FLASK_HOST=0.0.0.0          # Bind to all interfaces
FLASK_PORT=5000             # Port configuration
LIBGL_ALWAYS_SOFTWARE=1     # Force software OpenGL rendering
```

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