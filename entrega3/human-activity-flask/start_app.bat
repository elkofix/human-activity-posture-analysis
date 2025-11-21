@echo off
REM Startup script for Human Activity Recognition Flask App
REM This script sets up the environment and starts the Flask application

echo 🚀 Human Activity Recognition Flask App Startup
echo ================================================

REM Check if virtual environment exists
if not exist "venv\" (
    echo ❌ Virtual environment not found. Creating one...
    py -3.12 -m venv venv
    if errorlevel 1 (
        echo ❌ Failed to create virtual environment. Please check Python installation.
        pause
        exit /b 1
    )
    echo ✅ Virtual environment created successfully
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if requirements are installed
echo 🔍 Checking dependencies...
pip list | findstr "mediapipe" >nul
if errorlevel 1 (
    echo 📦 Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ Failed to install dependencies. Please check requirements.txt
        pause
        exit /b 1
    )
    echo ✅ Dependencies installed successfully
) else (
    echo ✅ Dependencies already installed
)

REM Check if model file exists
if not exist "models\pose_classification_model.joblib" (
    echo ❌ Model file not found at models\pose_classification_model.joblib
    echo Please ensure you have the trained model file in the models directory.
    pause
    exit /b 1
)

echo ✅ Model file found

REM Start the Flask application
echo 🌐 Starting Flask application...
echo Open your browser and navigate to: http://localhost:5000
echo Press Ctrl+C to stop the application
echo.

python app.py

echo.
echo 👋 Application stopped. Press any key to exit.
pause >nul