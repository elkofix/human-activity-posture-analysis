# Human Activity Recognition Flask App Startup Script
# PowerShell version for better compatibility

Write-Host "🚀 Human Activity Recognition Flask App Startup" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# Check if virtual environment exists
if (-not (Test-Path "venv")) {
    Write-Host "❌ Virtual environment not found. Creating one..." -ForegroundColor Yellow
    try {
        & py -3.12 -m venv venv
        Write-Host "✅ Virtual environment created successfully" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ Failed to create virtual environment. Please check Python installation." -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# Activate virtual environment
Write-Host "🔧 Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1

# Check if requirements are installed
Write-Host "🔍 Checking dependencies..." -ForegroundColor Yellow
$mediapipeInstalled = & pip list | Select-String "mediapipe"
if (-not $mediapipeInstalled) {
    Write-Host "📦 Installing dependencies..." -ForegroundColor Yellow
    try {
        & pip install -r requirements.txt
        Write-Host "✅ Dependencies installed successfully" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ Failed to install dependencies. Please check requirements.txt" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
} else {
    Write-Host "✅ Dependencies already installed" -ForegroundColor Green
}

# Check if model file exists
if (-not (Test-Path "models\pose_classification_model.joblib")) {
    Write-Host "❌ Model file not found at models\pose_classification_model.joblib" -ForegroundColor Red
    Write-Host "Please ensure you have the trained model file in the models directory." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "✅ Model file found" -ForegroundColor Green

# Start the Flask application
Write-Host "🌐 Starting Flask application..." -ForegroundColor Green
Write-Host "Open your browser and navigate to: http://localhost:5000" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the application" -ForegroundColor Yellow
Write-Host ""

try {
    & python app.py
}
catch {
    Write-Host "❌ Error starting the application: $_" -ForegroundColor Red
}
finally {
    Write-Host ""
    Write-Host "👋 Application stopped." -ForegroundColor Cyan
    Read-Host "Press Enter to exit"
}