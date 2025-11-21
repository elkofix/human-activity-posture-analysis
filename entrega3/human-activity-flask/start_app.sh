#!/bin/bash

# Human Activity Recognition Flask App - Linux/Mac Startup Script
# This script provides multiple ways to run the application

echo "🤸‍♀️ Human Activity Recognition Flask App"
echo "=========================================="
echo ""

# Function to check if Docker is installed
check_docker() {
    if command -v docker &> /dev/null && command -v docker-compose &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# Function to check if Python is installed
check_python() {
    if command -v python3 &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# Function to run with Docker
run_docker() {
    echo "🐳 Starting with Docker..."
    if [ "$1" == "prod" ]; then
        echo "📦 Building production image with Gunicorn..."
        docker build -f Dockerfile.production -t human-activity-prod .
        docker run -p 5000:5000 --name human-activity-app-prod human-activity-prod
    else
        echo "📦 Starting with Docker Compose (recommended)..."
        docker-compose up --build
    fi
}

# Function to run with Python
run_python() {
    echo "🐍 Starting with Python..."
    
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        echo "📦 Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    echo "🔧 Activating virtual environment..."
    source venv/bin/activate
    
    # Install dependencies
    echo "📚 Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    
    # Check if model exists
    if [ ! -f "models/pose_classification_model.joblib" ]; then
        echo "❌ Error: Model file not found!"
        echo "   Please ensure 'models/pose_classification_model.joblib' exists"
        exit 1
    fi
    
    # Start the application
    echo "🚀 Starting Flask application..."
    python app.py
}

# Function to display help
show_help() {
    echo "Usage: $0 [option]"
    echo ""
    echo "Options:"
    echo "  docker     - Run with Docker Compose (recommended)"
    echo "  docker-prod- Run with production Docker setup"
    echo "  python     - Run with Python virtual environment"
    echo "  help       - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 docker      # Run with Docker Compose"
    echo "  $0 python      # Run with Python"
    echo "  $0              # Auto-detect and run"
}

# Main script logic
case "${1:-auto}" in
    "docker")
        if check_docker; then
            run_docker
        else
            echo "❌ Docker is not installed or not available"
            echo "   Please install Docker and Docker Compose"
            exit 1
        fi
        ;;
    "docker-prod")
        if check_docker; then
            run_docker "prod"
        else
            echo "❌ Docker is not installed or not available"
            exit 1
        fi
        ;;
    "python")
        if check_python; then
            run_python
        else
            echo "❌ Python 3 is not installed or not available"
            echo "   Please install Python 3.8 or higher"
            exit 1
        fi
        ;;
    "help")
        show_help
        ;;
    "auto")
        echo "🔍 Auto-detecting best method..."
        if check_docker; then
            echo "✅ Docker detected - using Docker Compose"
            run_docker
        elif check_python; then
            echo "✅ Python detected - using Python setup"
            run_python
        else
            echo "❌ Neither Docker nor Python 3 found!"
            echo "   Please install Docker or Python 3.8+"
            exit 1
        fi
        ;;
    *)
        echo "❌ Unknown option: $1"
        show_help
        exit 1
        ;;
esac

echo ""
echo "🎉 Application should now be running at: http://localhost:5000"
echo "📱 Open your browser and allow camera access to start using the app!"