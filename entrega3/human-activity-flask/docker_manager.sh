#!/bin/bash

# Docker Management Script for Human Activity Recognition App
# Provides easy Docker operations for development and production

echo "🐳 Docker Management - Human Activity Recognition"
echo "==============================================="
echo ""

# Function to build and run development version
dev_mode() {
    echo "🔧 Development Mode - Flask Dev Server"
    echo "Building development image..."
    docker build -t human-activity-dev .
    echo "Running development container..."
    docker run --rm -p 5000:5000 --name human-activity-dev human-activity-dev
}

# Function to build and run production version
prod_mode() {
    echo "🚀 Production Mode - Gunicorn Server"
    echo "Building production image..."
    docker build -f Dockerfile.production -t human-activity-prod .
    echo "Running production container..."
    docker run --rm -p 5000:5000 --name human-activity-prod human-activity-prod
}

# Function to use docker-compose
compose_mode() {
    echo "🎼 Docker Compose Mode"
    if [ "$1" == "build" ]; then
        echo "Building and starting with Docker Compose..."
        docker-compose up --build
    else
        echo "Starting with Docker Compose..."
        docker-compose up
    fi
}

# Function to stop and clean up containers
cleanup() {
    echo "🧹 Cleaning up Docker containers and images..."
    
    # Stop containers
    docker stop human-activity-dev 2>/dev/null || true
    docker stop human-activity-prod 2>/dev/null || true
    docker-compose down 2>/dev/null || true
    
    # Remove containers
    docker rm human-activity-dev 2>/dev/null || true
    docker rm human-activity-prod 2>/dev/null || true
    
    # Optional: Remove images (uncomment if needed)
    # docker rmi human-activity-dev 2>/dev/null || true
    # docker rmi human-activity-prod 2>/dev/null || true
    
    echo "✅ Cleanup completed!"
}

# Function to show logs
show_logs() {
    if [ "$1" == "dev" ]; then
        docker logs -f human-activity-dev
    elif [ "$1" == "prod" ]; then
        docker logs -f human-activity-prod
    else
        docker-compose logs -f
    fi
}

# Function to display help
show_help() {
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  dev        - Run development mode (Flask dev server)"
    echo "  prod       - Run production mode (Gunicorn server)"
    echo "  compose    - Run with Docker Compose"
    echo "  build      - Build and run with Docker Compose"
    echo "  logs       - Show logs (dev|prod|compose)"
    echo "  cleanup    - Stop containers and clean up"
    echo "  help       - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 dev            # Development mode"
    echo "  $0 prod           # Production mode"
    echo "  $0 compose        # Docker Compose"
    echo "  $0 build          # Build + Docker Compose"
    echo "  $0 logs dev       # Show development logs"
    echo "  $0 cleanup        # Clean up containers"
}

# Main script logic
case "${1:-compose}" in
    "dev")
        dev_mode
        ;;
    "prod")
        prod_mode
        ;;
    "compose")
        compose_mode
        ;;
    "build")
        compose_mode "build"
        ;;
    "logs")
        show_logs "$2"
        ;;
    "cleanup")
        cleanup
        ;;
    "help")
        show_help
        ;;
    *)
        echo "❌ Unknown command: $1"
        show_help
        exit 1
        ;;
esac