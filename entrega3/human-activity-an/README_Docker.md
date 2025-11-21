# Human Activity & Posture Analysis - Docker Setup

## 🐳 Running with Docker

### Prerequisites
- Docker Desktop installed on your system
- Docker Compose (included with Docker Desktop)

### Quick Start

1. **Build and run the application:**
   ```bash
   docker-compose up --build
   ```

2. **Access the application:**
   Open your browser and go to `http://localhost:8501`

3. **Stop the application:**
   ```bash
   docker-compose down
   ```

### Docker Commands

#### Build the image manually:
```bash
docker build -t human-activity-app .
```

#### Run the container manually:
```bash
docker run -p 8501:8501 -v ./models:/app/models human-activity-app
```

#### View running containers:
```bash
docker ps
```

#### View logs:
```bash
docker-compose logs -f
```

### Camera Access in Docker

**Note:** Camera access in Docker containers can be complex and may require additional configuration depending on your system:

- **Linux:** You may need to pass through video devices:
  ```bash
  docker run -p 8501:8501 --device=/dev/video0 human-activity-app
  ```

- **macOS/Windows:** Camera access through Docker may be limited. Consider running the app locally for camera features.

### Environment Variables

The following environment variables are configured:
- `STREAMLIT_SERVER_PORT=8501`
- `STREAMLIT_SERVER_ADDRESS=0.0.0.0`
- `STREAMLIT_SERVER_HEADLESS=true`
- `STREAMLIT_BROWSER_GATHER_USAGE_STATS=false`

### Troubleshooting

1. **Port already in use:**
   ```bash
   # Change the port in docker-compose.yml or stop other services using port 8501
   docker-compose down
   # Edit docker-compose.yml to use a different port like "8502:8501"
   ```

2. **Model file missing:**
   Ensure `models/pose_classification_model.joblib` exists in your project directory.

3. **Permission issues on Linux:**
   ```bash
   sudo docker-compose up --build
   ```

4. **Camera not working:**
   Camera access in containerized environments requires host system permissions. For development, consider running locally:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```

### Development

For development with hot reloading, mount the source code:
```yaml
volumes:
  - .:/app
  - ./models:/app/models
```

### Production Considerations

- Use a production WSGI server for deployment
- Configure proper logging
- Set up reverse proxy (nginx) if needed
- Use environment-specific configuration files
- Consider using Docker secrets for sensitive data