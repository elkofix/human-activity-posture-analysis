# Human Activity Recognition - Docker Setup

## Quick Start

### Prerequisites
- Docker installed on your system
- Docker Compose (usually included with Docker Desktop)

### Build and Run with Docker Compose

1. **Navigate to the project directory:**
   ```bash
   cd entrega3/human-activity-flask
   ```

2. **Build and start the application:**
   ```bash
   docker-compose up --build
   ```

3. **Access the application:**
   Open your browser and go to: http://localhost:5000

4. **Stop the application:**
   ```bash
   docker-compose down
   ```

### Manual Docker Commands

If you prefer to use Docker directly:

1. **Build the Docker image:**
   ```bash
   docker build -t human-activity-recognition .
   ```

2. **Run the container:**
   ```bash
   docker run -p 5000:5000 --name human-activity-app human-activity-recognition
   ```

3. **Stop and remove the container:**
   ```bash
   docker stop human-activity-app
   docker rm human-activity-app
   ```

### Production Deployment

For production deployment, you may want to:

1. **Use a reverse proxy (nginx):**
   ```yaml
   # Add to docker-compose.yml
   nginx:
     image: nginx:alpine
     ports:
       - "80:80"
     volumes:
       - ./nginx.conf:/etc/nginx/nginx.conf
     depends_on:
       - human-activity-app
   ```

2. **Set production environment variables:**
   ```bash
   export FLASK_ENV=production
   export FLASK_DEBUG=False
   ```

3. **Use a production WSGI server:**
   Update the Dockerfile CMD to use gunicorn:
   ```dockerfile
   CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "app:app"]
   ```

### Troubleshooting

- **Camera access issues:** The app uses client-side camera access, so no special Docker camera permissions are needed
- **Model loading errors:** Ensure the `models/pose_classification_model.joblib` file exists
- **Port conflicts:** Change the port mapping in docker-compose.yml if 5000 is already in use

### Docker Image Size Optimization

The current image uses `python:3.11-slim` for a good balance of functionality and size. For further optimization:

- Consider using `python:3.11-alpine` for a smaller base image
- Use multi-stage builds to reduce final image size
- Remove unnecessary system packages after installation

### Environment Variables

You can customize the application behavior using environment variables:

```bash
docker run -p 5000:5000 \
  -e FLASK_DEBUG=False \
  -e FLASK_HOST=0.0.0.0 \
  -e FLASK_PORT=5000 \
  human-activity-recognition
```