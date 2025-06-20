# File: src/p1_api_deployment/Dockerfile

# Start from a slim Python base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy ONLY the P1 API's requirements file into the container
COPY ./fraud_api_lambda/requirements.txt .

# Install only the P1 API's dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the P1 API source code (api.py, model, etc.) into the container
COPY ./fraud_api_lambda/ .

# Tell Docker that the container will listen on port 8000
EXPOSE 8000

# The command to run when the container starts.
# This starts the Uvicorn web server to serve our FastAPI application.
CMD ["python", "-m", "uvicorn", "api_k8s:app", "--host", "0.0.0.0", "--port", "8000"]