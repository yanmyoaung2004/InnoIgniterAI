# Use a slim Python base image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip

# Install FastAPI, Uvicorn, and python-multipart
RUN pip install fastapi uvicorn python-multipart

# Copy application code
COPY . /app/

# Expose the port
EXPOSE 8003

# Start the app
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8003", "--reload"]
