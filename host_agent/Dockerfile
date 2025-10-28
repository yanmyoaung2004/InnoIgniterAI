# Use a slim Python base image
FROM python:3.12-slim

# Set working directory at project root
WORKDIR /app

# Install system dependencies required for Poetry and Uvicorn
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml uv.lock /app/

# Upgrade pip and install Poetry
RUN pip install --upgrade pip \
    && pip install --no-cache-dir poetry \
    && poetry config virtualenvs.create false \
    && poetry install --no-root || true

# Install uv (your runtime manager)
RUN pip install uv

# Copy the application code
COPY . /app/

# Add /app/app to PYTHONPATH so "from database..." works
ENV PYTHONPATH=/app/app

# Expose port for Uvicorn
EXPOSE 8000

# Start the application with Uvicorn
# NOTE: `app.main:app` since main.py is inside app/
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
