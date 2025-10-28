# Use a slim Python base image
FROM python:3.12-slim

# Set working directory
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

RUN pip install uv
RUN python -m nltk.downloader stopwords

# Copy the application code
COPY . /app/

# Expose the port your Uvicorn app will run on
EXPOSE 8000

# Start the application with Uvicorn
CMD ["uv", "run", "--active", "app", "--host", "0.0.0.0", "--port", "8001"]


