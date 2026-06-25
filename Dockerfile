# Use an official Python image
FROM python:3.14-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Install uv
RUN pip install --no-cache-dir uv

# Create and set the working directory
WORKDIR /app

# Copy the project files
COPY . /app

# Install project dependencies (including dev) using uv
RUN uv sync --dev --frozen

# Default command (override in docker-compose)
CMD ["uv", "run", "-m", "alexandria"]
