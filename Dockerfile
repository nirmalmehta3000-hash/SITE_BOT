# Use Python 3.11 slim image for better compatibility
FROM python:3.11-slim

# Set UTF-8 encoding environment variables
ENV PYTHONIOENCODING=utf-8
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install system dependencies including locale support
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    locales \
    && rm -rf /var/lib/apt/lists/* \
    && locale-gen C.UTF-8 \
    && update-locale LANG=C.UTF-8

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files (excluding .env and other sensitive files)
COPY . .
# Remove .env file if it exists to prevent encoding issues
RUN rm -f .env

# Make startup script executable
RUN chmod +x start.sh

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Expose port (Railway will set the PORT environment variable)
EXPOSE 8501

# Use startup script for proper UTF-8 handling
CMD ["./start.sh"]
