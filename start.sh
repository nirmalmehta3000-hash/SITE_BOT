#!/bin/bash
# Startup script for Railway deployment with UTF-8 support

# Set UTF-8 encoding
export PYTHONIOENCODING=utf-8
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

# Remove any problematic .env file if it exists
rm -f .env

# Start Streamlit with proper configuration
streamlit run app.py \
  --server.port=$PORT \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false
